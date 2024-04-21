import io
import struct
import time
import math
from typing import Any, SupportsFloat, List

import numpy as np
import pandas as pd
import pygame

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType, ActType, RenderFrame

from wilson_maze_env.envs.utils.MazeCell import MazeCell
from wilson_maze_env.envs.utils.bfs import find_shortest_path_bfs
from wilson_maze_env.envs.utils.generate_svg import write_svg
from wilson_maze_env.envs.utils.random_walks import random_walk
from wilson_maze_env.envs.utils.rewards import BAD_PICK_UP_COIN_REWARD, GOOD_PICK_UP_COIN_REWARD, calculate_reward_bounded_basic, calculate_reward_manhattan, pick_up_coin

maze_char_map = {
    '-': 0,
    '|': 1,
    '+': 2,
    ' ': 3,
    '\n': 4,
    'A': 5,
    'T': 6
}

WINDOW_HEIGHT, WINDOW_WIDTH = 500, 500
ACTION_SIZE = 4
ACTION_SIZE_WITH_COIN = 5


class WilsonMazeEnv(gym.Env):
    metadata = {"render_modes": [
        "human", "text", "rgb_array"], "render_fps": 120}

    TARGET_IDS = {
        "triangle": 0,
        "square": 1,
        "circle": 2,
        "diamond": 3,
    }
    PICK_UP_COINS_ACTION = 4

    def __init__(self, render_mode="text", size=7, timelimit=60, random_seed=42,
                 number_of_targets=4, add_coins=True, variable_target=False, terminate_on_wrong_target=False,
                 labels: np.ndarray = False, prompts: np.ndarray = None, prompt_mean=False,  prompt_size=0,
                 chosen_prompt=None, user_prompt=None, prompts_path=None, labels_path=None, embd_size=4096,
                 max_train_size=None, reward_type='basic'):
        """
            Wilson's maze environment for reinforcement learning.
            The agent is placed in the middle of the maze and has to reach the target.
            The agent can move in 4 directions: up, down, left, right.

            Args:
                size (int): The size of the maze. The maze will be size x size.
                timelimit (int): The time limit for the episode.
                random_seed (int): The random seed to use for the environment.
                add_coins (bool): Whether to add coins to the maze.
                number_of_targets (int): The number of targets to reach. Can be 1, 2, 3 or 4.
                target_id (int): The id of the target to reach. Can be 0, 1, 2 or 3.
                variable_target (bool): Whether to change the target if certain conditions are met.
                terminate_on_wrong_target (bool): Whether to terminate the episode if the agent reaches a wrong target.
                labels (np.ndarray): The labels to use for the environment. If add_coins is True, the labels must be a 2D array
                prompts (np.nddaray): The prompts to use.
                prompt_mean (bool): Whether to use the mean of the prompts.
                prompt_size (int): The size of the prompt to use.
                user_prompt (np.ndarray): The prompt to use for the current episode given directly by the user.
                chosen_prompt (int): The id of the prompt to use.
                prompts_path (str): The path to the prompts embeddings file.
                labels_path (str): The path to the labels file.
                embd_size (int): The size of the embeddings.
                max_train_size (int): The maximum size of the training set.
                reward_type (str): The type of reward to use. Can be 'basic' or 'manhattan'.
        """
        self.size = size
        self.timelimit = timelimit
        self.random_seed = random_seed
        self.add_coins = add_coins
        self.padding = 10
        self.reward_type = reward_type
        self.terminate_on_wrong_target = terminate_on_wrong_target
        self.time_resets = []
        self.time_steps = []
        self.variable_target = variable_target

        # assert labels is not None, 'Labels must be provided'
        # assert user_prompt is not None or prompts is not None, 'Either user_prompt or prompts_file must be provided'
        # assert len(labels.shape) == 2, 'Labels must be a 2D array'
        # if user_prompt is None:
        #     assert labels.shape[0] == prompts.shape[0], 'The number of prompts must be the same as the number of should_pickup_coins'
        # assert not (
        #     variable_target and chosen_prompt is not None), 'Cannot use variable_target and chosen_prompt at the same time'
        # assert not add_coins or labels.shape[1] == 2, 'Labels must have 2 columns if add_coins is True'

        self.current_prompt = None
        self.prompt_size = prompt_size
        if self.prompt_size:
            self.prompt_mean = prompt_mean
            self.prompts = None
            self.user_prompt = user_prompt
            self.current_prompt = chosen_prompt
            self.chosen_prompt = chosen_prompt

            if prompts_path is not None and labels_path is not None:
                prompts, labels = self._get_input_data(
                    prompts_path, labels_path, embd_size)
                if max_train_size is not None:
                    prompts = prompts[:max_train_size]
                    labels = labels[:max_train_size]

            if prompts is not None:
                if prompt_mean:
                    if prompts.shape[1] % self.prompt_size != 0:
                        factor = math.ceil(prompts.shape[1] / self.prompt_size)
                        total_size = factor * self.prompt_size
                        prompts = np.pad(prompts, [(0, 0), (0, total_size - prompts.shape[1])], mode='constant')
                        
                    prompts = np.mean(
                        np.split(prompts, prompts.shape[1] // self.prompt_size, axis=1), axis=0)
                else:
                    prompts = prompts[:, :self.prompt_size]

                self.prompts = prompts
            elif prompt_mean:
                if user_prompt.shape[0] % self.prompt_size != 0:
                    factor = math.ceil(user_prompt.shape[0] / self.prompt_size)
                    total_size = factor * self.prompt_size
                    user_prompt = np.pad(user_prompt, (0, total_size - user_prompt.shape[0]), mode='constant')
                self.user_prompt = np.mean(np.split(user_prompt, user_prompt.shape[0] // self.prompt_size, axis=0),
                                           axis=0)
            else:
                self.user_prompt = self.user_prompt[:self.prompt_size]

        """
            The following dictionary maps abstract actions from `self.action_space` to
            the direction we will walk in if that action is taken.
            I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: (np.array([1, 0]), 'E'),
            1: (np.array([0, 1]), 'S'),
            2: (np.array([-1, 0]), 'W'),
            3: (np.array([0, -1]), 'N'),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
            If human-rendering is used, `self.window` will be a reference
            to the window that we draw to. `self.clock` will be a clock that is used
            to ensure that the environment is rendered at the correct framerate in
            human-mode. They will remain `None` until human-mode is used for the
            first time.
        """
        self.window = None
        self.clock = None

        # Pad the maze all around by this amount.
        self.window_height = WINDOW_HEIGHT
        self.window_width = WINDOW_WIDTH

        # aspect_ratio = self.size / self.size
        # # Height and width of the maze image (excluding paddingding), in pixels
        # self.window_width = int(self.window_height * aspect_ratio)

        self.maze: list[list[MazeCell]] = None
        self.t0 = None
        self.out_of_bounds = None
        self.wall_collisions = None
        self.good_picked_up_coins = 0
        self.bad_picked_up_coins = 0

        # Agent and target positions
        self.agent_pos = (size // 2, size // 2)
        self.triangle_target_pos = (0, 0)
        self.circle_target_pos = (size - 1, size - 1)
        self.square_target_pos = (0, size - 1)
        self.diamond_target_pos = (size - 1, 0)
        self.targets_positions = []
        for target in [self.triangle_target_pos, self.square_target_pos, self.circle_target_pos,
                       self.diamond_target_pos]:
            self.targets_positions.append(np.array(target))

        self.targets = labels[:, 0]
        self.should_pickup_coins = labels[:, 1]
        self.number_of_targets = number_of_targets

        # Set the target and the current target position
        self.target_id = labels[0][0]
        self.pick_up_coins = labels[0][1]
        self.current_target_pos = self._get_current_target(self.target_id)

        # Set the wrong targets map
        self._wrong_targets_map = {
            self.triangle_target_pos: [self.square_target_pos, self.circle_target_pos, self.diamond_target_pos],
            self.square_target_pos: [self.triangle_target_pos, self.circle_target_pos, self.diamond_target_pos],
            self.circle_target_pos: [self.square_target_pos, self.triangle_target_pos, self.diamond_target_pos],
            self.diamond_target_pos: [
                self.square_target_pos, self.circle_target_pos, self.triangle_target_pos]
        }

        self.steps = 0
        self.score = 0

        self._generate_base_maze(self.random_seed)
        self._shortest_paths = {}
        for i in range(len(self.targets_positions)):
            self._shortest_paths[i] = find_shortest_path_bfs(
                self.agent_pos, self._get_current_target(i), self)
            assert self._shortest_paths[i], f'No path found from agent to target {i}'

        # assert len(self._shortest_paths) == self.number_of_targets, 'The number of shortest paths must be the same as the number of targets'

        # Set coins
        if self.add_coins:
            self.coins = []
            self.initial_coins = {i: 0 for i in range(self.number_of_targets)}

            for i in range(len(self.targets_positions)):
                for j, node in enumerate(self._shortest_paths[i]):
                    if not np.any(np.all(node == np.vstack(self.targets_positions), axis=1)) and j % 2 == 1:
                        self.coins.append(np.array(node))
                        self.initial_coins[i] += 1

        # The observation space is a 2D array of size (size, size)
        self.observation_shape = (self.size, self.size)

        # The observations space is a 1D array of size:
        # - prompt_size if prompt_size is not 0
        # - agent position = (2)
        # - target positions = (2 * 4)
        # - coins positions and values = (3 * number_of_coins)
        obs_space_size = prompt_size + 10
        if add_coins:
            obs_space_size += len(self.coins) * 3
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(obs_space_size,), dtype=np.float64)

        # We have 4 moving actions, corresponding to "right", "up", "left", "down"
        # and a "pick up coin" action if add_coins is True
        self.action_space = spaces.Discrete(
            ACTION_SIZE if not add_coins else ACTION_SIZE_WITH_COIN)

        self.reset_maze_values()

    @staticmethod
    def _get_input_data(prompts_path: str, labels_path: str, embd_size=4096) -> tuple[np.ndarray, np.ndarray]:
        """
            Load the input data from the dataset and the embeddings file.

            :param labels_path: the path to the dataset file
            :param prompts_path: the path to the prompts embeddings file
            :param embd_size: the size of the embeddings
            :return: a tuple of two numpy arrays, the first one is the input embedding data and the second one is the target data
        """
        df = pd.read_csv(labels_path, sep=',')
        targets = df['target'].to_numpy()
        coins = df['coin'].to_numpy()
        Y = np.stack([targets, coins], axis=1)

        with open(prompts_path, 'rb') as f:
            data = f.read()
            embds = struct.unpack('f' * int(len(data) / 4), data)
            X = np.vstack([np.array(embds[i:i + embd_size])
                           for i in range(0, len(embds), embd_size)])

        return X, Y

    def get_action_to_direction_map(self):
        return self._action_to_direction

    def set_user_prompt(self, user_prompt):
        self.user_prompt = user_prompt

    def get_number_of_targets(self):
        return self.number_of_targets

    def _get_current_target(self, target_id: int):
        if target_id == 0:
            return self.triangle_target_pos
        elif target_id == 1:
            return self.square_target_pos
        elif target_id == 2:
            return self.circle_target_pos
        else:
            return self.diamond_target_pos

    def _choose_random_cell(self) -> tuple[int, int]:
        return np.random.randint(0, self.size), np.random.randint(0, self.size)

    def is_position_invalid(self, position) -> bool:
        return position[0] <= -1 or position[0] >= self.size or position[1] <= -1 or position[1] >= self.size

    def is_way_blocked(self, position: tuple[int, int], direction: str) -> bool:
        return not self.maze[position[0]][position[1]].can_go_direction(direction)

    def _get_initial_coins(self):
        return self.initial_coins[self.target_id]

    def _get_current_coins(self):
        current_coins = 0
        for node in self._shortest_paths[self.target_id]:
            if self.maze[node[0]][node[1]].coin:
                current_coins += 1
        return current_coins

    def _generate_base_maze(self, seed=42) -> None:
        """
            Generating the base maze with Wilson's algorithm.

            Args:
                seed (int): The random seed to use for the environment.
        """
        np.random.seed(seed)
        # generate initial maze with empty cells
        self.maze = [[MazeCell() for _ in range(self.size)]
                     for _ in range(self.size)]

        # choose first random cell
        x, y = self._choose_random_cell()
        self.maze[x][y].already_belogns_to_maze = True

        remaining_cells = self.size * self.size - 1
        while remaining_cells > 0:
            # choose random walk starting point
            x, y = self._choose_random_cell()
            if self.maze[x][y].already_belogns_to_maze:
                continue

            remaining_cells -= random_walk(x, y, self)

    def _add_coins_to_maze(self):
        """
            Add coins to the maze if add_coins is True.
        """
        if self.add_coins:
            for coin in self.coins:
                self.maze[coin[0]][coin[1]].coin = True

    def reset_maze_values(self):
        for i in range(self.size):
            for j in range(self.size):
                self.maze[i][j].visited = False

        self.maze[self.agent_pos[0]][self.agent_pos[1]].agent = True
        for i, target in enumerate(self.targets_positions):
            self.maze[target[0]][target[1]].target = i

        self._add_coins_to_maze()

    def _encode_maze_str(self) -> np.ndarray:
        return np.array([maze_char_map[c] for c in self.__str__()])

    def __str__(self):
        """
            Return a (crude) string representation of the maze.
            Original at https://scipython.com/blog/making-a-maze/.
        """
        maze_rows = ['-' * (self.size * 2 + 1)]
        for y in range(self.size):
            maze_row = ['|']
            for x in range(self.size):
                if self.maze[x][y].walls['E']:
                    if self.maze[x][y].agent:
                        maze_row.append('A|')
                    elif self.maze[x][y].target != -1:
                        maze_row.append('T|')
                    else:
                        maze_row.append(' |')
                else:
                    if self.maze[x][y].agent:
                        maze_row.append('A ')
                    elif self.maze[x][y].target != -1:
                        maze_row.append('T ')
                    else:
                        maze_row.append('  ')
            maze_rows.append(''.join(maze_row))
            maze_row = ['|']
            for x in range(self.size):
                if self.maze[x][y].walls['S']:
                    maze_row.append('-+')
                else:
                    maze_row.append(' +')
            maze_rows.append(''.join(maze_row))
        return '\n'.join(maze_rows)

    def _get_obs(self):
        """
            Get the observation of the environment.
            The observation is a concatenation of the agent position, the target position, the coins positions and values and
            the prompt if prompt_size is not 0 (in inverse order), where each position is normalized by the maze size.
        """

        coins_obs = []
        if self.add_coins:
            for coin in self.coins:
                coins_obs.append(np.array(coin) / (self.size - 1))
                if self.maze[coin[0]][coin[1]].coin:
                    coins_obs.append(1)
                else:
                    coins_obs.append(0)
            coins_obs = np.hstack(coins_obs)

        target_obs = np.hstack([np.array(target_pos) / (self.size - 1)
                               for target_pos in self.targets_positions])
        non_prompt_obs = np.hstack(
            [coins_obs, target_obs, np.array(self.agent_pos) / (self.size - 1)])

        if self.prompt_size == 0:
            return non_prompt_obs

        if self.user_prompt is not None and isinstance(self.user_prompt, np.ndarray):
            prompt = self.user_prompt
        else:
            prompt = self.prompts[self.current_prompt]

        return np.hstack([prompt, non_prompt_obs])

    def _change_target(self):
        self.target_id = self.np_random.integers(
            0, self.number_of_targets, 1)[0]
        self.current_target_pos = self._get_current_target(self.target_id)

    def _choose_prompt_for_variable_target(self):
        idx_for_target = np.array(
            np.where(self.targets == self.target_id)).flatten()
        self.current_prompt = self.np_random.choice(idx_for_target, 1)[0]

    def _change_target_prompt(self):
        if self.prompt_size and self.chosen_prompt is None and self.user_prompt is None:
            if self.variable_target:
                self._choose_prompt_for_variable_target()
            else:
                self.current_prompt = self.np_random.integers(
                    0, self.prompts.shape[0], 1)[0]

            if self.add_coins:
                self.pick_up_coins = self.should_pickup_coins[self.current_prompt]

    def action_masks(self) -> List[bool]:
        actions_mask = []
        for action in range(self.action_space.n):
            if action != self.__class__.PICK_UP_COINS_ACTION:
                _, direction = self._action_to_direction[action]
                actions_mask.append(
                    self.maze[self.agent_pos[0]][self.agent_pos[1]].can_go_direction(direction))
            elif self.maze[self.agent_pos[0]][self.agent_pos[1]].coin:
                actions_mask.append(True)
            else:
                actions_mask.append(False)
        return actions_mask

    def check_if_wrong_target(self):
        for wrong_target in self._wrong_targets_map[self.current_target_pos]:
            if np.array_equal(wrong_target, self.agent_pos):
                return True
        return False

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.score = 0
        self.steps = 0
        super().reset(seed=seed)
        self.reset_maze_values()

        self.agent_pos = (self.size // 2, self.size // 2)
        self.maze[self.agent_pos[0]][self.agent_pos[1]].agent = True
        self.maze[self.agent_pos[0]][self.agent_pos[1]].visited = True
        if self.variable_target:
            self._change_target()
        self._change_target_prompt()

        self.out_of_bounds = 0
        self.wall_collisions = 0
        self.good_picked_up_coins = 0
        self.bad_picked_up_coins = 0

        if self.render_mode != "text":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_width + 2 * self.padding,
                 self.window_height + 2 * self.padding)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            self.render()

        observation = self._get_obs()
        self.t0 = time.time()

        return observation, {'target': self.target_id, 'prompt': self.current_prompt}

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # calculate new agent position
        # start = time.time()
        self.maze[self.agent_pos[0]][self.agent_pos[1]].visited = True

        if action != self.__class__.PICK_UP_COINS_ACTION:
            movement, direction = self._action_to_direction[action]
            new_agent_pos = self.agent_pos + movement

            if self.reward_type == 'basic':
                reward, terminated, truncated = calculate_reward_bounded_basic(
                    new_agent_pos, direction, self)
            else:
                reward, terminated, truncated = calculate_reward_manhattan(
                    new_agent_pos, direction, self)
        else:
            if self.add_coins == False:
                raise ValueError(
                    'Coins cannot be picked up if add_coins is False')

            truncated, terminated = False, False
            reward = pick_up_coin(self.agent_pos, self.pick_up_coins, self)

            if reward == GOOD_PICK_UP_COIN_REWARD:
                self.good_picked_up_coins += 1
            elif reward == BAD_PICK_UP_COIN_REWARD:
                self.bad_picked_up_coins += 1

        if self.render_mode == "human":
            self.render()

        observation = self._get_obs()

        self.steps += 1
        self.score += reward

        info = {'out_of_bounds': self.out_of_bounds,
                'wall_collisions': self.wall_collisions,
                'target': self.target_id, 'prompt': self.current_prompt}

        if self.add_coins:
            info['good_pickup_coins'] = self.good_picked_up_coins
            info['bad_pickup_coins'] = self.bad_picked_up_coins

            if self.chosen_prompt is not None or self.user_prompt is not None:
                if self.pick_up_coins:
                    info['coins_wins'] = 1 - \
                        (self._get_current_coins() / self._get_initial_coins())
                else:
                    info['coins_wins'] = self._get_current_coins() / \
                        self._get_initial_coins()

        # self.time_steps.append(time.time() - start)
        return observation, reward, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode != "text":
            # The following lines creates a new image of the appropriate size and fills it with white
            screen = pygame.Surface(
                (self.window_width + 2 * self.padding, self.window_height + 4 * self.padding))
            screen.fill((255, 255, 255))

            # The following lines draws the maze onto the image
            svg_frame = write_svg(self, filename=None)
            img_frame = pygame.image.load(io.BytesIO(svg_frame.encode()))
            screen.blit(img_frame, img_frame.get_rect(
                center=screen.get_rect().center))

            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(screen, screen.get_rect())
            pygame.event.pump()

            if self.render_mode == "human":
                pygame.display.flip()

                # We need to ensure that human-rendering occurs at the predefined frame rate.
                # The following line will automatically add a delay to keep the frame rate stable.
                self.clock.tick(self.metadata["render_fps"])
            elif self.render_mode == "rgb_array":
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
                )

        return None

    def _restart_pygame(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.clock = None
            self.window = None

    def close(self):
        super().close()
        self._restart_pygame()


if __name__ == '__main__':
    # seeds for 9: 613, 313
    X = np.random.randn(3000, 2560)
    Y = np.ones((3000, 2))
    Y[2][0] = 0
    Y[3][0] = 0
    Y[4][0] = 2
    Y[5][0] = 2
    Y[6][0] = 3
    Y[7][0] = 3

    _labels = [0, 1]
    _user_prompt = X[20]

    env = WilsonMazeEnv(render_mode="human", size=9, timelimit=30, random_seed=283,
                        add_coins=True, prompt_size=1024, labels=np.array([_labels]),
                        variable_target=False, user_prompt=_user_prompt, prompt_mean=True)
    # user_prompt=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))

    obs, info = env.reset()
    # for i in range(env.size):
    #     for j in range(env.size):
    #         print(env.maze[i][j].value, end=' ')
    #     print()

    targets = {0: 0, 1: 0, 2: 0, 3: 0}

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs.shape, reward, terminated, truncated, info)
        if terminated or truncated or i % 15 == 0:
            obs, info = env.reset()
        targets[info['target']] += 1

    print(targets)

    env.close()
