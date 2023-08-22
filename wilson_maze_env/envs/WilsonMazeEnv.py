import io
import math
import os
import time
from typing import Any, SupportsFloat, List
from collections import defaultdict

import PIL.Image
import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType, ActType, RenderFrame


def generate_random_walk_move():
    p = max(np.random.uniform() - 0.001, 0)
    return math.floor(4 * p)


def manhattan_distance(a, b):
    return np.sum(np.abs(np.array(a) - np.array(b))).item()


opposite_directions = {
    'N': 'S',
    'S': 'N',
    'W': 'E',
    'E': 'W'
}

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


class MazeCell:
    def __init__(self, value):
        self.value = value
        self.walls = {'N': 1, 'S': 1, 'W': 1, 'E': 1}

    def knock_wall(self, wall):
        self.walls[wall] = 0

    def can_go_direction(self, direction):
        return not self.walls[direction]

    def get_state(self):
        state = list(self.walls.values())
        return np.array(state, dtype=np.uint8)


class WilsonMazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "text", "rgb_array"], "render_fps": 120}

    def __init__(self, render_mode="text", size=7, timelimit=60, random_seed=42, obs_type='mlp', target_id=0,
                 variable_target=False, number_of_targets=4, random_target_on_step=False, prompts_file=None,
                 prompt_size=0, user_prompt=None, prompt_mean=False, terminate_on_wrong_target=False, reward_type='basic'):
        self.width = size
        self.height = size
        self.timelimit = timelimit
        self.random_seed = random_seed
        self.obs_type = obs_type
        self.padding = 10
        self.reward_type = reward_type
        self.random_target_on_step = random_target_on_step
        self.terminate_on_wrong_target = terminate_on_wrong_target
        self.time_resets = []
        self.time_steps = []

        if variable_target:
            assert prompts_file is not None
        self.number_of_targets = number_of_targets

        self.prompt_size = prompt_size
        self.prompt_mean = prompt_mean
        self.prompts = None
        self.current_prompt = user_prompt
        self.user_prompt = user_prompt
        if prompts_file is not None:
            prompts_data = np.load(prompts_file)
            self.prompts, self.number_of_prompts = {}, {}
            for k in prompts_data.files:
                if target_id == int(k[-1]) and not variable_target:
                    if self.prompt_size:
                        if prompt_mean:
                            prompts = prompts_data[k]
                            assert prompts.shape[1] % self.prompt_size == 0
                            prompts = np.mean(np.split(prompts, prompts.shape[1] // self.prompt_size), axis=0).tolist()
                        else:
                            prompts = prompts_data[k][:, :self.prompt_size].tolist()
                    else:
                        prompts = prompts_data[k].tolist()
                    self.prompts[k] = prompts
                    self.number_of_prompts[k] = len(prompts)

        self.observation_shape = (self.width, self.height)
        if self.obs_type == 'mlp':
            self.observation_space = spaces.Box(-np.inf, np.inf,
                                                shape=(prompt_size + 10,), dtype=np.float64)
        elif self.obs_type == 'cnn':
            self.observation_space = spaces.Box(0, 255, shape=(
                3, WINDOW_HEIGHT + 2 * self.padding, WINDOW_WIDTH + 2 * self.padding), dtype=np.uint8)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(ACTION_SIZE)

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

        # aspect_ratio = self.width / self.height
        # # Height and width of the maze image (excluding paddingding), in pixels
        # self.window_width = int(self.window_height * aspect_ratio)

        self.maze = None
        self.t0 = None
        self.out_of_bounds = None
        self.wall_collisions = None

        self.agent_pos = (size // 2, size // 2)
        self.triangle_target_pos = (0, 0)
        self.circle_target_pos = (size - 1, size - 1)
        self.square_target_pos = (0, size - 1)
        self.diamond_target_pos = (size - 1, 0)
        self.targets_positions = []
        for target in [self.triangle_target_pos, self.circle_target_pos, self.square_target_pos,
                       self.diamond_target_pos]:
            self.targets_positions.append(np.array(target) / (self.width - 1))

        self.target_id = target_id
        self.current_target_pos = self._get_current_target(target_id)
        self.variable_target = variable_target

        self._wrong_targets_map = {
            self.triangle_target_pos: [self.square_target_pos, self.circle_target_pos, self.diamond_target_pos],
            self.square_target_pos: [self.triangle_target_pos, self.circle_target_pos, self.diamond_target_pos],
            self.circle_target_pos: [self.square_target_pos, self.triangle_target_pos, self.diamond_target_pos],
            self.diamond_target_pos: [self.square_target_pos, self.circle_target_pos, self.triangle_target_pos]
        }

        self.visited_cells = set()
        self.steps = 0
        self.score = 0

        self._generate_base_maze(self.random_seed)
    
    def number_of_targets(self) -> int:
        return self.number_of_targets
    
    def number_of_prompts_for_target(self, target_id: int) -> int:
        return self.number_of_prompts[target_id]

    def _get_current_target(self, target_id: int):
        if target_id == 0:
            return self.triangle_target_pos
        elif target_id == 1:
            return self.square_target_pos
        elif target_id == 2:
            return self.circle_target_pos
        else:
            return self.diamond_target_pos

    def _choose_random_cell(self):
        return np.random.randint(0, self.width), np.random.randint(0, self.height)

    def _is_position_invalid(self, position):
        return position[0] <= -1 or position[0] >= self.width or position[1] <= -1 or position[1] >= self.height

    def _change_cell_value_at_position(self, position, new_value):
        self.maze[position[0]][position[1]].value = new_value

    def _is_way_blocked(self, position, direction):
        return not self.maze[position[0]][position[1]].can_go_direction(direction)

    def _random_walk(self, x_start, y_start):
        """
            Generate random walk. The walk stops when we found a cell that already belongs to the maze.
            We need to keep track of the walk directions for the erase-loop step.
        :param x: walk starting location x coord
        :param y: walk staring locations y cord
        :return: None
        """
        directions = {}
        x, y = x_start, y_start
        while True:
            random_direction = generate_random_walk_move()
            movement, direction = self._action_to_direction[random_direction]
            directions[f'{x}-{y}'] = (direction, movement)
            last_x, last_y = x, y
            x, y = (x, y) + movement

            if self._is_position_invalid((x, y)):
                x, y = last_x, last_y
                continue

            if self.maze[x][y].value == 0:
                # we've reached a cell that already belongs to the maze paths
                x, y = x_start, y_start
                new_cells = 0
                while self.maze[x][y].value != 0:
                    new_cells += 1
                    # create cells and break walls
                    self.maze[x][y].value = 0
                    dir_, mov = directions[f'{x}-{y}']
                    self.maze[x][y].knock_wall(dir_)

                    x, y = (x, y) + mov

                    # we need to break walls for going back
                    self.maze[x][y].knock_wall(opposite_directions[dir_])
                self.maze[x][y].knock_wall(opposite_directions[direction])

                return new_cells
            elif self.maze[x][y].value == -2:
                directions[f'{x}-{y}'] = (direction, movement)
            else:
                self.maze[x][y].value = -2

    def _generate_base_maze(self, seed=42):
        """
            Generating the base maze with Wilson's algorithm
        """
        np.random.seed(seed)
        # generate initial maze with empty cells
        self.maze = [[MazeCell(-1) for _ in range(self.height)] for _ in range(self.width)]

        # choose first random cell
        x, y = self._choose_random_cell()
        self.maze[x][y].value = 0

        remaining_cells = self.width * self.height - 1
        while remaining_cells > 0:
            # choose random walk starting point
            x, y = self._choose_random_cell()
            if self.maze[x][y].value == 0:
                continue

            remaining_cells -= self._random_walk(x, y)

    def _encode_maze_str(self):
        return np.array([maze_char_map[c] for c in self.__str__()])

    def __str__(self):
        """
            Return a (crude) string representation of the maze.
            Original at https://scipython.com/blog/making-a-maze/.
        """
        maze_rows = ['-' * (self.width * 2 + 1)]
        for y in range(self.height):
            maze_row = ['|']
            for x in range(self.width):
                if self.maze[x][y].walls['E']:
                    if self.maze[x][y].value == 1:
                        maze_row.append('A|')
                    elif self.maze[x][y].value == 10:
                        maze_row.append('T|')
                    else:
                        maze_row.append(' |')
                else:
                    if self.maze[x][y].value == 1:
                        maze_row.append('A ')
                    elif self.maze[x][y].value == 10:
                        maze_row.append('T ')
                    else:
                        maze_row.append('  ')
            maze_rows.append(''.join(maze_row))
            maze_row = ['|']
            for x in range(self.width):
                if self.maze[x][y].walls['S']:
                    maze_row.append('-+')
                else:
                    maze_row.append(' +')
            maze_rows.append(''.join(maze_row))
        return '\n'.join(maze_rows)

    def write_svg(self, filename):
        """Write an SVG image of the maze to filename.
            Original at https://scipython.com/blog/making-a-maze/.
        """

        # Scaling factors mapping maze coordinates to image coordinates
        scy, scx = self.window_height / self.height, self.window_width / self.width

        svg_string = io.StringIO()

        def get_cell_box_coordinates(position):
            return position[0] * scx, position[1] * scy, (position[0] + 1) * scx, (position[1] + 1) * scy

        def join_points(points):
            return ','.join([f'{p[0]},{p[1]}' for p in points])

        def write_wall(ww_f, ww_x1, ww_y1, ww_x2, ww_y2):
            """Write a single wall to the SVG string handle ww_f."""
            print('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="5"/>'
                  .format(ww_x1, ww_y1, ww_x2, ww_y2), file=ww_f)

        def write_circle(ww_f, cell_position, small=False):
            x1, y1, x2, y2 = get_cell_box_coordinates(cell_position)
            cell_size = x2 - x1
            factor = 10 if small else 3
            radius = max(5, cell_size / factor)
            print(
                f'<circle cx="{x1 + cell_size // 2}" cy="{y1 + cell_size // 2}" r="{radius}" fill="red" stroke="blue" stroke-width="2"/>',
                file=ww_f)

        def write_square(ww_f, cell_position, small=False):
            x1, y1, x2, y2 = get_cell_box_coordinates(cell_position)
            cell_size = x2 - x1
            factor = 10 if small else 5
            length = max(5, 3 * cell_size / factor)
            padding = cell_size / 5
            print(
                f'<rect x="{x1 + padding}" y="{y1 + padding}" width="{length}" height="{length}" fill="yellow" stroke="green" stroke-width="2"/>',
                file=ww_f)

        def write_triangle(ww_f, cell_position, small=False):
            x1, y1, x2, y2 = get_cell_box_coordinates(cell_position)
            cell_size = x2 - x1
            padding = cell_size / 10
            factor = 10 if small else 5
            length = 4 * cell_size / factor
            A = (x1 + cell_size / 2, y1 + padding)
            B = (A[0] - length / 2, y2 - padding)
            C = (A[0] + length / 2, y2 - padding)
            print(
                f'<polygon points="{join_points([A, B, C])}" fill="#03e8fc" stroke="red" stroke-width="2"/>',
                file=ww_f)

        def write_diamond(ww_f, position, small=False):
            x1, y1, x2, y2 = get_cell_box_coordinates(position)
            cell_size = x2 - x1
            factor = 10 if small else 5
            length = max(5, 4 * cell_size / factor)
            padding = cell_size / 10
            A = (x1 + length / 2 + padding, y1 + padding)
            B = (x1 + length + padding, y1 + length / 2 + padding)
            C = (x1 + length / 2 + padding, y1 + length + padding)
            D = (x1 + padding, y1 + length / 2 + padding)
            print(
                f'<polygon points="{join_points([A, B, C, D])}" fill="purple" stroke="orange" stroke-width="2"/>',
                file=ww_f)

        def write_cross(ww_f, position):
            x1, y1, x2, y2 = get_cell_box_coordinates(position)
            cell_size = x2 - x1
            length = max(5, 2 * cell_size / 10)
            padding = self.padding if self.padding <= (x2 - x1) / 10 else 5
            A = (x1 + (cell_size - length) / 2, y1 + padding)
            B = (A[0] + length, y1 + padding)
            C = (B[0], y1 + (cell_size - length) / 2)
            D = (x2 - padding, y1 + (cell_size - length) / 2)
            E = (x2 - padding, D[1] + length)
            F = (C[0], D[1] + length)
            G = (F[0], y2 - padding)
            H = (A[0], y2 - padding)
            I = (A[0], E[1])
            J = (x1 + padding, E[1])
            K = (x1 + padding, C[1])
            L = (A[0], C[1])
            print(
                f'<polygon points="{join_points([A, B, C, D, E, F, G, H, I, J, K, L])}" fill="blue" stroke="red" stroke-width="2"/>',
                file=ww_f)

        # Write the SVG image file for maze
        # SVG preamble and styles.
        print('<?xml version="1.0" encoding="utf-8"?>', file=svg_string)
        print('<svg xmlns="http://www.w3.org/2000/svg"', file=svg_string)
        print('    xmlns:xlink="http://www.w3.org/1999/xlink"', file=svg_string)
        print('    width="{:d}" height="{:d}" viewBox="{} {} {} {}">'
              .format(self.window_width + 2 * self.padding, self.window_height + 2 * self.padding,
                      -self.padding, -self.padding, self.window_width + 2 * self.padding,
                      self.window_height + 2 * self.padding),
              file=svg_string)

        # Draw the "South" and "East" walls of each cell, if present (these
        # are the "North" and "West" walls of a neighbouring cell in
        # general, of course).
        for x in range(self.width):
            for y in range(self.height):
                if self.maze[x][y].walls['S']:
                    x1, y1, x2, y2 = x * scx, (y + 1) * scy, (x + 1) * scx, (y + 1) * scy
                    write_wall(svg_string, x1, y1, x2, y2)
                if self.maze[x][y].walls['E']:
                    x1, y1, x2, y2 = (x + 1) * scx, y * scy, (x + 1) * scx, (y + 1) * scy
                    write_wall(svg_string, x1, y1, x2, y2)
        # Draw the North and West maze border, which won't have been drawn
        # by the procedure above.
        print('<line x1="0" y1="0" x2="{}" y2="0" stroke="black" stroke-width="5"/>'.format(self.window_width),
              file=svg_string)
        print('<line x1="0" y1="0" x2="0" y2="{}" stroke="black" stroke-width="5"/>'.format(self.window_height),
              file=svg_string)

        # Draw agent and targets
        write_triangle(svg_string, self.triangle_target_pos)
        write_circle(svg_string, self.circle_target_pos)
        write_square(svg_string, self.square_target_pos)
        write_diamond(svg_string, self.diamond_target_pos)
        write_cross(svg_string, self.agent_pos)

        if self.target_id == 0:
            write_triangle(svg_string, self.agent_pos)
        elif self.target_id == 1:
            write_square(svg_string, self.agent_pos)
        elif self.target_id == 2:
            write_circle(svg_string, self.agent_pos)
        else:
            write_diamond(svg_string, self.agent_pos)

        # close svg file
        print('</svg>', file=svg_string)

        svg_string_content = svg_string.getvalue()
        svg_string.close()

        # with open(os.path.join(os.path.dirname(__file__), filename), 'w') as f:
        #     print(svg_string_content, file=f)

        return svg_string_content

    def _render_frame(self):
        screen = pygame.Surface((self.window_width + 2 * self.padding, self.window_height + 4 * self.padding))
        screen.fill((255, 255, 255))

        svg_frame = self.write_svg('../test.svg')
        img_frame = pygame.image.load(io.BytesIO(svg_frame.encode()))

        screen.blit(img_frame, img_frame.get_rect(center=screen.get_rect().center))

        if self.render_mode != "text":
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

    def _get_obs_cnn(self):
        states = []
        for j in range(self.height):
            row = []
            for i in range(self.width):
                row.append(self.maze[i][j].get_state())
            states.append(np.vstack(row))
        return np.stack(states)

    def _get_obs_cnn_frame(self):
        if self.render_mode == "human":
            filename = "frame.png"
            pygame.image.save(self.window, filename)
            frame = PIL.Image.open(os.path.join(os.path.dirname(__file__), filename))
        else:
            frame = PIL.Image.new("RGB", (self.window_width + 2 * self.padding, self.window_height + 2 * self.padding),
                                  (255, 255, 255))
            svg = pygame.image.load(io.BytesIO(self.write_svg('').encode()))
            svg = PIL.Image.frombytes('RGBA', (WINDOW_WIDTH + 2 * self.padding, WINDOW_HEIGHT + 2 * self.padding),
                                      pygame.image.tostring(svg, 'RGBA'))
            frame.paste(svg, mask=svg.split()[3], box=(0, 10))
        prompt = self.prompts[f'arr_{self.target_id}'][self.current_prompt]
        return {'prompt': prompt, 'frame': np.asarray(frame).transpose(2, 0, 1)}

    def _get_obs_mlp(self):
        if self.prompt_size:
            prompt = self.prompts[f'arr_{self.target_id}'][self.current_prompt]
            return np.hstack([prompt, np.hstack(self.targets_positions), np.array(self.agent_pos) / (self.width - 1)])
        return np.hstack([np.hstack(self.targets_positions), np.array(self.agent_pos) / (self.width - 1)])

    def _get_obs(self):
        if self.obs_type == 'mlp':
            return self._get_obs_mlp()
        elif self.obs_type == 'cnn':
            return self._get_obs_cnn_frame()
        return self._encode_maze_str()

    def _change_target(self):
        self.visited_cells = set(tuple(self.agent_pos))
        # new_target_id = self.target_id
        # while new_target_id == self.target_id:
        #     new_target_id = np.random.choice([0, 1, 2, 3][:self.number_of_targets], 1)[0]
        # self.target_id = new_target_id
        self.target_id = self.np_random.choice([0, 1, 2, 3][:self.number_of_targets], 1)[0]
        if self.user_prompt is None:
            self.current_target_pos = self._get_current_target(self.target_id)

    def _change_target_prompt_prompt(self):
        if self.user_prompt is None:
            self.current_prompt = self.np_random.integers(0, self.number_of_prompts[f'arr_{self.target_id}'], 1)[0]

    def action_masks(self) -> List[bool]:
        actions_mask = []
        for action in range(self.action_space.n):
            _, direction = self._action_to_direction[action]
            actions_mask.append(self.maze[self.agent_pos[0]][self.agent_pos[1]].can_go_direction(direction))
        return actions_mask

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.score = 0
        self.steps = 0
        super().reset(seed=seed)

        self.agent_pos = (self.width // 2, self.height // 2)
        if self.variable_target:
            self._change_target()
        else:
            self.visited_cells.clear()
            self.visited_cells.add(self.agent_pos)
        #start = time.time()
        self._change_target_prompt_prompt()
        #self.time_resets.append(time.time() - start)

        self.out_of_bounds = 0
        self.wall_collisions = 0

        if self.render_mode != "text":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_width + 2 * self.padding, self.window_height + 2 * self.padding)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            self.render()

        observation = self._get_obs()
        self.t0 = time.time()

        return observation, {'out_of_bounds': self.out_of_bounds,
                             'wall_collisions': self.wall_collisions,
                             'target': self.target_id, 'prompt': self.current_prompt}

    def _check_if_wrong_target(self):
        for wrong_target in self._wrong_targets_map[self.current_target_pos]:
            if np.array_equal(wrong_target, self.agent_pos):
                return True
        return False

    def calculate_reward_manhattan(self, new_agent_pos, direction):
        reward, terminated, truncated = 0.0, False, False
        current_time = time.time()

        if current_time - self.t0 > self.timelimit:
            reward = -1.0
            truncated = True
        elif self._is_position_invalid(new_agent_pos):
            self.out_of_bounds += 1
            reward = -0.8
        elif self._is_way_blocked(self.agent_pos, direction):
            self.wall_collisions += 1
            reward = -0.75
        elif self.terminate_on_wrong_target and self._check_if_wrong_target():
            reward = -1.0
            terminated = True
        elif np.array_equal(new_agent_pos, self.current_target_pos):
            reward = 1.0 + 2.0 * ((self.timelimit - (current_time - self.t0)) / self.timelimit)
            terminated = True
        elif current_time - self.t0 > self.timelimit or self.score < -0.5 * self.width:
            reward = -1.0
            truncated = True
        else:
            # erase agent from current position
            self._change_cell_value_at_position(self.agent_pos, 0)

            # change agent position
            self.agent_pos = new_agent_pos

            # update maze map
            self._change_cell_value_at_position(self.agent_pos, 1)

            reward = -0.05 - manhattan_distance(self.agent_pos, self.current_target_pos) / (self.width * np.sqrt(2))

        return reward, terminated, truncated

    def calculate_reward_bounded_basic(self, new_agent_pos, direction):
        reward, terminated, truncated = 0.0, False, False
        current_time = time.time()

        if current_time - self.t0 > self.timelimit:
            reward = -1.0
            truncated = True
        elif self._is_position_invalid(new_agent_pos):
            self.out_of_bounds += 1
            reward = -0.8
        elif self._is_way_blocked(self.agent_pos, direction):
            self.wall_collisions += 1
            reward = -0.75
        elif self.terminate_on_wrong_target and self._check_if_wrong_target():
            reward = -1.0
            terminated = True
        elif np.array_equal(new_agent_pos, self.current_target_pos):
            reward = 1.0
            terminated = True
        else:
            # erase agent from current position
            self._change_cell_value_at_position(self.agent_pos, 0)

            # change agent position
            self.agent_pos = new_agent_pos

            # update maze map
            self._change_cell_value_at_position(self.agent_pos, 1)

            if tuple(new_agent_pos) in self.visited_cells:
                reward = -0.25
            else:
                reward = -0.05

        return reward, terminated, truncated

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # calculate new agent position
        # start = time.time()
        self.visited_cells.add(tuple(self.agent_pos))

        movement, direction = self._action_to_direction[action]
        new_agent_pos = self.agent_pos + movement

        if self.reward_type == 'basic':
            reward, terminated, truncated = self.calculate_reward_bounded_basic(new_agent_pos, direction)
        else:
            reward, terminated, truncated = self.calculate_reward_manhattan(new_agent_pos, direction)

        if self.render_mode == "human":
            self.render()

        observation = self._get_obs()

        # if self.variable_target and self.random_target_on_step and self.steps >= self.width * np.sqrt(
        #         2) and self.np_random.uniform(0, 1) >= 0.75:
        #     self._change_target()
        #     self._change_target_prompt_prompt()
        #
        # if self.prompt_size and self.np_random.uniform(0, 1) >= 0.75:
        #     self._change_target_prompt_prompt()

        self.steps += 1
        self.score += reward

        # self.time_steps.append(time.time() - start)
        return observation, reward, terminated, truncated, {'out_of_bounds': self.out_of_bounds,
                             'wall_collisions': self.wall_collisions,
                             'target': self.target_id, 'prompt': self.current_prompt}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode != "text":
            return self._render_frame()

    def _restart_pygame(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.clock = None
            self.window = None

    def close(self):
        super().close()
        # print(len(self.time_resets))
        # print(np.mean(np.array(self.time_resets)))
        # print(np.mean(np.array(self.time_steps)))
        self._restart_pygame()


if __name__ == '__main__':
    from sb3_contrib.common.maskable.utils import get_action_masks

    env = WilsonMazeEnv(render_mode="text", size=7, timelimit=30, random_seed=42, obs_type='mlp', prompt_size=512,
                        variable_target=True, prompts_file='../../../prompts/small_dataset/prompts.npz')

    targets = {0: 0, 1: 0, 2: 0, 3: 0}
    prompts = defaultdict(int)
    observation, info = env.reset()
    targets[info['target']] += 1
    r = 0
    t0 = time.time()
    for i in range(3 * 10**5):
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)
        prompts[info['prompt']] += 1
        targets[info['target']] += 1
        #print(observation[-10:], reward, terminated, truncated, get_action_masks(env))
        r += reward
        if terminated or truncated:
            observation, info = env.reset()
            targets[info['target']] += 1
            # print(r)
            r = 0
    env.close()
    print(time.time() - t0)

    total = sum(targets.values())
    dist = {k: v / total for k, v in sorted(targets.items())}
    print(dist)
    # print(prompts)
    #print(len(prompts.keys()))
