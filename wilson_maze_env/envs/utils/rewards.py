import time
import numpy as np

from wilson_maze_env.envs import WilsonMazeEnv
from .MazeCell import MazeCell

GOOD_PICK_UP_COIN_REWARD = 0.5
BAD_PICK_UP_COIN_REWARD = -0.5
NO_COIN_REWARD = -0.85

REACH_TARGET_REWARD = 1.0
ALL_COINS_REWARD = 3.0

OUT_OF_TIME_PENALTY = -1.0
OUT_OF_BOUNDS_PENALTY = -0.8
WALL_COLLISION_PENALTY = -0.75
WRONG_TARGET_PENALTY = -3.0

MOVE_TO_VISITED_CELL_PENALTY = -0.25
MOVE_TO_NON_SHORTEST_PATH_CELL_PENALTY = -0.15
MOVE_PENALTY = -0.05

def manhattan_distance(a, b):
    return np.sum(np.abs(np.array(a) - np.array(b))).item()

def reward_for_finishing(env: WilsonMazeEnv) -> float:
    if not env.add_coins:
        return REACH_TARGET_REWARD
    
    current_coins = env._get_current_coins()
    init_coins = env._get_initial_coins()
    if env.pick_up_coins:
        if current_coins == 0:
            reward = ALL_COINS_REWARD
        else:
            reward = REACH_TARGET_REWARD
    else:
        if current_coins == init_coins:
            reward = REACH_TARGET_REWARD
        else:
            reward = ALL_COINS_REWARD
    
    return reward

def calculate_reward_manhattan(new_agent_pos: tuple[int, int], direction: str, env: WilsonMazeEnv) -> tuple[float, bool, bool]:
    """
        If the agent reaches the target, it receives a reward of 1.0 + 2.0 * ((timelimit - (current_time - t0)) / timelimit).
        The agent receives a reward of -0.8 if it tries to go out of the maze.
        The agent receives a reward of -0.75 if it tries to go through a wall.
        The agent receives a reward of -0.05 - manhattan_distance / (size * sqrt(2)) if it moves to a cell that it has already visited.

        Args:
            new_agent_pos: The new agent position.
            direction: The direction of the move.
            env: The environment.
        
        Returns:
            The reward, whether the episode terminated and whether the episode was truncated.
    """
    reward, terminated, truncated = 0.0, False, False
    current_time = time.time()

    if current_time - env.t0 > env.timelimit:
        reward = OUT_OF_TIME_PENALTY
        truncated = True
    elif env.is_position_invalid(new_agent_pos):
        env.out_of_bounds += 1
        reward = OUT_OF_BOUNDS_PENALTY
    elif env.is_way_blocked(env.agent_pos, direction):
        env.wall_collisions += 1
        reward = WALL_COLLISION_PENALTY
    elif env.terminate_on_wrong_target and env.check_if_wrong_target():
        reward = WRONG_TARGET_PENALTY
        terminated = True
    elif np.array_equal(new_agent_pos, env.current_target_pos):
        reward = reward_for_finishing(env)
        reward = reward + 2.0 * ((env.timelimit - (current_time - env.t0)) / env.timelimit)
        terminated = True
    else:
        # erase agent from current position
        env.maze[env.agent_pos[0]][env.agent_pos[1]].agent = False

        # change agent position
        env.agent_pos = new_agent_pos

        # update maze map
        env.maze[env.agent_pos[0]][env.agent_pos[1]].agent = True

        reward = MOVE_PENALTY - manhattan_distance(env.agent_pos, env.current_target_pos) / (env.size * np.sqrt(2))

    return reward, terminated, truncated

def calculate_reward_bounded_basic(new_agent_pos: tuple[int, int], direction: str, env: WilsonMazeEnv) -> tuple[float, bool, bool]:
    """
        The agent receives a reward of 1 when it reaches the target and -1 if it runs out of time.
        The agent receives a reward of -0.8 if it tries to go out of the maze.
        The agent receives a reward of -0.75 if it tries to go through a wall.
        The agent receives a reward of -3.0 if it reaches the wrong target.
        The agent receives a reward of -0.05 if it moves to a cell that it has already visited.
        The agent receives a reward of -0.15 if it moves to a cell that is not on the shortest path to the target.
        The agent receives a reward of -0.25 if it moves to a cell that it already visited.

        Args:
            new_agent_pos: The new agent position.
            direction: The direction of the move.
            env: The environment.
        
        Returns:
            The reward, whether the episode terminated and whether the episode was truncated.
    """
    reward, terminated, truncated = 0.0, False, False
    current_time = time.time()

    if current_time - env.t0 > env.timelimit:
        reward = OUT_OF_TIME_PENALTY
        truncated = True
    elif env.is_position_invalid(new_agent_pos):
        env.out_of_bounds += 1
        reward = OUT_OF_BOUNDS_PENALTY
    elif env.is_way_blocked(env.agent_pos, direction):
        env.wall_collisions += 1
        reward = WALL_COLLISION_PENALTY
    elif env.terminate_on_wrong_target and env.check_if_wrong_target():
        reward = WRONG_TARGET_PENALTY
        terminated = True
    elif np.array_equal(new_agent_pos, env.current_target_pos):
        reward = reward_for_finishing(env)
        terminated = True
    else:
         # erase agent from current position
        env.maze[env.agent_pos[0]][env.agent_pos[1]].agent = False

        # change agent position
        env.agent_pos = new_agent_pos

        # update maze map
        env.maze[env.agent_pos[0]][env.agent_pos[1]].agent = True

        if env.maze[env.agent_pos[0]][env.agent_pos[1]].visited:
            reward = MOVE_TO_VISITED_CELL_PENALTY
        elif env.maze[env.agent_pos[0]][env.agent_pos[1]].short_path == env.target_id:
            reward = MOVE_TO_NON_SHORTEST_PATH_CELL_PENALTY
        else:
            reward = MOVE_PENALTY

    return reward, terminated, truncated

def pick_up_coin(agent_pos: tuple[int, int], pick_up_coins: True, env: WilsonMazeEnv) -> float:
    """
        Reward the agent for picking up a coin if pick_up_coins is True.
        Penalize the agent for picking up a coin if pick_up_coins is False.
        Double penalize the agent for trying to pick up a coin when there is no coin at the current position.
    """

    if env.maze[agent_pos[0]][agent_pos[1]].coin:
        env.maze[agent_pos[0]][agent_pos[1]].coin = False
        if pick_up_coins:
            return GOOD_PICK_UP_COIN_REWARD
        return BAD_PICK_UP_COIN_REWARD

    return NO_COIN_REWARD