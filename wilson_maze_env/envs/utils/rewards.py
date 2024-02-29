import time
import numpy as np

from wilson_maze_env.envs import WilsonMazeEnv
from .MazeCell import MazeCell

def manhattan_distance(a, b):
    return np.sum(np.abs(np.array(a) - np.array(b))).item()

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
        reward = -1.0
        truncated = True
    elif env.is_position_invalid(new_agent_pos):
        env.out_of_bounds += 1
        reward = -0.8
    elif env.is_way_blocked(env.agent_pos, direction):
        env.wall_collisions += 1
        reward = -0.75
    elif env.terminate_on_wrong_target and env.check_if_wrong_target():
        reward = -1.0
        terminated = True
    elif np.array_equal(new_agent_pos, env.current_target_pos):
        reward = 1.0 + 2.0 * ((env.timelimit - (current_time - env.t0)) / env.timelimit)
        terminated = True
    else:
        # erase agent from current position
        current_value = env.maze[env.agent_pos[0]][env.agent_pos[1]].value
        env.change_cell_value_at_position(env.agent_pos, current_value - MazeCell.AGENT_VALUE)

        # change agent position
        env.agent_pos = new_agent_pos

        # update maze map
        current_value = env.maze[env.agent_pos[0]][env.agent_pos[1]].value
        env.change_cell_value_at_position(env.agent_pos, MazeCell.AGENT_VALUE + current_value)

        reward = -0.05 - manhattan_distance(env.agent_pos, env.current_target_pos) / (env.size * np.sqrt(2))

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
        reward = -1.0
        truncated = True
    elif env.is_position_invalid(new_agent_pos):
        env.out_of_bounds += 1
        reward = -0.8
    elif env.is_way_blocked(env.agent_pos, direction):
        env.wall_collisions += 1
        reward = -0.75
    elif env.terminate_on_wrong_target and env.check_if_wrong_target():
        reward = -3.0
        terminated = True
    elif np.array_equal(new_agent_pos, env.current_target_pos):
        reward = 1.0
        terminated = True
    else:
        # erase agent from current position
        current_value = env.maze[env.agent_pos[0]][env.agent_pos[1]].value
        env.change_cell_value_at_position(env.agent_pos, current_value - MazeCell.AGENT_VALUE)

        # change agent position
        env.agent_pos = new_agent_pos

        # update maze map
        current_value = env.maze[env.agent_pos[0]][env.agent_pos[1]].value
        env.change_cell_value_at_position(env.agent_pos, MazeCell.AGENT_VALUE + current_value)

        if tuple(new_agent_pos) in env.visited_cells:
            reward = -0.25
        elif tuple(new_agent_pos) not in env._shortest_path:
            reward = -0.15
        else:
            reward = -0.05

    return reward, terminated, truncated

def pick_up_coin(agent_pos: tuple[int, int], pick_up_coins: True, env: WilsonMazeEnv) -> float:
    """
        Reward the agent for picking up a coin if pick_up_coins is True.
        Penalize the agent for picking up a coin if pick_up_coins is False.
        Double penalize the agent for trying to pick up a coin when there is no coin at the current position.
    """

    if env.maze[agent_pos[0]][agent_pos[1]].value == MazeCell.COIN_VALUE + MazeCell.AGENT_VALUE:
        env.maze[agent_pos[0]][agent_pos[1]].value -= MazeCell.COIN_VALUE
        if pick_up_coins:
            return 0.3
        return -0.3

    return -0.6