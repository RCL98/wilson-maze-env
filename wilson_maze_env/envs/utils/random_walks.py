import math
import numpy as np

from wilson_maze_env.envs import WilsonMazeEnv

opposite_directions = {
    'N': 'S',
    'S': 'N',
    'W': 'E',
    'E': 'W'
}


def generate_random_walk_move() -> int:
    """
        Generate a random walk move.

        Returns:
            The random walk move.
    """
    p = max(np.random.uniform() - 0.001, 0)
    return math.floor(4 * p)

def random_walk(x_start, y_start, env: WilsonMazeEnv) -> int:
        """
            Generate random walk. The walk stops when we found a cell that already belongs to the maze.
            We need to keep track of the walk directions for the erase-loop step.

            Args:
                x_start: The starting x coordinate.
                y_start: The starting y coordinate.
                env: The environment.
            
            Returns:
                The number of new cells added to the maze.
        """
        directions = {}
        x, y = x_start, y_start
        while True:
            random_direction = generate_random_walk_move()
            movement, direction = env._action_to_direction[random_direction]
            directions[f'{x}-{y}'] = (direction, movement)
            last_x, last_y = x, y
            x, y = (x, y) + movement

            if env.is_position_invalid((x, y)):
                x, y = last_x, last_y
                continue

            if env.maze[x][y].value == 0:
                # we've reached a cell that already belongs to the maze paths
                x, y = x_start, y_start
                new_cells = 0
                while env.maze[x][y].value != 0:
                    new_cells += 1
                    # create cells and break walls
                    env.maze[x][y].value = 0
                    dir_, mov = directions[f'{x}-{y}']
                    env.maze[x][y].knock_wall(dir_)

                    x, y = (x, y) + mov

                    # we need to break walls for going back
                    env.maze[x][y].knock_wall(opposite_directions[dir_])
                env.maze[x][y].knock_wall(opposite_directions[direction])

                return new_cells
            elif env.maze[x][y].value == -2:
                directions[f'{x}-{y}'] = (direction, movement)
            else:
                env.maze[x][y].value = -2