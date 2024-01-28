
from wilson_maze_env.envs import WilsonMazeEnv


def get_adjacent_cells(position: tuple, env: WilsonMazeEnv):
    """
        Gets the adjacent cells of a position in a maze.

        Args:
            position: The position.
            env: The environment.

        Returns:
            The adjacent cells of the position.
    """
    adjacent_cells = []
    for movement, direction in env.get_action_to_direction_map().values():
        if not env.is_way_blocked(position, direction):
            new_position = tuple(position + movement)
            if not env.is_position_invalid(new_position):
                adjacent_cells.append(new_position)
    return adjacent_cells

def find_shortest_path_bfs(start: tuple, end: tuple, env: WilsonMazeEnv):
    """
        Finds the shortest path between two points in a maze using BFS.

        Args:
            start: The starting position.
            end: The ending position.
            env: The environment.
        
        Returns:
            The shortest path between the start and end points.
    """
    queue = [[start]]
    visited = set()

    while queue:
        path = queue.pop(0)
        node = path[-1]

        if node == end:
            return path

        if tuple(node) not in visited:
            for adjacent in get_adjacent_cells(node, env):
                new_path = list(path)
                new_path.append(adjacent)
                queue.append(new_path)

            visited.add(tuple(node))
    
    return None