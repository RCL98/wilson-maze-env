import io
import os

from wilson_maze_env.envs import WilsonMazeEnv
from wilson_maze_env.envs.utils.MazeCell import MazeCell

def get_cell_box_coordinates(position: tuple[int, int], scx: float, scy: float):
            return position[0] * scx, position[1] * scy, (position[0] + 1) * scx, (position[1] + 1) * scy

def join_points(points: list[tuple[int, int]]):
    return ','.join([f'{p[0]},{p[1]}' for p in points])

def write_wall(svg_string, x1, y1, x2, y2):
    """
        Write a single wall to the SVG string handle svg_string.

        Args:
            svg_string: The SVG string.
            x1: The x coordinate of the first point.
            y1: The y coordinate of the first point.
            x2: The x coordinate of the second point.
            y2: The y coordinate of the second point.
    """
    print('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="5"/>'
            .format(x1, y1, x2, y2), file=svg_string)

def write_circle(svg_string, cell_position: tuple[int, int], scx: float, scy: float, small=False):
    """
        Write a circle to the SVG string handle svg_string at the given position.

        Args:
            svg_string: The SVG string.
            cell_position: The position.
            scx: The scaling factor for the x coordinate.
            scy: The scaling factor for the y coordinate.
            small: Whether the circle should be small.
    """
    x1, y1, x2, y2 = get_cell_box_coordinates(cell_position, scx, scy)
    
    # caclulate necessary values
    cell_size = x2 - x1
    factor = 10 if small else 3
    radius = max(5, cell_size / factor)
    
    print(
        f'<circle cx="{x1 + cell_size // 2}" cy="{y1 + cell_size // 2}" r="{radius}" fill="red" stroke="blue" stroke-width="2"/>',
        file=svg_string)

def write_square(svg_string, cell_position: tuple[int, int], scx: float, scy: float, small=False):
    """
        Write a square to the SVG string handle svg_string at the given position.

         Args:
            svg_string: The SVG string.
            cell_position: The position.
            scx: The scaling factor for the x coordinate.
            scy: The scaling factor for the y coordinate.
            small: Whether the square should be small.
    """
    x1, y1, x2, y2 = get_cell_box_coordinates(cell_position, scx, scy)
    
    # caclulate necessary values
    cell_size = x2 - x1
    factor = 10 if small else 5
    length = max(5, 3 * cell_size / factor)
    padding = cell_size / 5
    
    print(
        f'<rect x="{x1 + padding}" y="{y1 + padding}" width="{length}" height="{length}" fill="yellow" stroke="green" stroke-width="2"/>',
        file=svg_string)

def write_triangle(svg_string, cell_position: tuple[int, int], scx: float, scy: float, small=False):
    """
        Write a triangle to the SVG string handle svg_string at the given position.

        Args:
            svg_string: The SVG string.
            cell_position: The position.
            scx: The scaling factor for the x coordinate.
            scy: The scaling factor for the y coordinate.
            small: Whether the triangle should be small.
    """
    x1, y1, x2, y2 = get_cell_box_coordinates(cell_position, scx, scy)
    
    # caclulate necessary values
    cell_size = x2 - x1
    padding = cell_size / 10
    factor = 10 if small else 5
    length = 4 * cell_size / factor
    
    # triangle points
    A = (x1 + cell_size / 2, y1 + padding)
    B = (A[0] - length / 2, y2 - padding)
    C = (A[0] + length / 2, y2 - padding)
    
    print(
        f'<polygon points="{join_points([A, B, C])}" fill="#03e8fc" stroke="red" stroke-width="2"/>',
        file=svg_string)

def write_diamond(svg_string, cell_position: tuple[int, int], scx: float, scy: float, small=False):
    """
        Write a diamond to the SVG string handle svg_string at the given position.

        Args:
            svg_string: The SVG string.
            cell_position: The position.
            scx: The scaling factor for the x coordinate.
            scy: The scaling factor for the y coordinate.
            small: Whether the diamond should be small.
    """
    x1, y1, x2, y2 = get_cell_box_coordinates(cell_position, scx, scy)
    
    # caclulate necessary values
    cell_size = x2 - x1
    factor = 10 if small else 5
    length = max(5, 4 * cell_size / factor)
    padding = cell_size / 10
    
    # diamond points
    A = (x1 + length / 2 + padding, y1 + padding)
    B = (x1 + length + padding, y1 + length / 2 + padding)
    C = (x1 + length / 2 + padding, y1 + length + padding)
    D = (x1 + padding, y1 + length / 2 + padding)
    
    print(
        f'<polygon points="{join_points([A, B, C, D])}" fill="#FF33FC" stroke="#F99C04" stroke-width="2"/>',
        file=svg_string)

def write_cross(svg_string: any, scx: float, scy: float, env: WilsonMazeEnv) -> None:
    """
        Write a cross to the SVG string handle svg_string at the given position.

        Args:
            svg_string: The SVG string.
            scx: The scaling factor for the x coordinate.
            scy: The scaling factor for the y coordinate.
            env: The environment.
    """
    x1, y1, x2, y2 = get_cell_box_coordinates(env.agent_pos, scx, scy)
    
    # caclulate necessary values
    cell_size = x2 - x1
    length = max(5, 2 * cell_size / 10)
    padding = env.padding if env.padding <= (x2 - x1) / 10 else 5
    
    # cross points
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
        file=svg_string)
    
def write_coin(svg_string: any, cell_position: tuple[int, int], scx: float, scy: float, env: WilsonMazeEnv) -> None:
    """
        Write a coin to the SVG string handle svg_string at the given position.

        Args:
            svg_string: The SVG string.
            cell_position: The position.
            scx: The scaling factor for the x coordinate.
            scy: The scaling factor for the y coordinate.
            env: The environment.
    """
    x1, y1, x2, y2 = get_cell_box_coordinates(cell_position, scx, scy)
    
    # caclulate necessary values
    cell_size = x2 - x1
    factor = 7
    radius = max(5, cell_size / factor)
    
    print(
        f'<circle cx="{x1 + cell_size // 2}" cy="{y1 + cell_size // 2}" r="{radius}" fill="yellow" stroke="black" stroke-width="2"/>',
        file=svg_string)

def write_svg(env: WilsonMazeEnv, filename: str = None) -> str:
        """
            Write an SVG image of the maze to filename.
            Original at https://scipython.com/blog/making-a-maze/.

            Args:
                env: The environment.
                filename: The filename to save the SVG image to. Defaults to None.
        """

        # Scaling factors mapping maze coordinates to image coordinates
        scy, scx = env.window_height / env.size, env.window_width / env.size

        svg_string = io.StringIO()

        # Write the SVG image file for maze
        # SVG preamble and styles.
        print('<?xml version="1.0" encoding="utf-8"?>', file=svg_string)
        print('<svg xmlns="http://www.w3.org/2000/svg"', file=svg_string)
        print('    xmlns:xlink="http://www.w3.org/1999/xlink"', file=svg_string)
        print('    width="{:d}" height="{:d}" viewBox="{} {} {} {}">'
              .format(env.window_width + 2 * env.padding, env.window_height + 2 * env.padding,
                      -env.padding, -env.padding, env.window_width + 2 * env.padding,
                      env.window_height + 2 * env.padding),
              file=svg_string)
        
        # Draw agent and targets
        write_triangle(svg_string, env.triangle_target_pos, scx, scy)
        write_circle(svg_string, env.circle_target_pos, scx, scy)
        write_square(svg_string, env.square_target_pos, scx, scy)
        write_diamond(svg_string, env.diamond_target_pos, scx, scy)
        write_cross(svg_string, scx, scy, env)

        if env.target_id == 0:
            write_triangle(svg_string, env.agent_pos, scx, scy)
        elif env.target_id == 1:
            write_square(svg_string, env.agent_pos, scx, scy)
        elif env.target_id == 2:
            write_circle(svg_string, env.agent_pos, scx, scy)
        else:
            write_diamond(svg_string, env.agent_pos, scx, scy)

        # Draw the "South" and "East" walls of each cell, if present (these
        # are the "North" and "West" walls of a neighbouring cell in
        # general, of course).
        for x in range(env.size):
            for y in range(env.size):
                if env.maze[x][y].walls['S']:
                    x1, y1, x2, y2 = x * scx, (y + 1) * scy, (x + 1) * scx, (y + 1) * scy
                    write_wall(svg_string, x1, y1, x2, y2)
                if env.maze[x][y].walls['E']:
                    x1, y1, x2, y2 = (x + 1) * scx, y * scy, (x + 1) * scx, (y + 1) * scy
                    write_wall(svg_string, x1, y1, x2, y2)
                
                # Draw coins
                if env.maze[x][y].coin:
                    write_coin(svg_string, (x, y), scx, scy, env)

        # Draw the North and West maze border, which won't have been drawn
        # by the procedure above.
        print('<line x1="0" y1="0" x2="{}" y2="0" stroke="black" stroke-width="5"/>'.format(env.window_width),
              file=svg_string)
        print('<line x1="0" y1="0" x2="0" y2="{}" stroke="black" stroke-width="5"/>'.format(env.window_height),
              file=svg_string)

        # close svg file
        print('</svg>', file=svg_string)

        svg_string_content = svg_string.getvalue()
        svg_string.close()

        if filename:
            with open(os.path.join(os.path.dirname(__file__), filename), 'w') as f:
                print(svg_string_content, file=f)

        return svg_string_content