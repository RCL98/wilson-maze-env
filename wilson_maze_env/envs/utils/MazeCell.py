import numpy as np


class MazeCell:

    def __init__(self):
        self.coin = False
        self.agent = False
        self.visited = False
        self.target = -1
        self.short_path = -1
        
        # these flags are only relevant for the Wilson's algorithm
        self.already_belogns_to_maze = False
        self.passed_while_generating = False

        self.walls = {'N': 1, 'S': 1, 'W': 1, 'E': 1}

    def knock_wall(self, wall):
        self.walls[wall] = 0

    def can_go_direction(self, direction: str):
        return not self.walls[direction]

    def get_state(self):
        state = list(self.walls.values())
        return np.array(state, dtype=np.uint8)