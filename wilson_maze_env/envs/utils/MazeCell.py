import numpy as np


class MazeCell:
    TARGET_VALUE = 10
    AGENT_VALUE = 1
    COIN_VALUE = 5
    EMPTY_VALUE = 0
    NON_VISITED_VALUE = -1

    def __init__(self, value):
        self.value = value
        self.walls = {'N': 1, 'S': 1, 'W': 1, 'E': 1}

    def knock_wall(self, wall):
        self.walls[wall] = 0

    def can_go_direction(self, direction: str):
        return not self.walls[direction]

    def get_state(self):
        state = list(self.walls.values())
        return np.array(state, dtype=np.uint8)