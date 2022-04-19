"""Module for the game."""
import random
import numpy as np
from grid import Grid

SIZE = 4
START_TILES = 2
WRONG_MOVE_PUNISHMENT = -10
WRONG_MOVE_CAP = 5
DIR_VECTORS = [
    (-1, 0),  # up
    (0, 1),  # right
    (1, 0),  # down
    (0, -1),  # left
]


class TrainGame:
    """Manage instance of the game."""

    def __init__(self):
        """Initiate game on board of dims size x size"""
        self.grid = Grid(SIZE)
        self.reward = 0
        self.stuck_counter = 0
        self.over = False
        self.set_up()

    def add_starting_tile(self):
        """Add a 2 or 4 to a random cell on the grid."""
        value = 2 if random.random() < 0.9 else 4
        cell = self.grid.random_available_cell()
        self.grid.cells[cell] = value

    def set_up(self):
        """Set up new game."""
        self.grid = Grid(SIZE)
        self.reward = 0
        self.stuck = False
        self.over = False
        self.stuck_counter = 0
        for _ in range(START_TILES):
            self.add_starting_tile()

    def is_game_terminated(self):
        """Check if the game has ended."""
        return self.over or self.stuck_counter > WRONG_MOVE_CAP

    def move_tile(self, start: tuple[int, int], end: tuple[int, int]):
        """Move a tile to a given position.
        Args:
            start (tuple[int, int]): Position to move from.
            cell (tuple[int, int]): Position to move tile to.
        """
        temp = self.grid.cells[start] * 1
        self.grid.cells[start] = 0
        self.grid.cells[end] = temp

    def build_traversals(self, vector: tuple[int, int]):
        """Build array indicating how to traverse through grid.
        Args:
            vector (tuple[int, int]): tuple representing y and x direction.
        Returns:
            array: Array representing traversal orders for rows and colums."""
        traversals = np.zeros((2, 4)).astype(int)
        reverse_trav = range(SIZE - 1, -1, -1)
        for i in range(2):
            traversals[i] = reverse_trav if vector[i] == 1 else range(SIZE)
        return traversals

    def furthest_pos(self, cell: tuple[int, int], vector: tuple[int, int]):
        """Find furthest position a cell can move in a given vector."""
        previous = cell
        cell = (previous[0] + vector[0], previous[1] + vector[1])
        while self.grid.within_bounds(cell) and self.grid.cells[cell] == 0:
            previous = cell
            cell = (previous[0] + vector[0], previous[1] + vector[1])
        return {"furthest": previous, "next": cell}

    def merge_tiles(self, tile1: tuple[int, int], tile2: tuple[int, int]):
        """Merges two tiles into a new tile at second tile's position."""
        self.grid.cells[tile2] *= 2
        self.grid.cells[tile1] = 0
        self.reward += self.grid.cells[tile2]

    def move(self, direction: int):
        """Move all tiles in a given direction.
        Args:
            direction (int): Integer representation of a direction.
        """
        # 0: up, 1: right, 2: down, 3: left
        previous_grid = self.grid.flat_grid()
        self.reward = 0
        if self.is_game_terminated():
            return
        vector: tuple[int, int] = DIR_VECTORS[direction]
        traversals = self.build_traversals(vector)
        merged_cells = []  # Track merged cells to avoid double merges.
        for trav_row in traversals[0]:
            for trav_col in traversals[1]:
                cell: tuple[int, int] = (trav_row, trav_col)
                value: int = self.grid.cells[cell]
                if not value:
                    continue
                positions = self.furthest_pos(cell, vector)
                next_cell = positions["next"]
                if self.grid.within_bounds(next_cell):
                    next_value: int = self.grid.cells[next_cell]
                else:
                    next_value: int = 0
                    self.move_tile(cell, positions["furthest"])
                    continue
                if next_value == value and next_cell not in merged_cells:
                    self.merge_tiles(cell, next_cell)
                    merged_cells.append(next_cell)
                    self.reward += self.grid.cells[next_cell]
                else:
                    self.move_tile(cell, positions["furthest"])
        current_grid = self.grid.flat_grid()
        lower_right = self.grid.cells[(-1, -1)]
        max = np.max(current_grid)
        if max == lower_right:  # Reward ai for maximum tile in lower right.
            self.reward += max
        if not (previous_grid == current_grid).all():
            self.add_starting_tile()
            self.stuck_counter = 0
            if not self.moves_available():
                self.over = True
        else:
            self.reward = WRONG_MOVE_PUNISHMENT
            self.stuck_counter += 1

    def tile_matches_available(self):
        """Check if any tile matches can be made."""
        for i in range(SIZE):
            for j in range(SIZE):
                tile = self.grid.cells[i, j]
                if not tile:  # skip empty tiles.
                    continue
                for vector in DIR_VECTORS:
                    cell = (i + vector[0], j + vector[1])
                    if self.grid.within_bounds(cell):
                        other = self.grid.cells[cell]
                    else:
                        other = 0
                    if other == tile:
                        return True
        return False

    def moves_available(self):
        """Check to see if a move can still be made."""
        return self.grid.amount_empty() > 0 or self.tile_matches_available()


if __name__ == "__main__":
    test_game = TrainGame()
    wasd_dirs = {"w": 0, "d": 1, "s": 2, "a": 3}
    for row in test_game.grid.cells:
        print(row)

    while True:
        move_key = input()
        move_direction = wasd_dirs[move_key]
        test_game.move(move_direction)
        for row in test_game.grid.cells:
            print(row)
