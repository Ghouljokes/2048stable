"""Module for the game."""
import random
import numpy as np
from grid import Grid

SIZE = 4
START_TILES = 2
WRONG_MOVE_PUNISHMENT = -30
WRONG_MOVE_CAP = 5
DIR_VECTORS = [
    np.array((-1, 0)),  # up
    np.array((0, 1)),  # right
    np.array((1, 0)),  # down
    np.array((0, -1)),  # left
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

    def move_tile(self, start: tuple[int, int], end: np.ndarray):
        """Move a tile to a given position.
        Args:
            start (tuple[int, int]): Position to move from.
            end (np.ndarray): Position to move tile to.
        """
        temp = self.grid.cells[start]
        self.grid.cells[start] = 0
        self.grid.cells[tuple(end)] = temp

    def build_traversals(self, vector: np.ndarray):
        """Build array indicating how to traverse through grid.
        Args:
            vector (array): array representing y and x direction.
        Returns:
            array: Array representing traversal orders for rows and colums."""
        trav = np.array(range(SIZE), dtype=np.int64)
        return np.array([trav[::-1] if i == 1 else trav for i in vector])

    def furthest_pos(self, cell: tuple[int, int], vector: np.ndarray):
        """Find furthest position a cell can move in a given vector."""
        previous = np.array(cell)
        new_cell = cell + vector
        while (
            self.grid.within_bounds(new_cell)
            and not self.grid.cells[new_cell[0], new_cell[1]]
        ):
            previous = new_cell
            new_cell = previous + vector
        return {"furthest": previous, "next": tuple(new_cell)}

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
        previous_grid = self.grid.cells.copy()
        self.reward = 0
        if self.is_game_terminated():
            return
        vector = DIR_VECTORS[direction]
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
                next_value = 0
                if self.grid.within_bounds(np.array(next_cell)):
                    next_value = self.grid.cells[next_cell]
                else:
                    self.move_tile(cell, positions["furthest"])
                    continue
                if next_value == value and next_cell not in merged_cells:
                    self.merge_tiles(cell, next_cell)
                    merged_cells.append(next_cell)
                    self.reward += self.grid.cells[next_cell]
                else:
                    self.move_tile(cell, positions["furthest"])
        current_grid = self.grid.cells
        if (previous_grid == current_grid).all():
            self.reward = WRONG_MOVE_PUNISHMENT
            self.stuck_counter += 1
            return
        matches_available = self.tile_matches_available()
        self.reward += matches_available * 2
        self.add_starting_tile()
        self.stuck_counter = 0
        if not (self.grid.amount_empty() > 0 or matches_available):
            self.over = True

    def tile_matches_available(self):
        """Count the tile matches can be made."""
        amount_available = 0
        for i in range(SIZE):
            for j in range(SIZE):
                tile = self.grid.cells[i, j]
                if not tile:  # skip empty tiles.
                    continue
                for vector in [DIR_VECTORS[1], DIR_VECTORS[2]]:
                    cell = vector + (i, j)
                    if self.grid.within_bounds(cell):
                        other = self.grid.cells[tuple(cell)]
                    else:
                        other = 0
                    if other == tile:
                        amount_available += 1
        return amount_available


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
