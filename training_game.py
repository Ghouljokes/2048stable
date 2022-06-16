"""Module for the game."""
import random
import numpy as np
from defs import MAX_STUCK, WRONG_MOVE_PUNISHMENT, DIR_VECTORS, TRAVERSALS

SIZE = 4
START_TILES = 2


def create_positions():
    """Create list of all position coords on the board.

    Returns:
        list[tuple[int]]: List of all possible positions.
    """
    positions: list[tuple[int, int]] = []
    for row in range(SIZE):
        for col in range(SIZE):
            positions.append((row, col))
    return positions


POSITIONS = create_positions()


def on_grid(pos):
    """Check if given position is within grid bounds."""
    return tuple(pos) in POSITIONS


class TrainGame:
    """Manage instance of the game."""

    def __init__(self):
        """Initiate game on board of dims size x size."""
        self.grid = np.zeros((SIZE, SIZE), dtype=np.int64)
        self.reward = 0
        self.stuck_counter = 0
        self.over = False
        self.merges = []  # Track merged cells to avoid double merges.
        for _ in range(START_TILES):
            self.add_starting_tile()

    def available_cells(self):
        """Return list of positions of empty cells."""
        empty_cells: list[tuple[int, int]] = []
        for pos in POSITIONS:
            if self.grid[pos] == 0:
                empty_cells.append(pos)
        return empty_cells

    def add_starting_tile(self):
        """Add a 2 or 4 to a random cell on the grid."""
        cell = random.choice(self.available_cells())
        self.grid[cell] = 2 if random.random() < 0.9 else 4

    def is_terminated(self):
        """Check if the game has ended."""
        return self.over or self.stuck_counter > MAX_STUCK

    def pos_swap(self, start: tuple[int, int], end):
        """Move a tile to a given position.

        Args:
            start (tuple[int, int]): Position to move from.
            end (tuple[int, int]): Position to move tile to.
        """
        temp = self.grid[start]
        self.grid[start] = 0
        self.grid[end] = temp

    def furthest_pos(self, cell: tuple[int, int], vector: np.ndarray):
        """Get furthest and next tiles from cell in given direction.

        Args:
            cell (tuple[int, int]): Coordinates of starting cell
            vector (np.ndarray): Direction vector to check in.

        Returns:
            tuple[tuple, tuple]: furthest empty square and the cell after.
        """
        furthest_empty = cell
        next_cell = tuple(cell + vector)
        while on_grid(next_cell) and self.grid[next_cell] == 0:
            furthest_empty = next_cell
            next_cell = tuple(furthest_empty + vector)
        return furthest_empty, next_cell

    def merge(self, tile1: tuple[int, int], tile2: tuple[int, int]):
        """Merge tile1 into tile2."""
        self.grid[tile2] *= 2
        self.grid[tile1] = 0
        self.reward += self.grid[tile2]
        self.merges.append(tile2)

    def check_merge(self, cell: tuple[int, int], next_cell):
        """Check if a merge will occur between two cells.

        Args:
            cell (tuple): coordinates of the cell to check from.
            next_cell: coordinates of cell to merge to.

        Returns:
            bool: Whether or not a merge will happen
        """
        return (
            on_grid(next_cell)
            and self.grid[cell] == self.grid[next_cell]
            and next_cell not in self.merges
        )

    def move_cell(self, cell: tuple, vector):
        """Move a cell in a given direction.

        Args:
            cell (tuple): Cell to move the contents of.
            vector (np.ndarray): Direction vector for the cell to move in.
        """
        furthest_space, next_cell = self.furthest_pos(cell, vector)
        if self.check_merge(cell, next_cell):
            self.merge(cell, next_cell)
        else:
            self.pos_swap(cell, furthest_space)

    def move(self, direction: int):
        """Move all tiles in a given direction.

        Args:
            direction (int): Integer representation of a direction.
        """
        # 0: up, 1: right, 2: down, 3: left
        self.reward = 0
        previous_grid = self.grid.copy()
        vector = DIR_VECTORS[direction]
        # direction to traverse the grid.
        traversals = TRAVERSALS[direction]
        self.merges = []  # reset merged cells
        for trav_row in traversals[0]:
            for trav_col in traversals[1]:
                cell: tuple[int, int] = (trav_row, trav_col)
                if self.grid[cell] != 0:
                    self.move_cell(cell, vector)
        if (previous_grid == self.grid).all():  # if invalid move
            self.reward += WRONG_MOVE_PUNISHMENT
            self.stuck_counter += 1
            return
        self.add_starting_tile()
        self.stuck_counter = 0
        if not self.available_cells() and not self.matches_available():
            self.over = True

    def matches_available(self):
        """Check if any matches can be made."""
        for pos in POSITIONS:
            if self.grid[pos] == 0:
                continue
            for i in range(2):
                _, next_cell = self.furthest_pos(pos, DIR_VECTORS[i])
                if self.check_merge(pos, next_cell):
                    return True
        return False

    def show_board(self):
        """Display readable version of grid, for testing."""
        display = [["0", "0", "0", "0"] for _ in range(4)]
        for pos in POSITIONS:
            display[pos[0]][pos[1]] = str(self.grid[pos])
        for col in range(SIZE):
            lengths = [len(display[row][col]) for row in range(SIZE)]
            for row in range(SIZE):
                while len(display[row][col]) < max(lengths):
                    display[row][col] += " "
        for row in display:
            print("|".join(row))


if __name__ == "__main__":

    test_game = TrainGame()
    wasd_dirs = {"w": 0, "d": 1, "s": 2, "a": 3}
    test_game.show_board()

    while not test_game.is_terminated():
        move_key = input()
        move_direction = wasd_dirs[move_key]
        # move_direction = random.randint(0, 3)
        test_game.move(move_direction)
        test_game.show_board()
