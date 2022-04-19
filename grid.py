"""Container for game grid."""
import random
import numpy as np


class Grid:
    """Grid of tiles."""

    def __init__(self, size: int):
        self.size = size
        self.cells = np.zeros((size, size)).astype(int)

    def random_available_cell(self):
        """Return position of random empty cell.
        Returns:
            tuple[int, int]: Position of randomly chosen cell.
        """
        return random.choice(self.available_cells())

    def available_cells(self):
        """Return list of positions of empty cells."""
        empty_cells: list[tuple[int, int]] = []
        for row in range(self.size):
            for col in range(self.size):
                if self.cells[(row, col)] == 0:
                    empty_cells.append((row, col))
        return empty_cells

    def amount_empty(self):
        """Count amount of empty cells."""
        return len(self.available_cells())

    def within_bounds(self, pos: tuple[int, int]):
        """Check if given position is within grid bounds"""
        for coord in pos:
            if coord not in range(self.size):
                return False
        return True

    def flat_grid(self):
        return self.cells.flatten()
