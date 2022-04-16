"""Container for game grid."""
import random
from tile import Tile


class Grid:
    """Grid of tiles."""

    def __init__(self, size: int):
        self.size = size
        self.cells = self.empty()

    def empty(self):
        """Create empty grid.
        Returns:
            list: List of empty cells.
        """
        cells = []
        while len(cells) < self.size:
            cells.append([None for i in range(self.size)])
        return cells

    def random_available_cell(self):
        """Return position of random empty cell.
        Returns:
            tuple[int, int]: Position of randomly chosen cell.
        """
        return random.choice(self.available_cells())

    def available_cells(self):
        """Return list of positions of empty cells.
        Returns:
            list[tuple[int, int]]: Positions of all empty cells.
        """
        cells: list[tuple[int, int]] = []
        for row in range(self.size):
            for col in range(self.size):
                tile = self.cells[row][col]
                if not tile:
                    cells.append((row, col))
        return cells

    def amount_empty(self):
        """Count the amount of empty cells."""
        return len(self.available_cells()) > 0

    def cell_available(self, cell: tuple[int, int]):
        """Check if the cell at the given position is empty."""
        return self.cell_content(cell) is None

    def cell_content(self, cell: tuple[int, int]):
        """Return contents of cell at given position."""
        if self.within_bounds(cell):
            return self.cells[cell[0]][cell[1]]
        return None

    def insert_tile(self, tile: Tile):
        """Insert tile into the grid."""
        self.cells[tile.pos[0]][tile.pos[1]] = tile

    def remove_tile(self, tile: Tile):
        """Remove tile from grid."""
        self.cells[tile.pos[0]][tile.pos[1]] = None

    def within_bounds(self, position: tuple):
        """Check to see if given position is within bounds of grid."""
        return position[0] in range(self.size) and position[1] in range(self.size)

    def readable_grid(self):
        """Make readable version of the grid."""
        read_grid = []
        for row in self.cells:
            read_grid.append([tile.value if tile else 0 for tile in row])
        return read_grid
