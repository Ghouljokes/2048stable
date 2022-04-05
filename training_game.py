"""Module for the game."""
import random
import numpy as np
from grid import Grid
from tile import Tile


class TrainGame:
    """Manage instance of the game."""

    def __init__(self, size):
        """Initiate game on board of dims size x size"""
        self.size = size
        self.start_tiles = 2
        self.score = 0
        self.reward = 0
        self.keep_playing = False
        self.stuck_counter = 0
        self.over = False
        self.won = False
        self.dir_vectors = {
            0: (-1, 0),  # up
            1: (0, 1),  # right
            2: (1, 0),  # down
            3: (0, -1),  # left
        }
        self.set_up()

    def add_random_tile(self):
        """Add a 2 or 4 to a random spot on the grid."""
        if self.grid.amount_empty():
            value = 2 if random.random() < 0.9 else 4
            tile = Tile(self.grid.random_available_cell(), value)
            self.grid.insert_tile(tile)

    def add_start_tiles(self):
        """Add the starting tiles."""
        for _ in range(self.start_tiles):
            self.add_random_tile()

    def set_up(self):
        """Set up new game."""
        self.grid = Grid(self.size)
        self.score = 0
        self.reward = 0
        self.stuck = False
        self.over = False
        self.won = False
        self.keep_playing = False
        self.stuck_counter = 0
        self.add_start_tiles()

    def restart(self):
        """Reset game."""
        self.set_up()

    def set_keep_playing(self):
        """Continue game."""
        self.keep_playing = True

    def is_game_terminated(self):
        """Check if the game has ended."""
        if self.over or (self.won and not self.keep_playing) or self.stuck_counter > 5:
            return True
        return False

    def prepare_tiles(self):
        """Prepare tiles to be moved."""
        for grid_row in self.grid.cells:
            for tile in grid_row:
                if tile:
                    tile.merged_from = None

    def positions_equal(self, position, tile: Tile):
        """Check to see if a position is the same as a tile's position."""
        return position[0] == tile.pos[0] and position[1] == tile.pos[1]

    def move_tile(self, tile: Tile, cell: tuple[int, int]):
        """Move a tile to a given position.
        Args:
            tile (Tile): Tile to be moved.
            cell (tuple[int, int]): Position to move tile to.
        """
        self.grid.cells[tile.pos[0]][tile.pos[1]] = None
        self.grid.cells[cell[0]][cell[1]] = tile
        tile.update_position(cell)

    def move(self, direction: int):
        """Move all tiles in a given direction.
        Args:
            direction (int): Integer representation of a direction.
        """
        # 0: up, 1: right, 2: down, 3: left
        self.reward = 0
        if self.is_game_terminated():
            return
        vector = self.dir_vectors[direction]
        traversals = self.build_traversals(vector)
        moved = False
        self.prepare_tiles()
        for trav_row in traversals["row"]:
            for trav_col in traversals["col"]:
                cell = (trav_row, trav_col)
                tile = self.grid.cell_content(cell)
                if not tile:
                    continue
                positions = self.furthest_position(cell, vector)
                next_tile = self.grid.cell_content(positions["next"])
                if (
                    next_tile
                    and next_tile.value == tile.value
                    and not next_tile.merged_from
                ):
                    merged = Tile(positions["next"], tile.value * 2)
                    merged.merged_from = [tile, next_tile]
                    self.grid.insert_tile(merged)
                    self.grid.remove_tile(tile)
                    tile.update_position(positions["next"])
                    self.score += merged.value
                    self.reward += merged.value
                    if merged.value == 2048:
                        self.won = True
                else:
                    self.move_tile(tile, positions["furthest"])

                if not self.positions_equal(cell, tile):
                    moved = True

        if moved:
            self.add_random_tile()
            if not self.moves_available():
                self.over = True
                self.stuck_counter = 0
        else:
            self.score -= 20
            self.reward -= 20
            self.stuck_counter += 1

    def build_traversals(self, vector: tuple[int, int]):
        """Build lists indicating how to traverse through grid."""
        traversals = {"row": [], "col": []}
        for pos in range(self.size):
            traversals["row"].append(pos)
            traversals["col"].append(pos)
        if vector[0] == 1:
            traversals["row"].reverse()
        if vector[1] == 1:
            traversals["col"].reverse()
        return traversals

    def furthest_position(self, cell: tuple, vector: tuple):
        """Find furthest position a cell can move in a given vector."""
        previous = cell
        cell = (previous[0] + vector[0], previous[1] + vector[1])
        while self.grid.within_bounds(cell) and self.grid.cell_available(cell):
            previous = cell
            cell = (previous[0] + vector[0], previous[1] + vector[1])
        return {"furthest": previous, "next": cell}

    def tile_matches_available(self):
        """Check if any tile matches can be made."""
        for i in range(self.size):
            for j in range(self.size):
                tile = self.grid.cell_content((i, j))
                if tile:
                    for direction in range(4):
                        vector = self.dir_vectors[direction]
                        cell = (i + vector[0], j + vector[1])
                        other = self.grid.cell_content(cell)
                        if other and other.value == tile.value:
                            return True
        return False

    def moves_available(self):
        """Check to see if a move can still be made."""
        return self.grid.amount_empty() or self.tile_matches_available()

    def get_array(self):
        """Retrieve array of all squares on the board."""
        grid_read = self.grid.readable_grid()
        grid_array = np.array(grid_read).flatten()
        return grid_array


if __name__ == "__main__":
    test_game_manager = TrainGame(4)
    grid = test_game_manager.grid.readable_grid()
    for row in grid:
        print(row)

    while True:
        move_direction = int(input())
        test_game_manager.move(move_direction)
        grid = test_game_manager.grid.readable_grid()
        for row in grid:
            print(row)
        print(test_game_manager.score)
