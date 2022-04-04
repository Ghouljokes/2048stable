"""Container for tile class."""


class Tile:
    """Represent a tile on the board."""

    def __init__(self, position: tuple[int, int], value=2):
        self.pos = position
        self.value: int = value

        self.merged_from: list | None = None

    def update_position(self, position: tuple):
        """Update current position to new position."""
        self.pos = position

    def make_copy(self):
        """Return new fresh tile based off this one.

        Returns:
            Tile: Tile with same position and value.
        """
        return Tile(self.pos, self.value)
