"""Module for Game_Instance object."""
import time
import os
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import StaleElementReferenceException

GAME_LOCATION = os.path.join(os.getcwd(), "2048-master/index.html")
VALUES = {
    "tile-2": 2,
    "tile-4": 4,
    "tile-8": 8,
    "tile-16": 16,
    "tile-32": 32,
    "tile-64": 64,
    "tile-128": 128,
    "tile-256": 256,
    "tile-512": 512,
    "tile-1024": 1024,
    "tile-2048": 2048,
}
POSITIONS = {
    "tile-position-1-1": 0,
    "tile-position-2-1": 1,
    "tile-position-3-1": 2,
    "tile-position-4-1": 3,
    "tile-position-1-2": 4,
    "tile-position-2-2": 5,
    "tile-position-3-2": 6,
    "tile-position-4-2": 7,
    "tile-position-1-3": 8,
    "tile-position-2-3": 9,
    "tile-position-3-3": 10,
    "tile-position-4-3": 11,
    "tile-position-1-4": 12,
    "tile-position-2-4": 13,
    "tile-position-3-4": 14,
    "tile-position-4-4": 15,
}


class RenderGame:
    """Instance of 2048."""

    def __init__(self, position=(10, 10), size=(500, 600)):
        """Launch an instance of 2048.

        Args:
            game_location (str): Location game is located on computer.
            position (tuple): Position to open window at. Defaults to (10, 10).
            size (tuple): Size to open window as. Defaults to (500, 600)
        """
        self.driver = webdriver.Chrome()
        self.driver.set_window_size(size[0], size[1])
        self.driver.set_window_position(position[0], position[1])
        self.driver.get("file://" + GAME_LOCATION)
        # Store WebElements.
        self.body = self.driver.find_element(By.CSS_SELECTOR, "body")
        self.message = self.driver.find_element(By.CLASS_NAME, "game-message")
        self.tiles = self.driver.find_element(By.CLASS_NAME, "tile-container")
        self.tile_sub_elements = self.tiles.find_elements(By.XPATH, "./*")
        self.replay = self.driver.find_element(By.CLASS_NAME, "restart-button")

    def get_tile_classes(self) -> list[str]:
        """Get list of the classes for all tiles currently in self.tiles.

        Returns:
            list[str]: List of tile classes.
        """
        return [
            tile.get_attribute("class")
            for tile in self.tiles.find_elements(By.XPATH, "./*")
        ]

    def get_array(self):
        """Return array of all squares on the board.

        Returns:
            array: 1D array of all squares.
        """
        while True:
            try:
                playing_grid = np.zeros((16))
                for tile_class in self.get_tile_classes():
                    tile_split = tile_class.split(" ")
                    value = VALUES[tile_split[1]]
                    position = POSITIONS[tile_split[2]]
                    playing_grid[position] = value
                break
            except StaleElementReferenceException:
                continue
        return playing_grid

    def move(self, direction):
        """Move all tiles in given direction."""
        directions = [Keys.UP, Keys.RIGHT, Keys.DOWN, Keys.LEFT]
        self.body.send_keys(directions[direction])

    def is_game_terminated(self):
        """Check if game is over."""
        return "game-over" in self.message.get_attribute("class")

    def quit(self):
        """Quit the game."""
        self.driver.quit()


if __name__ == "__main__":
    test_game = RenderGame(GAME_LOCATION)
    print(test_game.get_tile_classes())
    time.sleep(2)
    test_game.quit()
