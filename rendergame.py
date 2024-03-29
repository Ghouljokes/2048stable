"""Module for Game_Instance object."""
import os
import numpy as np
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import StaleElementReferenceException
from defs import VALUES, POSITIONS, WINDOW_SIZE, WINDOW_POS

GAME_LOCATION = os.path.join(os.getcwd(), "2048-master/index.html")


class RenderGame:
    """Instance of 2048."""

    def __init__(self):
        """Launch an instance of 2048."""
        self.driver = WebDriver()
        self.driver.set_window_size(WINDOW_SIZE[0], WINDOW_SIZE[1])
        self.driver.set_window_position(WINDOW_POS[0], WINDOW_POS[1])
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
            # keep trying for grid until it's available.
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

    def is_terminated(self):
        """Check if the game is over."""
        message = self.message.get_attribute("class")
        if "game-won" in message:
            keep_playing = "keep-playing-button"
            self.driver.find_element(By.CLASS_NAME, keep_playing).click()
            return False
        return "game-over" in message
