"""Container for player module."""
import copy
import time
import numpy as np
from selenium.webdriver.common.keys import Keys
from rendergame import RenderGame


MOVES = [Keys.LEFT, Keys.DOWN, Keys.RIGHT, Keys.UP]


class Player:
    """Play a round of 2048."""

    def __init__(self, brain: Brain):
        """Initialize game to be played on game instance.

        Args:
            brain (Brain): Neural net that will make decisions.
        """
        self.brain = brain
        self.score = 0
        self.fitness = 0
        self.status = 'alive'
        self.last_grid = np.zeros((16))
        self.curr_grid = np.zeros((16))

    def new_game(self, game: RenderGame):
        """Reset everything to a new game state."""
        game.replay.click()
        self.score = 0
        self.status = 'alive'
        self.last_grid = np.zeros((16))
        self.curr_grid = np.zeros((16))

    def new_turn(self, game):
        """Ready the grid for a new turn.

        Args:
            game (GameInstance): Game to intiate new turn on."""
        self.last_grid = self.curr_grid
        self.curr_grid = game.get_grid()

    def make_move(self, game: RenderGame):
        """Choose direction to move in.

        Args:
            game (GameInstance): Game to move on."""
        net_output = self.brain.brain_blast(self.curr_grid)
        direction = MOVES[np.argmax(net_output)]
        game.body.send_keys(direction)

    def update_status(self, game: RenderGame):
        """Set status to reflect current state.

        Args:
            game (GameInstance): Game to get status from."""
        if "game-over" in game.message.get_attribute("class"):
            self.status = "dead"
        elif (self.curr_grid == self.last_grid).all():
            self.status = "stuck"
        elif 2048 in self.curr_grid:
            time.sleep(1000)

    def set_score(self, game: RenderGame):
        """Set score equal to score on the game board.

        Args:
            game (GameInstance): game to get score from."""
        score_text = game.score_element.text
        score = int(score_text.split("\n")[0])
        self.score = score

    def play_game(self, game: RenderGame):
        """Play through a game of 2048 and set score.
        Args:
            game (GameInstance): game to play."""
        self.new_game(game)
        while self.status == 'alive':
            self.new_turn(game)
            self.make_move(game)
            self.update_status(game)
        self.new_turn(game)
        self.set_score(game)

    def train_game(self, game)

    def offspring_brain(self):
        """Make slightly mutated copy of current brain.

        Returns:
            Brain: Offspring brain.
        """
        new_brain = copy.deepcopy(self.brain)
        new_brain.mutate(0.05)
        return new_brain


if __name__ == "__main__":
    test_game = RenderGame(GAME_LOCATION)
    layer_list = [random_layer(16, 16), random_layer(16, 4)]
    test_brain = Brain(layer_list)
    test_player = Player(test_brain)
    test_player.play_game(test_game)
