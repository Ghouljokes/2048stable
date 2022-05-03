import os
import sys
import learn
import PyQt6.QtWidgets as qtw
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt


class LearnWindow(qtw.QWidget):
    def __init__(self):
        super().__init__()
        self.model_type = "PPO"
        self.time_steps = 25000
        self.showgame = False
        # main learn button for game.
        self.learn_button = qtw.QPushButton("Run learning algorithm")
        self.learn_button.clicked.connect(self.run_game)
        # Checkbox to show game.
        self.show_game = qtw.QCheckBox("Show game", self)
        self.show_game.stateChanged.connect(self.set_showgame)
        # label for model type box
        self.mtype_label = qtw.QLabel("Model Type")
        # combo box for selecting model type
        self.model_types = qtw.QComboBox(self)
        self.model_types.addItem("PPO")
        self.model_types.addItem("A2C")
        self.model_types.addItem("DQN")
        self.model_types.textActivated[str].connect(self.set_model_type)
        # label for timesteps spinbox
        self.timestep_label = qtw.QLabel("Timesteps")
        # timesteps spinbox
        self.timesteps = qtw.QSpinBox()
        self.timesteps.setRange(10000, 1000000)
        self.timesteps.setValue(25000)
        self.initUI()

    def initUI(self):
        """Initialize the UI elements"""
        qtw.QToolTip.setFont(QFont("SansSerif", 10))
        self.setWindowTitle("2048 Stable Baselines")

        main_layout = qtw.QVBoxLayout()
        main_layout.addWidget(self.learn_button)
        main_layout.addWidget(self.show_game)
        grid = qtw.QGridLayout()
        main_layout.addLayout(grid)
        grid.addWidget(self.mtype_label, 0, 0)
        grid.addWidget(self.model_types, 0, 1)
        grid.addWidget(self.timestep_label, 1, 0)
        grid.addWidget(self.timesteps, 1, 1)
        self.setLayout(main_layout)
        self.show()

    def set_showgame(self, state):
        self.showgame = state == Qt.CheckState.Checked.value

    def set_model_type(self, text):
        self.model_type = text

    def run_game(self):
        print(f"Running model of type {self.model_type}.")
        print(f"SHOWGAME: {self.showgame}")
        print(f"TSTEPINTERVAL: {self.timesteps.value()}")
        model, total_timesteps = learn.initialize_model(self.model_type)
        timestep_interval = self.timesteps.value()
        while True:
            learn.train_model(model, timestep_interval)
            total_timesteps += timestep_interval
            model.save(f"models/{self.model_type}/{total_timesteps}")
            if self.showgame:
                learn.show_game(model)

    def end_game(self):
        pass


def main():
    app = qtw.QApplication(sys.argv)
    window = LearnWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
