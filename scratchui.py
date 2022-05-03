import os
import sys
import learn
import PyQt6.QtWidgets as qtw
from PyQt6.QtGui import QFont


class LearnWindow(qtw.QWidget):
    def __init__(self):
        super().__init__()
        self.model_type = "PPO"
        self.time_steps = 25000
        self.show_game = False
        self.initUI()

    def initUI(self):
        """Initialize the UI elements"""
        qtw.QToolTip.setFont(QFont("SansSerif", 10))
        self.setWindowTitle("2048 Stable Baselines")

        main_layout = qtw.QVBoxLayout()

        learn_button = qtw.QPushButton("Run learning algorithm")
        main_layout.addWidget(learn_button)

        show_game = qtw.QCheckBox("Show game", self)
        main_layout.addWidget(show_game)

        grid = qtw.QGridLayout()
        main_layout.addLayout(grid)

        mtype_label = qtw.QLabel("Model Type")
        grid.addWidget(mtype_label, 0, 0)

        model_types = qtw.QComboBox(self)
        model_types.addItem("PPO")
        model_types.addItem("A2C")
        model_types.addItem("DQN")
        grid.addWidget(model_types, 0, 1)

        timestep_label = qtw.QLabel("Timesteps")
        grid.addWidget(timestep_label, 1, 0)

        timesteps = qtw.QSpinBox()
        timesteps.setRange(10000, 1000000)
        timesteps.setValue(25000)
        grid.addWidget(timesteps, 1, 1)

        self.setLayout(main_layout)
        self.show()


def main():
    app = qtw.QApplication(sys.argv)
    window = LearnWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
