import os
import sys
from stable_baselines3.a2c.a2c import A2C

# 1. Import `QApplication` and all the required widgets
from PyQt6.QtWidgets import QApplication, QWidget

app = QApplication(sys.argv)


window = QWidget()
window.setWindowTitle("2048 Stable Ui")


window.show()

sys.exit(app.exec())
