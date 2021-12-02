import sys
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QDate
from PyQt5.QtWidgets import *
import yfinance as yf
import numpy as np
import torch
import matplotlib.pyplot as plt


class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setWindowTitle("Análise Temporais - IFTM")
        self.icon = self.setWindowIcon(QIcon("images/icon/icon.png"))
        self.setGeometry(250, 150, 1020, 400)
        self.setWindowFlags(QtCore.Qt.WindowType.CustomizeWindowHint | QtCore.Qt.WindowType.WindowCloseButtonHint | QtCore.Qt.WindowType.WindowMinimizeButtonHint)
        self.initUI()
        self.show()
       


def main():
    app = QApplication(sys.argv)
    win = MyWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
