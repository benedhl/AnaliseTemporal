import sys
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QDate
from PyQt5.QtWidgets import *
import yfinance as yf
import numpy as np
import torch
import matplotlib.pyplot as plt


# Classe que controla habilitação dos botões de ações
class ControlButtonEnability():
    def __init__(self, widgetText, widgetButton ):
        self.textbox = widgetText
        self.button = widgetButton

    def checkStatus(self):
        if self.textbox.text() == "":
            self.textbox.setFocus()
            self.button.setDisabled(True)
        else:
            self.button.setEnabled(True)
            
# Classe do modelo da Rede Neural
class Net(torch.nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super(Net, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.fc1 = torch.nn.Linear(self.inputSize, self.hiddenSize)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hiddenSize, 1)
    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.relu(output)
        return output



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
