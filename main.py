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
       
    def initUI(self):
        self.criarWidgets()
        self.gerarLayouts()

    '''Cria os widgets que encorporam o Menu e widgets que executaram ações'''
    def criarWidgets(self):
        self.currentDate = QDate.currentDate()
        self.lastDay = self.currentDate.addDays(-1)

        # Widgets do formulário da ação da bolsa de valores
        self.stockTicket = QLineEdit()
        self.stockTicket.setFixedWidth(120)
        
        self.companyName = QLineEdit('')
        self.companyName.setReadOnly(True)
        self.companyName.setFixedWidth(120)
        self.companyName.setStyleSheet("border: 0px; background-color: transparent")

        self.companySection = QLineEdit('')
        self.companySection.setReadOnly(True)
        self.companySection.setFixedWidth(120)
        self.companySection.setStyleSheet("border: 0px; background-color: transparent")

        self.companyCountry = QLineEdit('')
        self.companyCountry.setReadOnly(True)
        self.companyCountry.setFixedWidth(120)
        self.companyCountry.setStyleSheet("border: 0px; background-color: transparent")

        self.currency = QLineEdit('')
        self.currency.setReadOnly(True)
        self.currency.setFixedWidth(120)
        self.currency.setStyleSheet("border: 0px; background-color: transparent")

        self.buttonSearch = QPushButton(self)
        self.buttonSearch.setMaximumWidth(100)
        self.buttonSearch.setText("Procurar") 
        self.buttonSearch.clicked.connect(self.searchStockData)
        self.buttonSearch.setDisabled(True)
        self.controlButtonSearchEnability = ControlButtonEnability(self.stockTicket, self.buttonSearch)
        self.stockTicket.textChanged.connect(self.controlButtonSearchEnability.checkStatus)


def main():
    app = QApplication(sys.argv)
    win = MyWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
