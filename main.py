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
        
        
   # Widgets do formulário da série Temporal
        self.epoch = QSpinBox()
        self.epoch.setRange(0,1000000)
        self.epoch.setSingleStep(100)
        self.epoch.setFixedWidth(90)
        self.epoch.setValue(15000)
        
        self.totalHiddenLayers = QSpinBox()
        self.totalHiddenLayers.setRange(0,1000)
        self.totalHiddenLayers.setSingleStep(1)
        self.totalHiddenLayers.setFixedWidth(90)
        self.totalHiddenLayers.setValue(100)

        self.learningRate = QDoubleSpinBox ()
        self.learningRate.setSingleStep(0.1)
        self.learningRate.setFixedWidth(90)
        self.learningRate.setValue(0.09)

        self.momentumValue = QDoubleSpinBox ()
        self.momentumValue.setSingleStep(0.1)
        self.momentumValue.setFixedWidth(90)
        self.momentumValue.setValue(0.03)

        self.inicialDate = QDateEdit(calendarPopup=True)
        self.inicialDate.setDate(self.lastDay)
        self.inicialDate.setMaximumDate(self.lastDay)
        self.inicialDate.setFixedWidth(90)

        self.finalDate = QDateEdit(calendarPopup=True)
        self.finalDate.setDate(self.currentDate)
        self.finalDate.setMaximumDate(self.currentDate)
        self.finalDate.setFixedWidth(90)

        self.buttonSubmit = QPushButton(self)
        self.buttonSubmit.setMaximumWidth(100)
        self.buttonSubmit.setText("Submeter") 
        self.buttonSubmit.clicked.connect(self.handleTemporalAnalysis)
        self.buttonSubmit.setDisabled(True)

        self.controlButtonSubmitEnability = ControlButtonEnability(self.companyName, self.buttonSubmit)
        self.companyName.textChanged.connect(self.controlButtonSubmitEnability.checkStatus)
        
        # Widgets dos gráficos
        self.bottom = QFrame()
        self.bottom.setFrameShape(QFrame.StyledPanel)

        self.image = QLabel()
        self.dirImage = QtGui.QPixmap('images/imagemPadrao.png')
        self.image.setPixmap(self.dirImage)
        self.image.setAlignment(QtCore.Qt.AlignCenter)

        self.buttonClean = QPushButton(self)
        self.buttonClean.setMaximumWidth(100)
        self.buttonClean.setText("Limpar") 
        self.buttonClean.clicked.connect(self.setDefaultImage)

        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(self.image)

        self.scrollArea = QScrollArea()
        self.scrollArea.setWidget(self.splitter)       
        
        # Grupo com os elementos que exibirão os dados da ação da bolsa de valores
        self.stockGroupBox = QGroupBox("Dados da Ação da Bolsa de Valores")
        layout = QFormLayout()
        layout.addRow(QLabel("Informe o símbolo da Ação:"), self.stockTicket)
        layout.addRow(QLabel("Nome da empresa:"), self.companyName)
        layout.addRow(QLabel("Setor:"), self.companySection)
        layout.addRow(QLabel("País:"), self.companyCountry)
        layout.addRow(QLabel("Moeda:"), self.currency)
        layout.addWidget(self.buttonSearch)
        layout.setVerticalSpacing(10)
        self.stockGroupBox.setLayout(layout)

        # Grupo com os elementos exibirão os valores para a análise temporal
        self.temporalAnalysisGroupBox = QGroupBox("Dados Série Temporal")
        layout = QFormLayout()
        layout.addRow(QLabel("Número de épocas de treinamento:"), self.epoch)
        layout.addRow(QLabel("Número de neurônios na camada oculta:"), self.totalHiddenLayers)
        layout.addRow(QLabel("Valor Learning Rate:"), self.learningRate)
        layout.addRow(QLabel("Valor Momentum:"), self.momentumValue)
        layout.addRow(QLabel("Data Inicial:"), self.inicialDate)
        layout.addRow(QLabel("Data Final:"), self.finalDate)
        layout.addWidget(self.buttonSubmit)
        layout.setVerticalSpacing(10)
        self.temporalAnalysisGroupBox.setLayout(layout)    
        
        '''Gerar Layouts'''
    def gerarLayouts(self):
        # Criando janela
        self.janelaAreaVisualizacao = QWidget(self)
        self.setCentralWidget(self.janelaAreaVisualizacao)     

        # Criando os layouts
        self.mainLayout = QHBoxLayout()
        self.leftLayout = QVBoxLayout()
        self.rightLayout = QHBoxLayout()

        # Adicionando os widgets
        self.leftLayout.addWidget(self.stockGroupBox)
        self.leftLayout.addWidget(self.temporalAnalysisGroupBox)
        self.rightLayout.addWidget(self.scrollArea)
        self.rightLayout.addWidget(self.buttonClean)

        # Adicionando layouts filhos na janela principal
        self.mainLayout.addLayout(self.leftLayout, 2)
        self.mainLayout.addLayout(self.rightLayout, 20)

        self.janelaAreaVisualizacao.setLayout(self.mainLayout)

    def setDefaultImage(self):
        self.image.setPixmap(self.dirImage)
        self.image.setAlignment(QtCore.Qt.AlignCenter)



def main():
    app = QApplication(sys.argv)
    win = MyWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
    
    
