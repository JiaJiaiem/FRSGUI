import pickle
import xgboost as xgb
import sys
from PyQt5.QtWidgets import QMessageBox, QApplication, QFileDialog, QWidget
import numpy as np
from FRS import Ui_GeneratingFloorResponseSpectraDirectly
from functools import partial
import sys
import pickle
import re
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib
 
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import pyplot
 
pyplot.rcParams['font.sans-serif'] = ['Times New Roman']
pyplot.rcParams['axes.unicode_minus'] = False



class MyFigure(FigureCanvasQTAgg):
   def __init__(self,width=5,height=4,dpi = 100):
      self.fig = Figure(figsize=(width,height),dpi=dpi)
      super(MyFigure, self).__init__(self.fig)
 
   def plot(self,x,y):
      self.axes0 = self.fig.add_subplot(111)
      self.axes0.plot(x,y)

class MyUiComputer(Ui_GeneratingFloorResponseSpectraDirectly, QWidget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.predict.clicked.connect(partial(self.FRSPredict))
        self.save.clicked.connect(partial(self.Savetxt))

    def FRSPredict(self,window):
        try:
            N1 = self.N.text()
            Lt1 = self.Lt.text()
            Ll1 = self.Ll.text() 
            H1 = self.H.text()
            zh1 = self.zh.text()
            B1 = self.B.text()
            textbox_values = [N1, Lt1, Ll1, H1, zh1, B1]
            I1 = self.I.currentText()
            F1 = self.F.currentText()
            d1 = self.d.currentText()
            S1 = self.d_2.currentText()
            if any([not N1, not Lt1, not Ll1, not H1, not zh1, not B1]):
                QMessageBox.warning(self, 'Warning', 'Please enter a value for all fields.', QMessageBox.Ok)
                return
            textbox_values.append(d1)
            textbox_values = [float(x) for x in textbox_values]
            if F1 == "lateral":
                textbox_values.append(1)
            else:
                textbox_values.append(0)

            if I1 == "concrete":
                textbox_values.extend([1, 0, 0])
            elif I1 == "steel":
                textbox_values.extend([0, 1, 0])
            else:
                textbox_values.extend([0, 0, 1])

            if S1 == "C":
                textbox_values.extend([1, 0, 0])
            elif S1 == "D":
                textbox_values.extend([0, 1, 0])
            else:
                textbox_values.extend([0, 0, 1])
            loaded_model1 = pickle.load(open("PGAXGB.pickle.dat", "rb"))
            loaded_model2 = pickle.load(open("PFAXGB.pickle.dat", "rb"))
        
            inputdata = np.array([])

            Stru = np.tile(textbox_values, (31, 1))
            T = np.arange(0, 3.1, 0.1)
            inputdata = np.concatenate((Stru, np.transpose([T])), axis=1)
            inputdata = np.insert(inputdata, 8, inputdata[:,-1], axis=1)
            inputdata = np.delete(inputdata, -1, axis=1)
            FRS_PGA = loaded_model1.predict(inputdata)
            FRS_PFA = loaded_model2.predict(inputdata)
            self.FRS_PGA = FRS_PGA
            self.FRS_PFA = FRS_PFA
            self.T = T
            self.N1=N1
            self.Ll1=Ll1
            self.Lt1=Lt1
            self.H1=H1
            self.S1=S1
            self.zh1=zh1
            self.B1=B1
            self.I1=I1
            self.F1=F1
            self.d1=d1


        except Exception as e:
            QMessageBox.warning(self, 'Warning', f"Error:{e}.", QMessageBox.Ok)

        F1 = MyFigure(width=5, height=4, dpi=100)
        F1.axes1 = F1.fig.add_subplot(111)
        F1.axes1.plot(T, FRS_PGA)
        # F1.axes1.set_title("FRS/PGA")
        F1.axes1.set_ylim(np.min(FRS_PGA)-0.5, np.ceil(np.max(FRS_PGA)+0.5))
        F1.axes1.set_xticks(np.arange(0, 3.1, 0.5))
        F1.axes1.set_yticks(np.linspace(0, np.ceil(np.max(FRS_PGA)+0.5), 5))
        F1.axes1.set_xlabel("T (s)")
        F1.axes1.set_ylabel("FRS/PGA")
        width,height = self.graphicsView_2.width(),self.graphicsView_2.height()
        F1.resize(width*0.9,height*0.8)
        F1.fig.subplots_adjust(bottom=0.3, left=0.25)

        F2 = MyFigure(width=5, height=4, dpi=100)
        F2.axes1 = F2.fig.add_subplot(111)
        F2.axes1.plot(T, FRS_PFA)
        # F1.axes1.set_title("FRS/PFA")
        F2.axes1.set_ylim(np.min(FRS_PFA)-0.5, np.ceil(np.max(FRS_PFA)+0.5))
        F2.axes1.set_xticks(np.arange(0, 3.1, 0.5))
        F2.axes1.set_yticks(np.linspace(0, np.ceil(np.max(FRS_PFA)+0.5), 5))
        F2.axes1.set_xlabel("T (s)")
        F2.axes1.set_ylabel("FRS/PFA")
        width,height = self.graphicsView.width(),self.graphicsView.height()
        F2.resize(width*0.9,height*0.8)
        F2.fig.subplots_adjust(bottom=0.3, left=0.25)

        self.scene = QGraphicsScene()
        self.scene.addWidget(F1)
        self.graphicsView_2.setScene(self.scene)
        self.scene2 = QGraphicsScene()
        self.scene2.addWidget(F2)
        self.graphicsView.setScene(self.scene2)

        cols_num = self.tableWidget.columnCount()

        FRS_PGA_list = list(FRS_PGA)
        FRS_PFA_list = list(FRS_PFA)

        for i in range(cols_num):           
            FRS_PGA2 = "{:.3f}".format(FRS_PGA_list[i])
            FRS_PFA2 = "{:.3f}".format(FRS_PFA_list[i])
            item_pga = QTableWidgetItem(FRS_PGA2)
            item_pfa = QTableWidgetItem(FRS_PFA2)
            self.tableWidget.setItem(0, i, item_pga)
            self.tableWidget.setItem(1, i, item_pfa)

    def Savetxt(self):
        try:
            data = np.concatenate((np.transpose([self.T]), np.transpose([self.FRS_PGA]), np.transpose([self.FRS_PFA])), axis=1)

            options = QFileDialog.Options()
            filePath, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Text Files (*.txt)", options=options)

            if filePath:
                line1 = 'Number of stories: {}'.format(self.N1)
                line2 = 'Lateral length (m): {}'.format(self.Lt1)
                line3 = 'Longitudinal length (m): {}'.format(self.Ll1)
                line4 = 'Total height (m): {}'.format(self.H1)
                line6 = 'Relative height of NSCs: {}'.format(self.zh1)
                line7 = 'Number of basement stories: {}'.format(self.B1)
                line8 = 'Structure types: {}'.format(self.I1)
                line9 = 'Direction: {}'.format(self.F1)
                line10 = 'Damping ratio: {}'.format(self.d1)
                line5 = 'Soil condition: {}'.format(self.S1)
                column_names = ['T', 'FRS/PGA', 'FRS/PFA']
                header_lines = [line1, line2, line3, line4, line6, line7, line8, line9, line10, line5]
                header = '\n'.join(header_lines) + '\n' + '\t'.join(column_names)
                np.savetxt(filePath, data, fmt='%0.3f', header=header, comments='')
        except Exception as e:
            QMessageBox.warning(self, 'Warning', f"Error:{e}.", QMessageBox.Ok)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    demo = MyUiComputer()
    demo.show()
    sys.exit(app.exec_())
