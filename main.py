import pickle
import re
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
        self.XGBoost.toggled.connect(self.update_plot)
        self.CatBoost.toggled.connect(self.update_plot)
        self.LightGBM.toggled.connect(self.update_plot)
        self.EC8.toggled.connect(self.update_plot)
        self.ASCE.toggled.connect(self.update_plot)
        self.NZS_2.toggled.connect(self.update_plot)
        self.savePGA.clicked.connect(partial(self.save_PGA_image))
        self.savePFA.clicked.connect(partial(self.save_PFA_image))

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
            loaded_modelPGA_XGB = pickle.load(open("PGAXGBbest最终.pickle.dat", "rb"))
            loaded_modelPGA_CAT = pickle.load(open("PGACATbest最终.pickle.dat", "rb"))
            loaded_modelPGA_LGBM = pickle.load(open("PGALGBMbest最终.pickle.dat", "rb"))
            loaded_modelPFA_XGB = pickle.load(open("PFAXGBbest最终.pickle.dat", "rb"))
            loaded_modelPFA_CAT = pickle.load(open("PFACATbest最终.pickle.dat", "rb"))
            loaded_modelPFA_LGBM = pickle.load(open("PFALGBMbest最终.pickle.dat", "rb"))
        
            inputdata = np.array([])

            Stru = np.tile(textbox_values, (31, 1))
            T = np.arange(0, 3.1, 0.1)
            self.T=T
            inputdata = np.concatenate((Stru, np.transpose([T])), axis=1)
            inputdata = np.insert(inputdata, 8, inputdata[:,-1], axis=1)
            inputdata = np.delete(inputdata, -1, axis=1)
            FRS_PGA_XGB = loaded_modelPGA_XGB.predict(inputdata)
            FRS_PGA_CAT = loaded_modelPGA_CAT.predict(inputdata)
            FRS_PGA_LGBM = loaded_modelPGA_LGBM.predict(inputdata)
            FRS_PFA_XGB = loaded_modelPFA_XGB.predict(inputdata)
            FRS_PFA_CAT = loaded_modelPFA_CAT.predict(inputdata)
            FRS_PFA_LGBM = loaded_modelPFA_LGBM.predict(inputdata)
            self.FRS_PGA_XGB = FRS_PGA_XGB
            self.FRS_PGA_CAT = FRS_PGA_CAT
            self.FRS_PGA_LGBM = FRS_PGA_LGBM
            self.FRS_PFA_XGB = FRS_PFA_XGB
            self.FRS_PFA_CAT = FRS_PFA_CAT
            self.FRS_PFA_LGBM = FRS_PFA_LGBM
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

        cols_num = self.tableWidget.columnCount()

        FRS_PGA_XGB_list = list(FRS_PGA_XGB)
        FRS_PFA_XGB_list = list(FRS_PFA_XGB)
        FRS_PGA_CAT_list = list(FRS_PGA_CAT)
        FRS_PFA_CAT_list = list(FRS_PFA_CAT)
        FRS_PGA_LGBM_list = list(FRS_PGA_LGBM)
        FRS_PFA_LGBM_list = list(FRS_PFA_LGBM)

        for i in range(cols_num):           
            FRS_PGA_XGB2 = "{:.3f}".format(FRS_PGA_XGB_list[i])
            FRS_PFA_XGB2 = "{:.3f}".format(FRS_PFA_XGB_list[i])
            FRS_PGA_CAT2 = "{:.3f}".format(FRS_PGA_CAT_list[i])
            FRS_PFA_CAT2 = "{:.3f}".format(FRS_PFA_CAT_list[i])
            FRS_PGA_LGBM2 = "{:.3f}".format(FRS_PGA_LGBM_list[i])
            FRS_PFA_LGBM2 = "{:.3f}".format(FRS_PFA_LGBM_list[i])
            item_PGA_XGB = QTableWidgetItem(FRS_PGA_XGB2)
            item_PFA_XGB = QTableWidgetItem(FRS_PFA_XGB2)
            item_PGA_CAT = QTableWidgetItem(FRS_PGA_CAT2)
            item_PFA_CAT = QTableWidgetItem(FRS_PFA_CAT2)
            item_PGA_LGBM = QTableWidgetItem(FRS_PGA_LGBM2)
            item_PFA_LGBM = QTableWidgetItem(FRS_PFA_LGBM2)
            self.tableWidget.setItem(0, i, item_PGA_XGB)
            self.tableWidget.setItem(1, i, item_PFA_XGB)
            self.tableWidget.setItem(2, i, item_PGA_CAT)
            self.tableWidget.setItem(3, i, item_PFA_CAT)
            self.tableWidget.setItem(4, i, item_PGA_LGBM)
            self.tableWidget.setItem(5, i, item_PFA_LGBM)
    def update_plot(self):
        
        F1 = MyFigure(width=5, height=4, dpi=100)
        self.F1=F1
        F1.axes1 = F1.fig.add_subplot(111)
        if self.XGBoost.isChecked():
            F1.axes1.plot(self.T, self.FRS_PGA_XGB, color='blue', linestyle='-', linewidth=2, label='XGBoost')
        if self.CatBoost.isChecked():
            F1.axes1.plot(self.T, self.FRS_PGA_CAT, color='red', linestyle='-.', linewidth=2, label='CatBoost')
        if self.LightGBM.isChecked():
            F1.axes1.plot(self.T, self.FRS_PGA_LGBM, color='green', linestyle='--', linewidth=2, label='LightGBM')
        zh1 = float(self.zh1) 
        ECMAX = (3 * (1 + zh1)) / (1 + (1 - self.T / 0.3) ** 2) - 0.5 
        if self.EC8.isChecked():
            Ts_text = self.T1_2.text()
            if not Ts_text:
                QMessageBox.warning(self, 'Warning', 'Please enter a value for T1.', QMessageBox.Ok)
                return
            
            try:
                Ts = float(Ts_text)
                if Ts <= 0:
                    raise ValueError("Ts must be positive")
            except ValueError:
                QMessageBox.warning(self, 'Warning', 'Invalid value for T1. Please enter a positive number.', QMessageBox.Ok)
                return
            
            zh1 = float(self.zh1) 
            
            SA1_EC = np.zeros_like(self.T)
            try:
                SA1_EC = (3 * (1 + zh1)) / (1 + (1 - self.T / Ts) ** 2) - 0.5
            except Exception as e:
                print(f"Calculation error: {e}")
                return
            SA1_EC = np.clip(SA1_EC, 1, None)   
            F1.axes1.plot(self.T, SA1_EC, color='orange', linestyle='-', linewidth=1.5, label='EC8')
        
        a_p = [1 if t < 0.06 else 2.5 for t in self.T]
        zh1 = float(self.zh1)
        SA1_ASCE = [a * (1 + 2 * zh1) / 1.5 for a in a_p]
        if self.ASCE.isChecked():
            F1.axes1.plot(self.T, SA1_ASCE, color='darkviolet', linestyle='-', linewidth=1.5, label='ASCE 7-16')
        
        C = [None] * len(self.T)
        zh1 = float(self.zh1)
        for i in range(len(self.T)):
            if self.T[i] <= 0.75:
                C[i] = 2
            elif 0.75 < self.T[i] < 1.5:
                C[i] = 2 - (self.T[i] - 0.75) / (1.5 - 0.75) * (2.0 - 0.5)
            else:
                C[i] = 0.5

        if zh1 == 0:
            CH = 1
        elif 0 < zh1 < 0.2:
            CH = 1 + 10 * zh1
        else:
            CH = 3
        SA1_NZS = [CH * c for c in C]
        if self.NZS_2.isChecked():
            F1.axes1.plot(self.T, SA1_NZS, color='orange', linestyle='-.', linewidth=1.5, label='NZS 1170.5')   
        # F1.axes1.set_title("FRS/PGA")
        max_values = np.array([np.max(self.FRS_PGA_XGB),np.max(self.FRS_PGA_CAT),np.max(self.FRS_PGA_LGBM),np.max(ECMAX),np.max(SA1_ASCE),np.max(SA1_NZS)])
        min_values = np.array([np.min(self.FRS_PGA_XGB),np.min(self.FRS_PGA_CAT),np.min(self.FRS_PGA_LGBM),np.min(ECMAX),np.min(SA1_ASCE),np.min(SA1_NZS)])
        max_value = np.max(max_values)
        min_value = np.min(min_values)
        F1.axes1.set_ylim(min_value-0.5, np.ceil(max_value+0.5))
        F1.axes1.set_xticks(np.arange(0, 3.1, 0.5))
        F1.axes1.set_yticks(np.linspace(0, np.ceil(max_value+0.5), 5))
        F1.axes1.set_xlabel("$T_{a}$ (s)")
        F1.axes1.set_ylabel("FRS/PGA")
        F1.axes1.legend()
        width,height = self.graphicsView_2.width(),self.graphicsView_2.height()
        F1.resize(width*0.9,height*0.8)
        F1.fig.subplots_adjust(bottom=0.2, left=0.2)

        F2 = MyFigure(width=5, height=4, dpi=100)
        self.F2 = F2
        F2.axes1 = F2.fig.add_subplot(111)
        if self.XGBoost.isChecked():
            F2.axes1.plot(self.T, self.FRS_PFA_XGB, color='blue', linestyle='-', linewidth=2, label='XGBoost')
        if self.CatBoost.isChecked():  
            F2.axes1.plot(self.T, self.FRS_PFA_CAT, color='red', linestyle='-.', linewidth=2, label='CatBoost')
        if self.LightGBM.isChecked():
            F2.axes1.plot(self.T, self.FRS_PFA_LGBM, color='green', linestyle='--', linewidth=2, label='LightGBM')
        zh1 = float(self.zh1) 
        ECMAX = ((3 * (1 + zh1)) / (1 + (1 - self.T / 0.3) ** 2) - 0.5)*(1/(1+1.5*zh1))
        ECMAX2 = np.clip(ECMAX, 1/(1+1.5*zh1), None) 
        if self.EC8.isChecked():
            Ts_text = self.T1_2.text()
            if not Ts_text:
                QMessageBox.warning(self, 'Warning', 'Please enter a value for T1.', QMessageBox.Ok)
                return
            try:
                Ts = float(Ts_text)
                if Ts <= 0:
                    raise ValueError("Ts must be positive")
            except ValueError:
                QMessageBox.warning(self, 'Warning', 'Invalid value for T1. Please enter a positive number.', QMessageBox.Ok)
                return           
            SA1_EC = np.zeros_like(self.T)
            try:
                SA1_EC = ((3 * (1 + zh1)) / (1 + (1 - self.T / Ts) ** 2) - 0.5)*(1/(1+1.5*zh1))
            except Exception as e:
                print(f"Calculation error: {e}")
                return
            SA1_EC = np.clip(SA1_EC, 1/(1+1.5*zh1), None) 
            F2.axes1.plot(self.T, SA1_EC, color='orange', linestyle='-', linewidth=1.5, label='EC8')
      
        a_p = [1 if t < 0.06 else 2.5 for t in self.T]
        zh1 = float(self.zh1)
        SA1_ASCE = [a * (1 + 2 * zh1) / 1.5 for a in a_p]
        SA2_ASCE= [val * (1/(1+2*zh1)) for val in SA1_ASCE]
        if self.ASCE.isChecked():
            F2.axes1.plot(self.T, SA2_ASCE, color='darkviolet', linestyle='--', linewidth=1.5, label='ASCE 7-16')
        
        C = [None] * len(self.T)
        zh1 = float(self.zh1)
        for i in range(len(self.T)):
            if self.T[i] <= 0.75:
                C[i] = 2
            elif 0.75 < self.T[i] < 1.5:
                C[i] = 2 - (self.T[i] - 0.75) / (1.5 - 0.75) * (2.0 - 0.5)
            else:
                C[i] = 0.5

        if zh1 == 0:
            CH = 1
        elif 0 < zh1 < 0.2:
            CH = 1 + 10 * zh1
        else:
            CH = 3
        SA1_NZS = [CH * c for c in C]
        SA2_NZS= [val * (1/(1+2*zh1)) for val in SA1_NZS]
        if self.NZS_2.isChecked():
            F2.axes1.plot(self.T, SA2_NZS, color='orange', linestyle='-.', linewidth=1.5, label='NZS 1170.5')
        # F1.axes1.set_title("FRS/PFA")
        
        max_values = np.array([np.max(self.FRS_PFA_XGB),np.max(self.FRS_PFA_CAT),np.max(self.FRS_PFA_LGBM),np.max(ECMAX),np.max(SA2_ASCE),np.max(SA2_NZS)])
        min_values = np.array([np.min(self.FRS_PFA_XGB),np.min(self.FRS_PFA_CAT),np.min(self.FRS_PFA_LGBM),np.min(ECMAX),np.min(SA2_ASCE),np.min(SA2_NZS)])
        max_value = np.max(max_values)
        min_value = np.min(min_values)
        F2.axes1.set_ylim(min_value-0.5, np.ceil(max_value+0.5))
        F2.axes1.set_xticks(np.arange(0, 3.1, 0.5))
        F2.axes1.set_yticks(np.linspace(0, np.ceil(max_value+0.5), 5))
        F2.axes1.set_xlabel("$T_{a}$ (s)")
        F2.axes1.set_ylabel("FRS/PFA")
        F2.axes1.legend()
        width,height = self.graphicsView.width(),self.graphicsView.height()
        F2.resize(width*0.9,height*0.8)
        F2.fig.subplots_adjust(bottom=0.2, left=0.2)

        self.scene = QGraphicsScene()
        self.scene.addWidget(F1)
        self.graphicsView_2.setScene(self.scene)
        self.scene2 = QGraphicsScene()
        self.scene2.addWidget(F2)
        self.graphicsView.setScene(self.scene2)

    def Savetxt(self):
        try:
            data = (
                    np.transpose([self.T]),
                    np.transpose([self.FRS_PGA_XGB]),
                    np.transpose([self.FRS_PFA_XGB]),
                    np.transpose([self.FRS_PGA_CAT]),
                    np.transpose([self.FRS_PFA_CAT]),
                    np.transpose([self.FRS_PGA_LGBM]),
                    np.transpose([self.FRS_PFA_LGBM])
                )
            data2 = np.concatenate(data, axis=1)
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
                column_names = ['Ta', 'FRS/PGA (XGBoost)', 'FRS/PFA (XGBoost)', 'FRS/PGA (CatBoost)', 'FRS/PFA (CatBoost)', 'FRS/PGA (LightGBM)', 'FRS/PFA (LightGBM)']
                header_lines = [line1, line2, line3, line4, line6, line7, line8, line9, line10, line5]
                header = '\n'.join(header_lines) + '\n' + '\t'.join(column_names)
                np.savetxt(filePath, data2, fmt='%0.3f', header=header, comments='')
        except Exception as e:
            QMessageBox.warning(self, 'Warning', f"Error:{e}.", QMessageBox.Ok)

    def save_PGA_image(self):
        try:
            fig = self.F1.fig
            file_dialog = QFileDialog(self)
            file_dialog.setNameFilter("PNG files (*.png);;JPEG files (*.jpg *.jpeg)")
            if file_dialog.exec_():
                file_path = file_dialog.selectedFiles()[0]
                fig.savefig(file_path)
        except Exception as e:
            QMessageBox.warning(self, 'Warning', f"Error: {e}", QMessageBox.Ok)

    def save_PFA_image(self):
        try:
            fig = self.F2.fig
            file_dialog = QFileDialog(self)
            file_dialog.setNameFilter("PNG files (*.png);;JPEG files (*.jpg *.jpeg)")
            if file_dialog.exec_():
                file_path = file_dialog.selectedFiles()[0]
                fig.savefig(file_path)
        except Exception as e:
            QMessageBox.warning(self, 'Warning', f"Error: {e}", QMessageBox.Ok)

        




if __name__ == "__main__":
    app = QApplication(sys.argv)
    demo = MyUiComputer()
    demo.show()
    sys.exit(app.exec_())
