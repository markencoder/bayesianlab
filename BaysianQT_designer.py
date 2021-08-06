from PySide2.QtCore import QAbstractTableModel, Qt
from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton, QLineEdit, QFileDialog, QTableView, \
    QPlainTextEdit, QWidget, QGraphicsScene,QMessageBox,QProgressBar
from PySide2.QtUiTools import QUiLoader
#from PyQt5.QtCore import QBasicTimer
from PySide2 import QtWidgets
import pyqtgraph as pg
import pandas as pd
import numpy as np
from pgmpy.estimators import PC
import matplotlib.pyplot as plt
import sys
from pgmpy.estimators import K2Score
from pgmpy.estimators import PC, HillClimbSearch, ExhaustiveSearch
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
import seaborn as sns
import bnlearn as bn
matplotlib.use("Qt5Agg")  # 声明使用QT5


#Pyside2中嵌入Matplotlib的绘图 类方法
class MyFigureCanvas(FigureCanvas):
    '''
    通过继承FigureCanvas类，使得该类既是一个PyQt5的Qwidget，又是一个matplotlib的FigureCanvas，这是连接pyqt5与matplotlib的关键
    '''

    def __init__(self, xlabel,parent=None, width=6, height=6, dpi=100):
        # 创建一个Figure
        fig = plt.Figure(figsize=(width, height), dpi=dpi, tight_layout=True)  # tight_layout: 用于去除画图时两边的空白

        FigureCanvas.__init__(self, fig)  # 初始化父类
        self.setParent(parent)

        self.axes = fig.add_subplot(111)  # 添加子图
        self.axes.set_ylim(0,1)
        self.axes.set_xlabel(xlabel,fontsize=10)
        self.axes.tick_params(labelsize=7,labelrotation=45)
        self.axes.set_ylabel('frequency', fontsize=10)
        #self.axes.spines['top'].set_visible(False)  # 去掉绘图时上面的横线
        #self.axes.spines['right'].set_visible(False)  # 去掉绘图时右面的横线


#在tableview中展示dataframe的类方法
class PdTable(QAbstractTableModel):
    def __init__(self, data, showAllColumn=False):
        QAbstractTableModel.__init__(self)
        self.showAllColumn = showAllColumn
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if type(self._data.columns[col]) == tuple:
                return self._data.columns[col][-1]
            else:
                return self._data.columns[col]
        elif orientation == Qt.Vertical and role == Qt.DisplayRole:
            return (self._data.axes[0][col])
        return None


class Loginwindow():
    def __init__(self):
        self.ui = QUiLoader().load('login.ui')
        ### 登录界面
        self.ui.loginButton.clicked.connect(self.mainshow)

    def mainshow(self):

        self.ui2 = Mainwindow()
        self.ui2.ui.show()
        self.ui.hide()

#创建主窗口
class Mainwindow():
    def __init__(self):
        # 从文件中加载UI定义

        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit

        self.ui = QUiLoader().load('Datainput.ui')

        #文件输入 按钮的槽
        self.ui.Selectbutton.clicked.connect(self.Fileselect)
        #下一步 按钮的槽
        self.ui.Nextbutton.clicked.connect(self.childShowFun1)
        #数据处理 按钮的槽
        self.ui.Runbutton.clicked.connect(self.process)

        self.ui.tableView_result_2.doubleClicked.connect(self.dataview)

    #出现子窗口的方法
    def childShowFun1(self):
        self.ui2 = ChildWindow()
        self.ui2.ui.show()
        self.ui.hide()

    #数据处理的方法
    def process(self):
        global data_processed
        search_result_model = PdTable(data_processed)
        self.ui.tableView_result_2.setModel(search_result_model)    #展示数据处理结果

    #文件选择的方法
    def Fileselect(self):
        global data_processed
        Selectinterface = QUiLoader().load('self.ui')     #导出一个空白的文件路径窗口
        filePath, _ = QFileDialog.getOpenFileName(
            Selectinterface,  # 父窗口对象
            "选择你要上传的文件",  # 标题
            r"F:\郑意德\2021研究\横向课题-贝叶斯软件设计\data",  # 起始目录
            "数据类型 (*.xls *.xlsx *.csv)"  # 选择类型过滤项，过滤内容在括号中
        )
        self.ui.line1.setText(filePath)
        data = pd.read_csv(filePath, header=0)
        data = data[
            ['SalePrice', 'Lot Frontage', 'Lot Area', 'Overall Qual', 'Overall Cond', 'Kitchen Qual', 'House Style',
             'Roof Style']]
        data_nna = data.dropna()
        search_result_model = PdTable(data_nna)
        self.ui.tableView_result.setModel(search_result_model)       #展示未处理的数据

        #数据处理部分
        data_nna['sale_after'] = data_nna['SalePrice'].apply(lambda
                                                                 x: '12789-198,342' if 12789 <= x <= 198342 else '198342-383895' if 198342 < x <= 383895 else '383895-569448' if 383895 < x <= 569448 else '569448-755000')
        data_nna['lotf_after'] = data_nna['Lot Frontage'].apply(
            lambda x: '21-94' if 21 <= x <= 94 else '94-167' if 94 < x <= 167 else '167-240' if 167 < x <= 240 else '240-313')
        data_nna['lota_after'] = data_nna['Lot Area'].apply(lambda
                                                                x: '1300-54786' if 1300 <= x <= 54786 else '54786-108272' if 54786 < x <= 108272 else '108272-161758' if 108272 < x <= 161758 else '161758-215245')
        data_processed = data_nna[
            ['sale_after', 'lotf_after', 'lota_after', 'Overall Qual', 'Overall Cond', 'Kitchen Qual', 'House Style',
             'Roof Style']]

        #将data_processed里面全部变成str类型
        data_processed = data_processed.applymap(str)

    #展示数据分布
    def dataview(self):

        #data = self.ui.tableView_result_2.currentColumn().data()
        column = self.ui.tableView_result_2.currentIndex().column()
        self.draw = drawWindow(column)
        self.draw.ui.show()
        #self.ui.hide()

class drawWindow():

    def __init__(self,column):
        global data_processed
        self.ui = QUiLoader().load('data-view.ui')
        #self.ui.show()
        self.dataview1 = MyFigureCanvas(width=self.ui.dataview.width() / 101,
                                                     height=self.ui.dataview.height() / 101,
                                                     xlabel=data_processed.columns[column])

        view = data_processed.iloc[:, column].value_counts() / len(data_processed.iloc[:, column])

        self.dataview1.axes.bar(view.index,view)
        # self.gv_visual_data_content1.axes.legend()

        self.graphic_scene = QGraphicsScene()  # 创建一个QGraphicsScene
        self.graphic_scene.addWidget(self.dataview1)  # 把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到放到QGraphicsScene中的
        self.ui.dataview.setScene(self.graphic_scene)  # 把QGraphicsScene放入QGraphicsView
        self.ui.dataview.show()



#创建子窗口1 的类
class ChildWindow():
    def __init__(self):
        #导入窗口2
        self.t = 0
        self.ui = QUiLoader().load('dataprocessing.ui')

        #设置窗口2的下一步按钮  的槽
        self.ui.Nextbutton2.clicked.connect(self.childShowFun2)

        #设置窗口2的多选项按钮  的槽
        self.ui.buttonGroup.buttonClicked.connect(self.methodselect)

        self.ui.BNButton.clicked.connect(self.BNdrawing)
        self.ui.progressBar.setValue(0)
       # self.timer = QBasicTimer()



    #创建窗口3的方法
    def childShowFun2(self):
        self.ui2 = ChildWindow2()
        self.ui2.ui.show()
        self.ui.hide()

    #模型选择方法
    def methodselect(self):
        global data_processed
        global model_struct
        self.ui.progressBar.setValue(0)
        #选择的id
        selectedbutton = self.ui.buttonGroup.checkedId()

        if selectedbutton == -2:
            self.completed = 0
            while self.completed < 100:
                self.completed += 0.001
                self.ui.progressBar.setValue(self.completed)
            #贝叶斯网络运算
            scoring_method = K2Score(data=data_processed)
            est = HillClimbSearch(data=data_processed)
            estimated_model = est.estimate(scoring_method=scoring_method, max_indegree=4, max_iter=int(1e4))
            model_struct = BayesianModel(estimated_model.edges())
            model_struct.fit(data=data_processed, estimator=MaximumLikelihoodEstimator)
            df = pd.DataFrame(list(estimated_model.edges()))
            df = df.rename(columns={0: '父节点', 1: '子节点'})
            netgraph = PdTable(df)
            self.ui.graphview.setModel(netgraph)
            if model_struct.check_model() != True:
                self.ui.v_result.setText('模型异常，请更换方法')
            else:
                self.ui.v_result.setText('模型正常，可继续使用')

            self.t = self.t + 1
            #self.ui.Evidence.show()


        elif selectedbutton == -4:
            self.completed = 0
            while self.completed < 100:
                self.completed += 1
                self.ui.progressBar.setValue(self.completed)
            est = PC(data_processed)
            estimated_model = est.estimate(variant='orig', max_cond_vars=4)
            model_struct = BayesianModel(estimated_model.edges())
            model_struct.fit(data=data_processed, estimator=MaximumLikelihoodEstimator)
            df = pd.DataFrame(list(estimated_model.edges()))
            df = df.rename(columns={0: '父节点', 1: '子节点'})
            netgraph = PdTable(df)
            self.ui.graphview.setModel(netgraph)
            if model_struct.check_model() != True:
                self.ui.v_result.setText('模型异常，请更换方法')
            else:
                self.ui.v_result.setText('模型正常，可继续使用')
            self.t = self.t + 1
            #self.ui.Evidence.show()

    def BNdrawing(self):
        if self.t == 0:
            MainWindow = QMainWindow()
            MessageBox = QMessageBox()
            MessageBox.warning(MainWindow, "Bayesian-Network", "请先选择网络训练方法并对数据训练，生成B-N网络后再进行绘图！")
        else:
            bn.plot(model_struct, figsize=(15, 12))



class ChildWindow2():
    def __init__(self):
        global data_processed

        #导入窗口3
        self.ui = QUiLoader().load('modelinference.ui')

        #目标节点按钮选择 槽
        self.ui.TargetcomboBox.addItems(list(data_processed.columns))
        # 证据节点按钮选择 槽
        self.ui.EvidencecomboBox.addItems(list(data_processed.columns))

        # 区间选择节点按钮信号 槽
        self.ui.EvidencecomboBox.currentIndexChanged.connect(self.intervalselect)
        # 模型推断 槽
        self.ui.IntervalcomboBox.currentIndexChanged.connect(self.modelinference)

        self.ui.pushButton.clicked.connect(self.exit)


    def intervalselect(self):
        global data_processed
        global model_struct

        #实例化窗口
        self.gv_visual_data_content1 = MyFigureCanvas(width=self.ui.Evidence.width() / 101,
                                                      height=self.ui.Evidence.height() / 101,
                                                      xlabel=self.ui.EvidencecomboBox.currentText())  # 实例化一个FigureCanvas

        #清楚区间选择的选项
        self.ui.IntervalcomboBox.clear()
        #读取证据节点的输入
        variable = self.ui.EvidencecomboBox.currentText()
        #将对应证据节点的区间输入到按钮中
        self.ui.IntervalcomboBox.addItems([str(i) for i in data_processed[variable].unique().tolist()])

        model_infer = VariableElimination(model_struct)

        #画证据节点的分布图
        q_evidence = model_infer.query(variables=[self.ui.EvidencecomboBox.currentText()])
        self.gv_visual_data_content1.axes.bar(q_evidence.state_names[self.ui.EvidencecomboBox.currentText()],q_evidence.values,label=self.ui.EvidencecomboBox.currentText())
        #self.gv_visual_data_content1.axes.legend()

        self.graphic_scene = QGraphicsScene()  # 创建一个QGraphicsScene
        self.graphic_scene.addWidget(self.gv_visual_data_content1)  # 把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到放到QGraphicsScene中的
        self.ui.Evidence.setScene(self.graphic_scene)  # 把QGraphicsScene放入QGraphicsView
        self.ui.Evidence.show()


    def modelinference(self):
        global model_struct

        #画目标节点修正前与修正后的图
        self.gv_visual_data_content2 = MyFigureCanvas(width=self.ui.Targetbefore.width() / 101,
                                                      height=self.ui.Targetbefore.height() / 101,
                                                      xlabel=self.ui.TargetcomboBox.currentText())  # 实例化一个FigureCanvas

        self.gv_visual_data_content3 = MyFigureCanvas(width=self.ui.Targetafter.width() / 101,
                                                      height=self.ui.Targetafter.height() / 101,
                                                      xlabel=self.ui.TargetcomboBox.currentText())  # 实例化一个FigureCanvas

        model_infer = VariableElimination(model_struct)
        q_before = model_infer.query(variables=[self.ui.TargetcomboBox.currentText()])

        #bar(q.state_names[目标节点],q.values,label='''证据节点 : 证据节点的选择区间''')

        self.gv_visual_data_content2.axes.bar(q_before.state_names[self.ui.TargetcomboBox.currentText()],
                                              q_before.values, label='Evidence distribution')
        self.graphic_scene = QGraphicsScene()  # 创建一个QGraphicsScene
        self.graphic_scene.addWidget(
            self.gv_visual_data_content2)  # 把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到放到QGraphicsScene中的
        self.ui.Targetbefore.setScene(self.graphic_scene)  # 把QGraphicsScene放入QGraphicsView
        self.ui.Targetbefore.show()

        q_after = model_infer.query(variables=[self.ui.TargetcomboBox.currentText()], evidence={self.ui.EvidencecomboBox.currentText(): self.ui.IntervalcomboBox.currentText()})

        self.gv_visual_data_content3.axes.bar(q_after.state_names[self.ui.TargetcomboBox.currentText()],
                                              q_after.values, label=self.ui.TargetcomboBox.currentText())
        self.graphic_scene = QGraphicsScene()  # 创建一个QGraphicsScene
        self.graphic_scene.addWidget(
            self.gv_visual_data_content3)  # 把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到放到QGraphicsScene中的
        self.ui.Targetafter.setScene(self.graphic_scene)  # 把QGraphicsScene放入QGraphicsView
        self.ui.Targetafter.show()
    def exit(self):
        sys.exit()


if __name__ == '__main__':
    app = QApplication([])
    stats = Loginwindow()
    stats.ui.show()
    app.exec_()
