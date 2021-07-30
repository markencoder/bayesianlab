from PySide2.QtCore import QAbstractTableModel, Qt
from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton, QLineEdit, QFileDialog, QTableView, \
    QPlainTextEdit, QWidget, QGraphicsScene
from PySide2.QtUiTools import QUiLoader
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

matplotlib.use("Qt5Agg")  # 声明使用QT5


#Pyside2中嵌入Matplotlib的绘图 类方法
class MyFigureCanvas(FigureCanvas):
    '''
    通过继承FigureCanvas类，使得该类既是一个PyQt5的Qwidget，又是一个matplotlib的FigureCanvas，这是连接pyqt5与matplotlib的关键
    '''

    def __init__(self, parent=None, width=10, height=5, dpi=100):
        # 创建一个Figure
        fig = plt.Figure(figsize=(width, height), dpi=dpi, tight_layout=True)  # tight_layout: 用于去除画图时两边的空白

        FigureCanvas.__init__(self, fig)  # 初始化父类
        self.setParent(parent)

        self.axes = fig.add_subplot(111)  # 添加子图
        self.axes.spines['top'].set_visible(False)  # 去掉绘图时上面的横线
        self.axes.spines['right'].set_visible(False)  # 去掉绘图时右面的横线


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
                                                                 x: '12789-127500' if 12789 <= x <= 127500 else '127500-157500' if 127500 < x <= 157500 else '157500-212350' if 157500 < x <= 212350 else '212350-755000')
        data_nna['lotf_after'] = data_nna['Lot Frontage'].apply(
            lambda x: '21-58' if 21 <= x <= 58 else '58-68' if 58 < x <= 68 else '68-80' if 68 < x <= 80 else '80-313')
        data_nna['lota_after'] = data_nna['Lot Area'].apply(lambda
                                                                x: '1300-7225' if 1300 <= x <= 7225.25 else '7225-9248' if 7225.25 < x <= 9248.5 else '9248-11207' if 9248.5 < x <= 11207.75 else '11207-215245')
        data_processed = data_nna[
            ['sale_after', 'lotf_after', 'lota_after', 'Overall Qual', 'Overall Cond', 'Kitchen Qual', 'House Style',
             'Roof Style']]

        #将data_processed里面全部变成str类型
        data_processed = data_processed.applymap(str)


#创建子窗口1 的类
class ChildWindow():
    def __init__(self):
        #导入窗口2
        self.ui = QUiLoader().load('dataprocessing.ui')

        #设置窗口2的下一步按钮  的槽
        self.ui.Nextbutton2.clicked.connect(self.childShowFun2)

        #设置窗口2的多选项按钮  的槽
        self.ui.buttonGroup.buttonClicked.connect(self.methodselect)

    #创建窗口3的方法
    def childShowFun2(self):
        self.ui2 = ChildWindow2()
        self.ui2.ui.show()
        self.ui.hide()

    #模型选择方法
    def methodselect(self):
        global data_processed
        global model_struct

        #选择的id
        selectedbutton = self.ui.buttonGroup.checkedId()

        if selectedbutton == -2:

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


        elif selectedbutton == -4:
            est = PC(data_processed)
            estimated_model = est.estimate(variant='orig', max_cond_vars=4)
            model_struct = BayesianModel(estimated_model.edges())
            model_struct.fit(data=data_processed, estimator=MaximumLikelihoodEstimator)
            self.listView.setModel(list(model_struct.edges()))
            self.ui.v_result.setText('模型异常，请更换方法')


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


    def intervalselect(self):
        global data_processed
        global model_struct

        #实例化窗口
        self.gv_visual_data_content1 = MyFigureCanvas(width=self.ui.Evidence.width() / 101,
                                                      height=self.ui.Evidence.height() / 101,
                                                      )  # 实例化一个FigureCanvas

        #清楚区间选择的选项
        self.ui.IntervalcomboBox.clear()
        #读取证据节点的输入
        variable = self.ui.EvidencecomboBox.currentText()
        #将对应证据节点的区间输入到按钮中
        self.ui.IntervalcomboBox.addItems([str(i) for i in data_processed[variable].unique().tolist()])

        model_infer = VariableElimination(model_struct)

        #画证据节点的分布图
        q_evidence = model_infer.query(variables=[self.ui.EvidencecomboBox.currentText()])
        self.gv_visual_data_content1.axes.bar(q_evidence.state_names[self.ui.EvidencecomboBox.currentText()],q_evidence.values,label='Evidence distribution')
        self.graphic_scene = QGraphicsScene()  # 创建一个QGraphicsScene
        self.graphic_scene.addWidget(
            self.gv_visual_data_content1)  # 把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到放到QGraphicsScene中的
        self.ui.Evidence.setScene(self.graphic_scene)  # 把QGraphicsScene放入QGraphicsView
        self.ui.Evidence.show()


    def modelinference(self):
        global model_struct

        #画目标节点修正前与修正后的图
        self.gv_visual_data_content2 = MyFigureCanvas(width=self.ui.Targetbefore.width() / 101,
                                                      height=self.ui.Targetbefore.height() / 101,
                                                      )  # 实例化一个FigureCanvas

        self.gv_visual_data_content3 = MyFigureCanvas(width=self.ui.Targetafter.width() / 101,
                                                      height=self.ui.Targetafter.height() / 101,
                                                      )  # 实例化一个FigureCanvas

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
                                              q_after.values, label='Evidence distribution')
        self.graphic_scene = QGraphicsScene()  # 创建一个QGraphicsScene
        self.graphic_scene.addWidget(
            self.gv_visual_data_content3)  # 把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到放到QGraphicsScene中的
        self.ui.Targetafter.setScene(self.graphic_scene)  # 把QGraphicsScene放入QGraphicsView
        self.ui.Targetafter.show()


if __name__ == '__main__':
    app = QApplication([])
    stats = Mainwindow()
    stats.ui.show()
    app.exec_()
