import matplotlib

matplotlib.use('Qt5Agg')

from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class VectorBinary(QDialog):

    def __init__(self, df, lines, *args, **kwargs):
        super(VectorBinary, self).__init__(*args, **kwargs)
        self.setWindowTitle("HELLO!")
        self.lines = lines
        self.amount = 0
        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.data = df
        self.x = self.data.columns[0]
        self.y = self.data.columns[1]
        self.grouping_column = self.data.columns[-1]

        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        previous_button = QPushButton()
        previous_button.setText("Previous")
        previous_button.clicked.connect(self.previous_action)

        show_all_button = QPushButton()
        show_all_button.setText("Show all")
        show_all_button.clicked.connect(self.show_all_action)

        next_button = QPushButton()
        next_button.setText("Next")
        next_button.clicked.connect(self.next_action)

        layout_button = QHBoxLayout()
        layout_button.addWidget(previous_button)
        layout_button.addWidget(show_all_button)
        layout_button.addWidget(next_button)

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        toolbar = NavigationToolbar(self.canvas, self)

        self.init_plot()

        layout_plot = QVBoxLayout()
        layout_plot.addWidget(toolbar)
        layout_plot.addWidget(self.canvas)

        self.plot = QWidget()
        self.plot.setLayout(layout_plot)

        action = QWidget()
        action.setLayout(layout_button)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.plot)
        self.layout.addWidget(action)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
        # for i in range(20):
        #     self.next_action()

    def init_plot(self):
        temp_data = self.data.groupby(self.grouping_column)
        for name, group in temp_data:
            self.canvas.axes.scatter(group[self.x], group[self.y], label=name)

    def previous_action(self):
        if self.amount > 0:
            self.canvas.axes.cla()
            self.init_plot()
            self.amount -= 1
            for index in range(self.amount):
                if self.lines[index]['column'] == self.x:
                    self.canvas.axes.plot([self.lines[index]['value'], self.lines[index]['value']],
                                          [self.data[self.y].min(), self.data[self.y].max()],
                                          'r')
                else:
                    self.canvas.axes.plot([self.data[self.x].min(), self.data[self.x].max()],
                                          [self.lines[index]['value'], self.lines[index]['value']],
                                          'g')
        self.canvas.draw_idle()

    def show_all_action(self):
        for i in range(len(self.lines) + 1):
            self.next_action()

    def next_action(self):
        if self.amount < len(self.lines):
            if self.lines[self.amount]['column'] == self.x:
                self.canvas.axes.plot([self.lines[self.amount]['value'], self.lines[self.amount]['value']],
                                      [self.data[self.y].min(), self.data[self.y].max()],
                                      'r')
            else:
                self.canvas.axes.plot([self.data[self.x].min(), self.data[self.x].max()],
                                      [self.lines[self.amount]['value'], self.lines[self.amount]['value']],
                                      'g')
            self.amount += 1
        self.canvas.draw_idle()
