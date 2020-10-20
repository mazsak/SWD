from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem
from pandas import DataFrame


class Table_UI(QWidget):

    def __init__(self, df: DataFrame):
        super().__init__()
        self.data = df
        self.table_ui: QTableWidget = QTableWidget()

        self.init_layout()

    def init_layout(self):
        self.layout: QVBoxLayout = QVBoxLayout()
        self.layout.addWidget(self.table_ui)
        self.setLayout(self.layout)

    def create_table_ui(self):
        # self.table_ui.setRowCount(len(self.table.index))
        # self.table_ui.setRowCount(1)
        # self.table_ui.setColumnCount(len(self.table.columns))
        # self.table_ui.setColumnCount(1)
        self.table_ui.setRowCount(4)
        self.table_ui.setColumnCount(2)
        self.table_ui.setItem(0, 0, QTableWidgetItem("Cell (1,1)"))
        self.table_ui.setItem(0, 1, QTableWidgetItem("Cell (1,2)"))
        self.table_ui.setItem(1, 0, QTableWidgetItem("Cell (2,1)"))
        self.table_ui.setItem(1, 1, QTableWidgetItem("Cell (2,2)"))
        self.table_ui.setItem(2, 0, QTableWidgetItem("Cell (3,1)"))
        self.table_ui.setItem(2, 1, QTableWidgetItem("Cell (3,2)"))
        self.table_ui.setItem(3, 0, QTableWidgetItem("Cell (4,1)"))
        self.table_ui.setItem(3, 1, QTableWidgetItem("Cell (4,2)"))
        self.table_ui.move(0, 20)
