from typing import List, Dict

import pandas as pd
from PyQt5.QtWidgets import QMainWindow, QAction, QFileDialog, QMenuBar, QInputDialog, QTableWidget, QTableWidgetItem


def check_float(potential_float):
    try:
        float(potential_float)
        return True
    except ValueError:
        return False


class SWDMain(QMainWindow):

    def __init__(self) -> None:
        super().__init__()

        self.file_name: str = ''
        self.table: pd.DataFrame = pd.DataFrame()
        self.table_ui: QTableWidget = QTableWidget()

        self.init_ui()

    def init_ui(self) -> None:
        self.default_settings()

        self.init_menu_bar()
        self.init_layout()

        self.show()

    def init_layout(self):
        self.layout().addWidget(self.table_ui)
        self.table_ui.move(0, 20)

    def default_settings(self):
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('SWD')

    def init_menu_bar(self) -> None:
        self.init_file()
        self.init_update()

    def init_file(self) -> None:
        load_file: QAction = QAction('Open file', self)
        load_file.setShortcut('Ctrl+O')
        load_file.setStatusTip('Loading file')
        load_file.triggered.connect(self.load_file_action)

        self.statusBar()

        menu_bar: QMenuBar = self.menuBar()
        file_menu = menu_bar.addMenu('&File')
        file_menu.addAction(load_file)

    def load_file_action(self):
        file_name, _ = QFileDialog.getOpenFileName()
        if file_name:
            self.file_name: str = file_name
            if self.file_name.endswith('.csv'):
                self.table = pd.read_csv(self.file_name)
            elif self.file_name.endswith('.xls'):
                self.table = pd.read_excel(self.file_name)
            else:
                separator, ok = QInputDialog.getText(self, 'Choose separator',
                                                     'Enter your separator(default: ","):')
                if ok:
                    if not separator:
                        separator = ','
                    elif "\\t" == separator:
                        separator = '\t'
                    header: List[str] = []
                    with open(self.file_name, 'r') as f:
                        for line in f.readlines():
                            if line.startswith('#') or line == '\n':
                                continue
                            else:
                                data: List[str] = line.replace("\n", '').split(separator)
                                if len(self.table.columns) == 0:
                                    header = data
                                    self.table = pd.DataFrame(columns=data)
                                else:
                                    dict_data: Dict[str] = {}
                                    for index, name in enumerate(header):
                                        if check_float(data[index]) or check_float(data[index].replace(',', '.')):
                                            dict_data[name] = float(data[index].replace(',', '.'))
                                        elif data[index].isdigit():
                                            dict_data[name] = int(data[index])
                                        else:
                                            dict_data[name] = data[index]
                                    self.table = self.table.append(dict_data, ignore_index=True)
            self.create_table_ui()

    def create_table_ui(self):
        self.table_ui.setRowCount(len(self.table.index))
        self.table_ui.setColumnCount(len(self.table.columns))
        self.table_ui.setHorizontalHeaderLabels(self.table.columns)
        for index, column in enumerate(self.table.columns):
            for row in self.table.index:
                self.table_ui.setItem(row, index, QTableWidgetItem(str(self.table.at[row, column])))
        self.table_ui.move(0, 20)

    def init_update(self):
        change_text_to_digit: QAction = QAction('Change text to digit', self)
        change_text_to_digit.setShortcut('Ctrl+U')
        change_text_to_digit.setStatusTip('Change data')
        change_text_to_digit.triggered.connect(self.change_text_to_digit_action)

        self.statusBar()

        menu_bar: QMenuBar = self.menuBar()
        update_menu = menu_bar.addMenu('&Update')
        update_menu.addAction(change_text_to_digit)

    def change_text_to_digit_action(self):
        column, ok = QInputDialog.getItem(self, 'Update', "Choose column",
                             [f'{column}, {type}' for column in self.table.select_dtypes(['object', 'string']).columns
                              for type in ['alphabetically', 'order of appearance']])
        encoder = 0
        if column.endswith('order of appearance'):
            encoder = 1
        if ok:
            self.label_encoding(column.split(',')[0],encoder)
            self.create_table_ui()


    def label_encoding(self, column_name, encoding_type):
        if encoding_type % 2 == 0:
            # alphabet
            values: List[str] = list(sorted(set(self.table[column_name].values)))
            encoded_values = {x: i for i, x in enumerate(values)}
            self.table[column_name] = self.table[column_name].map(encoded_values)
        else:
            # order
            values: List[str] = list(set(self.table[column_name].values))
            encoded_values = {x: i for i, x in enumerate(values)}
            self.table[column_name] = self.table[column_name].map(encoded_values)
