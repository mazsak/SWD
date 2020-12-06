import math
from collections import Counter
from typing import List, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QMainWindow, QAction, QFileDialog, QMenuBar, QInputDialog, QTableWidget, QTableWidgetItem, \
    QMessageBox, QProgressBar
from numpy.linalg import LinAlgError

from vector_binary_dialog import VectorBinary

matplotlib.use('Qt5Agg')


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
        self.basic_table: pd.DataFrame = pd.DataFrame()
        self.table_ui: QTableWidget = QTableWidget()
        self.figure_counter = 0

        self.init_ui()

    def init_ui(self) -> None:
        self.default_settings()

        self.init_menu_bar()
        self.init_layout()

        self.show()

    def init_layout(self):
        self.layout().addWidget(self.table_ui)
        self.table_ui.resizeColumnsToContents()
        self.table_ui.move(0, 20)
        self.table_ui.setMinimumSize(1200, 800)

    def default_settings(self):
        self.setMinimumSize(1200, 820)
        self.setWindowTitle('SWD')

    def init_menu_bar(self) -> None:
        self.init_file()
        self.init_update()
        self.init_action()
        self.init_plots()
        self.init_classifier()

    def init_file(self) -> None:
        load_file: QAction = QAction('Open file', self)
        load_file.setShortcut('Ctrl+O')
        load_file.setStatusTip('Loading file')
        load_file.triggered.connect(self.load_file_action)

        reload_file: QAction = QAction('Reload file', self)
        reload_file.setShortcut('Ctrl+R')
        reload_file.setStatusTip('Reloading file')
        reload_file.triggered.connect(self.reload_file_action)

        self.statusBar()

        menu_bar: QMenuBar = self.menuBar()
        file_menu = menu_bar.addMenu('&File')
        file_menu.addAction(load_file)
        file_menu.addAction(reload_file)

    def reload_file_action(self):
        self.table = self.basic_table.copy()
        self.create_table_ui()

    def load_file_action(self):
        file_name, _ = QFileDialog.getOpenFileName()
        if file_name:
            self.table = pd.DataFrame()
            self.file_name: str = file_name
            if self.file_name.endswith('.csv'):
                self.table = pd.read_csv(self.file_name)
            elif self.file_name.endswith('.xls') or self.file_name.endswith('.xlsx'):
                self.table = pd.read_excel(self.file_name)
            else:
                separator, ok = QInputDialog.getText(self, 'Choose separator',
                                                     'Enter your separator(default: ","):')
                if ok:
                    headers = QMessageBox.question(self, 'Header', 'Does the first line have headers?',
                                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
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
                                if len(self.table.columns) == 0 and headers == QMessageBox.Yes:
                                    header = data
                                    self.table = pd.DataFrame(columns=data)
                                else:
                                    if not header:
                                        header = [str(x) for x in range(len(data))]
                                        self.table = pd.DataFrame(columns=header)
                                    dict_data: Dict[str] = {}
                                    for index, name in enumerate(header):
                                        if check_float(data[index]) or check_float(data[index].replace(',', '.')):
                                            dict_data[name] = float(data[index].replace(',', '.'))
                                        elif data[index].isdigit():
                                            dict_data[name] = int(data[index])
                                        else:
                                            dict_data[name] = data[index]
                                    self.table = self.table.append(dict_data, ignore_index=True)
            self.basic_table = self.table.copy()
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
                                          [f'{column}, {type}' for column in
                                           self.table.select_dtypes(['object', 'string']).columns
                                           for type in ['alphabetically', 'order of appearance']])
        encoder = 0
        if column.endswith('order of appearance'):
            encoder = 1
        if ok:
            self.label_encoding(column.split(',')[0], encoder)
            self.create_table_ui()

    def label_encoding(self, column_name, encoding_type):
        if encoding_type % 2 == 0:
            # alphabet
            values: List[str] = list(sorted(set(self.table[column_name].values)))
            encoded_values = {x: i for i, x in enumerate(values)}
            self.table[f'{column_name}_encoded'] = self.table[column_name].map(encoded_values)
        else:
            # order
            values: List[str] = list(set(self.table[column_name].values))
            encoded_values = {x: i for i, x in enumerate(values)}
            self.table[f'{column_name}_encoded'] = self.table[column_name].map(encoded_values)

    def init_action(self):
        binning: QAction = QAction('Binning', self)
        binning.setShortcut('Ctrl+B')
        binning.setStatusTip('Binning data')
        binning.triggered.connect(self.binning_action)

        normalization: QAction = QAction('Normalization', self)
        normalization.setShortcut('Ctrl+N')
        normalization.setStatusTip('Normalize data')
        normalization.triggered.connect(self.normalization_action)

        interpolate: QAction = QAction('Interpolate', self)
        interpolate.setShortcut('Ctrl+I')
        interpolate.setStatusTip('Interpolate data')
        interpolate.triggered.connect(self.interpolate_action)

        selected_min_max: QAction = QAction('Selected MIN/MAX', self)
        selected_min_max.setShortcut('Ctrl+M')
        selected_min_max.setStatusTip('Selected data')
        selected_min_max.triggered.connect(self.selected_min_max_action)

        self.statusBar()

        menu_bar: QMenuBar = self.menuBar()
        action = menu_bar.addMenu('&Action')
        action.addAction(binning)
        action.addAction(normalization)
        action.addAction(interpolate)
        action.addAction(selected_min_max)

    def binning_action(self):
        column, ok = QInputDialog.getItem(self, 'Binning', "Choose column",
                                          [f'{column}' for column in
                                           self.table.select_dtypes(['float']).columns])
        if ok:
            bins_amount, ok_amount = QInputDialog.getInt(self, 'Binning', "Enter amount of bins")
            if ok_amount:
                self.table[f'{column}_binned_{bins_amount}'] = pd.cut(self.table[column], bins=bins_amount)
        self.create_table_ui()

    def normalization_action(self):
        column, ok = QInputDialog.getItem(self, 'Binning', "Choose column",
                                          [f'{column}' for column in
                                           self.table.select_dtypes(['float']).columns])
        if ok:
            self.table[f'{column}_normalization'] = (
                    (self.table[column] - self.table[column].mean()) / self.table[column].std()).round(5)
        self.create_table_ui()

    def interpolate_action(self):
        column, ok = QInputDialog.getItem(self, 'Interpolate', "Choose column",
                                          [f'{column}' for column in
                                           self.table.select_dtypes(['float']).columns])
        if ok:
            a, ok_a = QInputDialog.getInt(self, 'Left range', "Enter beginning of range")
            if ok_a:
                b, ok_b = QInputDialog.getInt(self, 'Right range', "Enter end of range")
                if ok_b:
                    self.table[f'{column}_interpolated <{a},{b}>'] = self.table[column].apply(
                        lambda x: np.interp(x, [self.table[column].min(), self.table[column].max()], [a, b])).round(5)
        self.create_table_ui()

    def selected_min_max_action(self):
        column, ok = QInputDialog.getItem(self, 'Selected MIN/MAX', "Choose column",
                                          [f'{column}' for column in
                                           self.table.select_dtypes(['float', 'int']).columns])

        if ok:
            percent, ok_a = QInputDialog.getInt(self, 'Selected MIN/MAX', "Enter percentage:")
            if ok_a:
                df = self.table.copy().sort_values(column)
                df_min = df[df[column] < np.percentile(df[column], percent)].reset_index().drop(columns=['index'])
                df_max = df[df[column] > np.percentile(df[column], 100 - percent)].reset_index().drop(columns=['index'])
                self.min_table: QTableWidget = QTableWidget()
                self.min_table.move(0, 0)
                self.min_table.setRowCount(len(df_min.index))
                self.min_table.setColumnCount(len(df_min.columns))
                self.min_table.setHorizontalHeaderLabels(df_min.columns)

                self.max_table: QTableWidget = QTableWidget()
                self.max_table.move(0, 0)
                self.max_table.setRowCount(len(df_max.index))
                self.max_table.setColumnCount(len(df_max.columns))
                self.max_table.setHorizontalHeaderLabels(df_max.columns)
                for index, column in enumerate(df_min.columns):
                    for row in df_min.index:
                        self.min_table.setItem(row, index, QTableWidgetItem(str(df_min.at[row, column])))
                self.min_table.move(0, 0)
                self.min_table.setWindowTitle(f'Min table({percent}%)')
                self.min_table.show()

                for index, column in enumerate(df_max.columns):
                    for row in df_max.index:
                        self.max_table.setItem(row, index, QTableWidgetItem(str(df_max.at[row, column])))
                self.max_table.move(0, 0)
                self.max_table.setWindowTitle(f'Max table({percent}%)')
                self.max_table.show()

    def init_plots(self) -> None:
        plot_2D: QAction = QAction('Plot 2D', self)
        plot_2D.setShortcut('Ctrl+2')
        plot_2D.setStatusTip('Create plot 2D')
        plot_2D.triggered.connect(self.plot_2D_action)

        plot_3D: QAction = QAction('Plot 3D', self)
        plot_3D.setShortcut('Ctrl+3')
        plot_3D.setStatusTip('Create plot 3D')
        plot_3D.triggered.connect(self.plot_3D_action)

        histogram: QAction = QAction('Histogram', self)
        histogram.setShortcut('Ctrl+H')
        histogram.setStatusTip('Create histogram')
        histogram.triggered.connect(self.histogram_action)

        self.statusBar()

        menu_bar: QMenuBar = self.menuBar()
        plots_menu = menu_bar.addMenu('&Plots')
        plots_menu.addAction(plot_2D)
        plots_menu.addAction(plot_3D)
        plots_menu.addAction(histogram)

    def plot_2D_action(self):
        x, ok = QInputDialog.getItem(self, '2D Plot', "Choose column for x values",
                                     [f'{column}' for column in
                                      self.table.columns])
        if ok:
            y, ok_2 = QInputDialog.getItem(self, '2D Plot', "Choose column for y values",
                                           [f'{column}' for column in
                                            self.table.columns])
            if ok_2:
                grouping_column, ok_3 = QInputDialog.getItem(self, '2D Plot', "Choose column to group by",
                                                             [f'{column}' for column in
                                                              self.table.select_dtypes(['object', 'string']).columns])
                if ok_3:
                    temp_data = self.table.groupby(grouping_column)
                    for name, group in temp_data:
                        plt.scatter(group[x], group[y], label=name)
                    plt.xlabel(x)
                    plt.ylabel(y)
                    plt.legend()
                    plt.title(f'2D Scatter plot, x={x}, y={y}, grouped by {grouping_column}')
                    plt.show()
                else:
                    plt.scatter(self.table[x], self.table[y])
                    plt.xlabel(x)
                    plt.ylabel(y)
                    plt.title(f'2D Scatter plot, x={x}, y={y}')
                    plt.show()

    def plot_3D_action(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, ok = QInputDialog.getItem(self, '3D Plot', "Choose column for x values",
                                     [f'{column}' for column in
                                      self.table.columns])
        if ok:
            y, ok_2 = QInputDialog.getItem(self, '3D Plot', "Choose column for y values",
                                           [f'{column}' for column in
                                            self.table.columns])
            if ok_2:
                z, ok_3 = QInputDialog.getItem(self, '3D Plot', "Choose column for z values",
                                               [f'{column}' for column in
                                                self.table.columns])
                if ok_3:
                    grouping_column, ok_4 = QInputDialog.getItem(self, '2D Plot', "Choose column to group by",
                                                                 [f'{column}' for column in
                                                                  self.table.select_dtypes(
                                                                      ['object', 'string']).columns])
                    if ok_4:
                        temp_data = self.table.groupby(grouping_column)
                        for name, group in temp_data:
                            ax.scatter(group[x], group[y], group[z], label=name)
                        ax.set_xlabel(x)
                        ax.set_ylabel(y)
                        ax.set_zlabel(z)
                        plt.legend()
                        plt.title(f'3D Scatter plot, x={x}, y={y}, z={z}, grouped by {grouping_column}')
                        plt.show()
                    else:
                        ax.scatter(self.table[x], self.table[y], self.table[z])
                        ax.set_xlabel(x)
                        ax.set_ylabel(y)
                        ax.set_zlabel(z)
                        plt.title(f'3D Scatter plot, x={x}, y={y}, z={z}')
                        plt.show()

    def histogram_action(self):
        fig, ax = plt.subplots()
        x, ok = QInputDialog.getItem(self, 'Histogram', "Choose column for x values",
                                     [f'{column}' for column in
                                      self.table.columns])
        if ok:
            amount, ok_1 = QInputDialog.getInt(self, 'Histogram', "Enter amount of bins")
            if ok_1:

                if x not in self.table.select_dtypes(['object', 'string', 'category']).columns:
                    counts, bins, patches = ax.hist(self.table[x], edgecolor='gray', bins=amount)
                    ax.set_xticks(bins)
                else:
                    counts, bins, patches = ax.hist(self.table[x], edgecolor='gray')
                plt.xticks(rotation='vertical')
                plt.show()

    def init_classifier(self):
        knn: QAction = QAction('KNN', self)
        knn.setShortcut('Ctrl+K')
        knn.setStatusTip('Exercise KNN')
        knn.triggered.connect(self.knn_action)

        knn_leave_one_out: QAction = QAction('KNN leave-one-out', self)
        knn_leave_one_out.setShortcut('Ctrl+L')
        knn_leave_one_out.setStatusTip('Exercise KNN leave-one-out')
        knn_leave_one_out.triggered.connect(self.knn_leave_one_out_action)

        knn_leave_one_out_all: QAction = QAction('KNN leave-one-out for all k and metrics', self)
        knn_leave_one_out_all.setShortcut('Ctrl+A')
        knn_leave_one_out_all.setStatusTip('Exercise KNN leave-one-out for all k and metrics')
        knn_leave_one_out_all.triggered.connect(self.knn_leave_one_out_all_action)

        calculate: QAction = QAction('Calculate binary vector', self)
        calculate.setShortcut('Ctrl+C')
        calculate.setStatusTip('Calculate')
        calculate.triggered.connect(self.calculate_binary_vector_action)

        self.statusBar()

        menu_bar: QMenuBar = self.menuBar()
        classifier_menu = menu_bar.addMenu('&Classifier')
        classifier_menu.addAction(knn)
        classifier_menu.addAction(knn_leave_one_out)
        classifier_menu.addAction(knn_leave_one_out_all)
        classifier_menu.addAction(calculate)

    def knn_action(self):
        class_name, ok = QInputDialog.getItem(self, 'Classifier KNN', "Choose class column",
                                              [f'{column}' for column in
                                               self.table.columns])
        if ok:
            k, ok_k = QInputDialog.getInt(self, 'Classifier KNN', "Enter k")
            if ok_k:
                metric, ok_metric = QInputDialog.getItem(self, 'Classifier KNN', "Choose metric",
                                                         ['Euclidean', 'Manhattan', 'Chebyshev', 'Mahalanobis'])

                go = True
                row = {}
                for name in self.table.columns.tolist():
                    if name != class_name:
                        if self.table[name].dtypes in ['float', 'int']:
                            data, ok_data = QInputDialog.getDouble(self, 'Data', f'Enter {name} column')
                            if not ok_data:
                                go = False
                                break
                            else:
                                row[name] = data
                        else:
                            data, ok_data = QInputDialog.getText(self, 'Data', f'Enter {name} column')
                            if not ok_data:
                                go = False
                                break
                            else:
                                row[name] = data

                if go:
                    if ok_metric:
                        y = self.__calculate_distance(k, metric, class_name, row, self.table)
                        QMessageBox.information(self, 'KNN Classification',
                                                f'The item has been classified using {metric} metric as {y} for parameters k={k}, class={class_name}, row_info={row}',
                                                QMessageBox.Ok)

    def __calculate_distance(self, k, metric, class_name, new_row, data):
        for column in data.select_dtypes(['object', 'string']).columns:
            if column != class_name:
                values: List[str] = list(set(data[column].values))
                encoded_values = {x: i for i, x in enumerate(values)}
                data[column] = data[column].map(encoded_values)

        if metric == 'Euclidean':
            distance = [{'distance': math.sqrt(
                sum([math.pow(row[column_name] - new_row[column_name], 2) for column_name in data.columns.tolist()
                     if
                     column_name != class_name])), 'y': row[class_name]} for
                row in data.iloc]
        elif metric == 'Manhattan':
            distance = [{'distance':
                             sum([abs(row[column_name] - new_row[column_name]) for column_name in
                                  data.columns.tolist()
                                  if
                                  column_name != class_name]), 'y': row[class_name]} for
                        row in data.iloc]
        elif metric == 'Chebyshev':
            distance = [{'distance': max(
                [abs(row[column_name] - new_row[column_name]) for column_name in data.columns.tolist()
                 if
                 column_name != class_name]), 'y': row[class_name]} for
                row in data.iloc]
        elif metric == 'Mahalanobis':
            columns = data.columns.tolist()
            columns.remove(class_name)
            cov = data[columns].cov().to_numpy()
            inv_covmat = np.linalg.inv(cov)
            distance = [{'distance': np.subtract(np.array(list(new_row.values())), np.array(
                [row[name_column] for name_column in data.columns.tolist() if name_column != class_name])).T.dot(
                inv_covmat).dot(np.subtract(np.array(list(new_row.values())), np.array(
                [row[name_column] for name_column in data.columns.tolist() if name_column != class_name]))),
                         'y': row[class_name]} for row in data.iloc]
        distance.sort(key=lambda x: x['distance'])
        closest = distance[0: min(k, len(distance))]
        counter = Counter([x['y'] for x in closest])
        y = counter.most_common(1)

        return y[0][0]

    def knn_leave_one_out_action(self):
        data = self.basic_table.copy()
        class_name, ok = QInputDialog.getItem(self, 'Classifier KNN leave-one-out', "Choose class column",
                                              [f'{column}' for column in
                                               data.columns])
        if ok:
            k, ok_k = QInputDialog.getInt(self, 'Classifier KNN leave-one-out', "Enter k")
            if ok_k:
                metric, ok_metric = QInputDialog.getItem(self, 'Classifier KNN leave-one-out', "Choose metric",
                                                         ['Euclidean', 'Manhattan', 'Chebyshev', 'Mahalanobis'])
                if ok_metric:
                    normalization = QMessageBox.question(self, 'Normalization', 'Should the dataset be normalized?',
                                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if normalization == QMessageBox.Yes:
                        for column in data.select_dtypes(['float', 'int']).columns:
                            data[column] = ((data[column] - data[column].mean()) / self.table[column].std()).round(5)
                    for column in data.select_dtypes(['object', 'string']).columns:
                        if column != class_name:
                            values: List[str] = list(set(data[column].values))
                            encoded_values = {x: i for i, x in enumerate(values)}
                            data[column] = data[column].map(encoded_values)
                    results = []
                    columns = data.columns.tolist()
                    columns.remove(class_name)
                    if metric == 'Mahalanobis':
                        cov = data[columns].cov().to_numpy()
                        inv_covmat = np.linalg.inv(cov)
                    amount = math.pow(len(data.index.tolist()), 2) + len(data.index.tolist())
                    all = amount
                    pbar = QProgressBar(self)
                    pbar.setGeometry(550, 385, 100, 30)
                    pbar.setMinimum(0)
                    pbar.setMaximum(all)
                    pbar.show()
                    for index, row in enumerate(data.iloc):
                        results_row = []
                        for index_second, column in enumerate(data.iloc):
                            if index_second <= index:
                                if metric == 'Euclidean':
                                    results_row.append({
                                        'distance': math.sqrt(
                                            sum([
                                                math.pow(row[column_name] - column[column_name], 2)
                                                for column_name in columns
                                            ])),
                                        'class_row': row[class_name],
                                        'class_column': column[class_name]
                                    })
                                elif metric == 'Manhattan':
                                    results_row.append({
                                        'distance': sum([
                                            abs(row[column_name] - column[column_name])
                                            for column_name in columns
                                        ]),
                                        'class_row': row[class_name],
                                        'class_column': column[class_name]
                                    })
                                elif metric == 'Chebyshev':
                                    results_row.append({
                                        'distance': max([
                                            abs(row[column_name] - column[column_name])
                                            for column_name in columns
                                        ]),
                                        'class_row': row[class_name],
                                        'class_column': column[class_name]
                                    })
                                elif metric == 'Mahalanobis':
                                    results_row.append({
                                        'distance': np.subtract(
                                            np.array([
                                                column[name_column]
                                                for name_column in columns
                                            ]),
                                            np.array([
                                                row[name_column]
                                                for name_column in columns
                                            ]))
                                            .T.dot(inv_covmat)
                                            .dot(
                                            np.subtract(
                                                np.array([
                                                    column[name_column]
                                                    for name_column in columns
                                                ]),
                                                np.array([
                                                    row[name_column]
                                                    for name_column in columns
                                                ]))),
                                        'class_row': row[class_name],
                                        'class_column': column[class_name]
                                    })
                                pbar.setValue((all - amount))
                                amount -= 1
                            else:
                                pbar.setValue((all - amount))
                                amount -= 1
                                continue
                        results.append(results_row)

                    correct = 0
                    for index, row in enumerate(results):
                        if len(row) == 1:
                            distance = []
                        else:
                            distance = row[:-1]
                        distance.extend([d[index] for i, d in enumerate(results) if i > index])
                        distance.sort(key=lambda x: x['distance'])
                        closest = distance[0: min(k, len(distance))]
                        counter = Counter([x['class_row'] for x in closest])
                        y = counter.most_common(1)

                        if y[0][0] == data.iloc[index][class_name]:
                            correct += 1
                        pbar.setValue((all - amount))
                        amount -= 1
                    pbar.hide()

                    QMessageBox.information(self, 'KNN Classification leave-one-out',
                                            f'Quality of KNN classification using {metric} metric with k={k}, class={class_name}: {correct}/{len(results)} - {round(correct / len(results) * 100, 2)}%',
                                            QMessageBox.Ok)

    def knn_leave_one_out_all_action(self):
        class_name, ok = QInputDialog.getItem(self, 'Classifier KNN leave-one-out all', "Choose class column",
                                              [f'{column}' for column in
                                               self.table.columns])
        if ok:
            data = self.basic_table.copy()
            normalization = QMessageBox.question(self, 'Normalization', 'Should the dataset be normalized?',
                                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if normalization == QMessageBox.Yes:
                for column in data.select_dtypes(['float', 'int']).columns:
                    data[column] = ((data[column] - data[column].mean()) / self.table[column].std()).round(5)
            for column in data.select_dtypes(['object', 'string']).columns:
                if column != class_name:
                    values: List[str] = list(set(data[column].values))
                    encoded_values = {x: i for i, x in enumerate(values)}
                    data[column] = data[column].map(encoded_values)
            results = []
            columns = data.columns.tolist()
            columns.remove(class_name)
            cov = data[columns].cov().to_numpy()
            check_inverse = True
            try:
                inv_covmat = np.linalg.inv(cov)
            except LinAlgError:
                check_inverse = False
            amount = math.pow(len(data.index.tolist()), 2) * 2
            all = amount
            pbar = QProgressBar(self)
            pbar.setGeometry(550, 385, 100, 30)
            pbar.setMinimum(0)
            pbar.setMaximum(all)
            pbar.show()
            for index, row in enumerate(data.iloc):
                if check_inverse:
                    results_row = {
                        'Euclidean': [],
                        'Manhattan': [],
                        'Chebyshev': [],
                        'Mahalanobis': []
                    }
                else:
                    results_row = {
                        'Euclidean': [],
                        'Manhattan': [],
                        'Chebyshev': []
                    }
                for index_second, column in enumerate(data.iloc):
                    if index_second <= index:
                        results_row['Euclidean'].append({
                            'distance': math.sqrt(
                                sum([
                                    math.pow(row[column_name] - column[column_name], 2)
                                    for column_name in columns
                                ])),
                            'class_row': row[class_name],
                            'class_column': column[class_name]
                        })
                        results_row['Manhattan'].append({
                            'distance': sum([
                                abs(row[column_name] - column[column_name])
                                for column_name in columns
                            ]),
                            'class_row': row[class_name],
                            'class_column': column[class_name]
                        })
                        results_row['Chebyshev'].append({
                            'distance': max([
                                abs(row[column_name] - column[column_name])
                                for column_name in columns
                            ]),
                            'class_row': row[class_name],
                            'class_column': column[class_name]
                        })
                        if check_inverse:
                            results_row['Mahalanobis'].append({
                                'distance': np.subtract(
                                    np.array([
                                        column[name_column]
                                        for name_column in columns
                                    ]),
                                    np.array([
                                        row[name_column]
                                        for name_column in columns
                                    ]))
                                    .T.dot(inv_covmat)
                                    .dot(
                                    np.subtract(
                                        np.array([
                                            column[name_column]
                                            for name_column in columns
                                        ]),
                                        np.array([
                                            row[name_column]
                                            for name_column in columns
                                        ]))),
                                'class_row': row[class_name],
                                'class_column': column[class_name]
                            })
                        pbar.setValue((all - amount))
                        amount -= 1
                    else:
                        pbar.setValue((all - amount))
                        amount -= 1
                        continue
                results.append(results_row)
            if check_inverse:
                correct = {
                    'Euclidean': {},
                    'Manhattan': {},
                    'Chebyshev': {},
                    'Mahalanobis': {}
                }
            else:
                correct = {
                    'Euclidean': {},
                    'Manhattan': {},
                    'Chebyshev': {}
                }
            for k in range(1, len(data.index.tolist())):
                for metric in correct.keys():
                    correct[metric][k] = 0
                for index, row in enumerate(results):
                    distance = {}
                    for metric in row.keys():
                        if len(row[metric]) == 1:
                            distance[metric] = []
                        else:
                            distance[metric] = row[metric][:-1]
                        distance[metric].extend([d[metric][index] for i, d in enumerate(results) if i > index])
                        distance[metric].sort(key=lambda x: x['distance'])
                        closest = distance[metric][0: min(k, len(distance[metric]))]
                        counter = Counter([x['class_row'] for x in closest])
                        y = counter.most_common(1)

                        if y[0][0] == data.iloc[index][class_name]:
                            correct[metric][k] += 1
                    pbar.setValue((all - amount))
                    amount -= 1
            pbar.hide()
            if not check_inverse:
                QMessageBox.information(self, 'Classifier KNN leave-one-out all',
                                        "Couldn't get an inverse of covariance matrix, skipping Mahalanobis",
                                        QMessageBox.Ok)
            df = pd.DataFrame(correct)
            plt.figure(self.figure_counter)
            self.figure_counter += 1
            plt.plot(df)
            plt.legend(df.columns)
            # plt.savefig()
            plt.show()

    def calculate_binary_vector_action(self):
        grouping_column = self.basic_table.columns[-1]

        df = self.basic_table.copy()
        lines_data = []
        lines = []
        amount = len(self.basic_table.index)
        all = amount
        pbar = QProgressBar(self)
        pbar.setGeometry(550, 385, 100, 30)
        pbar.setMinimum(0)
        pbar.setMaximum(all)
        pbar.show()
        counter = 0
        while not df.empty:
            directions = []
            for column in self.table.select_dtypes(['float', 'int']).columns.tolist():
                if column != grouping_column:
                    directions.append(
                        self.search_ranked(df.sort_values(by=[column], ascending=False), column, grouping_column,
                                           'END'))  # end
                    directions.append(
                        self.search_ranked(df.sort_values(by=[column]), column, grouping_column, 'START'))  # start
            directions.sort(key=lambda e: (e[2], e[1]), reverse=True)
            direction = directions[0]
            data = direction[0]
            if direction[4] == '0':
                counter += 1
                print('co')
            if counter == 7:
                print('stop')
            if direction[3] == 'END':
                value = direction[0][direction[4]].min()
                data = self.basic_table[self.basic_table[direction[4]] >= value]
                lines.append({'value': value, 'column': direction[4], 'direction': direction[3]})
            else:
                value = direction[0][direction[4]].max()
                data = self.basic_table[self.basic_table[direction[4]] <= value]
                lines.append({'value': value, 'column': direction[4], 'direction': direction[3]})
            lines_data.append(data)
            df = df.drop(direction[0].index)

            pbar.setValue((all - amount))
            amount -= len(direction[0].index)

            if len(df.index) == 1:
                lines.append({'value': value, 'column': direction[4],
                              'direction': 'END' if direction[3] == 'START' else 'START'})
                break

        lines_data = lines_data[:-1]
        vectors = {}
        for index in self.basic_table.index:
            for i, line in enumerate(lines_data):
                name_column = f'vector_{i + 1}'
                if name_column not in vectors:
                    vectors[name_column] = []
                if self.basic_table.index[index] in line.index.tolist():
                    vectors[name_column].append('1')
                else:
                    vectors[name_column].append('0')

        for key, vector in vectors.items():
            self.table[key] = vector
        pbar.hide()
        self.create_table_ui()

        if len(self.basic_table.columns) == 3:
            dialog = VectorBinary(parent=self, df=self.basic_table, lines=lines)
            if dialog.exec_():
                print(f'stop show plot {len(self.basic_table.columns)}')
            else:
                print(f'stop {len(self.basic_table.columns)}')
        else:
            print(f"stop {len(self.basic_table.columns)}")

    def search_ranked(self, df, column, grouping_column, direction):
        first_row = df.iloc[0]
        group = first_row[grouping_column]
        indexs = []
        df_group = pd.DataFrame()
        delete_element = 1
        for index, row in enumerate(df.index):
            if df.iloc[index][grouping_column] == group:
                indexs.append(index)
            else:
                for i in indexs:
                    if df.iloc[index][column] != df.iloc[i][column]:
                        df_group = df_group.append(df.iloc[i])
                if df_group.empty:
                    delete_element = 0
                    df_group = df.loc[df[column] == df.iloc[index][column]]
                    group = Counter(df_group[grouping_column]).most_common(1)[0][0]

                    indexs = []
                    for index, row in enumerate(df.index):
                        if df.index[index] not in df_group.index.tolist():
                            if df.iloc[index][grouping_column] == group:
                                indexs.append(index)
                            else:
                                for i in indexs:
                                    if df.iloc[index][column] != df.iloc[i][column]:
                                        df_group = df_group.append(df.iloc[i])
                                break
                break
        return df_group, len(df_group.index.tolist()), delete_element, direction, column
