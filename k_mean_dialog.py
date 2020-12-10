import pandas as pd
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QComboBox, QHBoxLayout, QLabel, QWidget, QLineEdit

import distance_function


class KMeanDialog(QDialog):

    def __init__(self, name_columns, df, *args, **kwargs):
        super(KMeanDialog, self).__init__(*args, **kwargs)

        self.setWindowTitle("K mean")
        self.layout = QVBoxLayout()
        self.df = df
        self.__choose_class(name_columns)
        self.__choose_metric()
        self.__choose_k()
        self.__init_buttons()

        self.setLayout(self.layout)

    def __choose_class(self, name_columns):
        combo_layout = QHBoxLayout()
        label = QLabel('Choose class: ', self)
        if name_columns:
            self.class_name = name_columns[0]
        self.metric = 'Euclidean'
        combo = QComboBox(self)
        combo.addItems(name_columns)
        combo.activated[str].connect(self.__choose_class_action)
        combo_layout.addWidget(label)
        combo_layout.addWidget(combo)
        widget = QWidget()
        widget.setLayout(combo_layout)
        self.layout.addWidget(widget)

    def __choose_metric(self):
        combo_layout = QHBoxLayout()
        label = QLabel('Choose metric: ', self)
        combo = QComboBox(self)
        combo.addItems(['Euclidean', 'Manhattan', 'Chebyshev', 'Mahalanobis'])
        combo.activated[str].connect(self.__choose_metric_action)
        combo_layout.addWidget(label)
        combo_layout.addWidget(combo)
        widget = QWidget()
        widget.setLayout(combo_layout)
        self.layout.addWidget(widget)

    def __choose_class_action(self, name_class):
        self.class_name = name_class
        self.k_field.setText(str(len(pd.unique(self.df[self.class_name]))))

    def __choose_metric_action(self, metric):
        self.metric = metric

    def __choose_k(self):
        integer_layout = QHBoxLayout()
        label = QLabel('Enter k: ', self)
        self.k_field = QLineEdit(self)
        self.k_field.setValidator(QIntValidator())
        self.k_field.setText(str(len(pd.unique(self.df[self.df.columns.tolist()[0]]))))
        integer_layout.addWidget(label)
        integer_layout.addWidget(self.k_field)
        widget = QWidget()
        widget.setLayout(integer_layout)
        self.layout.addWidget(widget)

    def __init_buttons(self):
        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        button_box = QDialogButtonBox(buttons)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self.layout.addWidget(button_box)

    def k_mean(self):
        k = int(self.k_field.text())
        centroid = self.df.sample(n=k)
        while True:
            centroid_temp = centroid.copy()
            distance = []
            classes = [[] for i in range(k)]
            for index, c in enumerate(centroid.index):
                row = {column: centroid.iloc[index][column] for column in self.df.columns.tolist() if column != self.class_name}
                if self.metric == 'Euclidean':
                    distance.append(distance_function.distance_euclidean(self.class_name, self.df, row))
                elif self.metric == 'Manhattan':
                    distance.append(distance_function.distance_manhattan(self.class_name, self.df, row))
                elif self.metric == 'Chebyshev':
                    distance.append(distance_function.distance_chebyshev(self.class_name, self.df, row))
                elif self.metric == 'Mahalanobis':
                    distance.append(distance_function.distance_mahalanobis(self.class_name, self.df, row))

            for index, d in enumerate(distance[0]):
                element = [{'distance': di[index]['distance'], 'index': i} for i, di in enumerate(distance)]
                element.sort(key=lambda e: e['distance'])
                classes[element[0]['index']].append(index)

            df_groups = [self.df.iloc[c] for c in classes]
            centroid = []
            for index, g in enumerate(df_groups):
                centroid.append({
                    column: df_groups[index][column].mean().round(2)
                    if column != self.class_name
                    else ' '
                    for column in self.df.columns.tolist()})
            centroid = pd.DataFrame(data=centroid)

            if centroid_temp.drop([self.class_name], axis=1).equals(centroid.drop([self.class_name], axis=1)):
                response = []
                for index in self.df.index.tolist():
                    for group, c in enumerate(classes):
                        if index in c:
                            response.append(group + 1)
                return response
