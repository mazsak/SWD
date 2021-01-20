import pandas as pd
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QWidget, QDialogButtonBox, QLineEdit


class DecisionTree(QDialog):

    def __init__(self, df, *args, **kwargs):
        super(DecisionTree, self).__init__(*args, **kwargs)

        self.df = df

        self.setWindowTitle("Decision Tree")
        self.layout = QVBoxLayout()

        self.__choose_class(df.columns.tolist())
        self.__choose_bins()
        self.__init_buttons()

        self.setLayout(self.layout)

    def __choose_class(self, name_columns):
        combo_layout = QHBoxLayout()
        label = QLabel('Choose class: ', self)
        if name_columns:
            self.class_name = name_columns[0]
        combo = QComboBox(self)
        combo.addItems(name_columns)
        combo.activated[str].connect(self.__choose_class_action)
        combo_layout.addWidget(label)
        combo_layout.addWidget(combo)
        widget = QWidget()
        widget.setLayout(combo_layout)
        self.layout.addWidget(widget)

    def __choose_bins(self):
        integer_layout = QHBoxLayout()
        label = QLabel('Enter amount of bins: ', self)
        self.bins_field = QLineEdit(self)
        self.bins_field.setValidator(QIntValidator())
        integer_layout.addWidget(label)
        integer_layout.addWidget(self.bins_field)
        widget = QWidget()
        widget.setLayout(integer_layout)
        self.layout.addWidget(widget)

    def __choose_class_action(self, name_class):
        self.class_name = name_class

    def __init_buttons(self):
        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        button_box = QDialogButtonBox(buttons)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self.layout.addWidget(button_box)

    def calculate(self):
        bins = int(self.bins_field.text())
        for column in self.df.select_dtypes(['float']).columns:
            if column != self.class_name:
                self.df[f'{column}_binned_{bins}'] = pd.cut(self.table[column], bins=bins)
        print("tu dalej napisz cos")
