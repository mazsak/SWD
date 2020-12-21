from PyQt5.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout, QDialogButtonBox, QLabel, QComboBox, QWidget


class Similarity(QDialog):

    def __init__(self, df, *args, **kwargs):
        super(Similarity, self).__init__(*args, **kwargs)

        self.method = 'Jaccard'
        self.first_column = df.columns.tolist()[0]
        self.second_column = df.columns.tolist()[1]
        self.df = df

        self.setWindowTitle("Similarity")
        self.layout = QVBoxLayout()

        self.__choose_method()
        self.__choose_first_column()
        self.__choose_second_column()
        self.__init_buttons()

        self.setLayout(self.layout)

    def __choose_method(self):
        combo_layout = QHBoxLayout()
        label = QLabel('Choose method: ', self)
        combo = QComboBox(self)
        combo.addItem('Jaccard')
        combo.addItem('Simple matching coefficient')
        combo.activated[str].connect(self.__choose_method_action)
        combo_layout.addWidget(label)
        combo_layout.addWidget(combo)
        widget = QWidget()
        widget.setLayout(combo_layout)
        self.layout.addWidget(widget)

    def __choose_first_column(self):
        combo_layout = QHBoxLayout()
        label = QLabel('Choose first column: ', self)
        combo = QComboBox(self)
        combo.addItems(self.df.columns.tolist())
        combo.activated[str].connect(self.__choose_first_column_action)
        combo_layout.addWidget(label)
        combo_layout.addWidget(combo)
        widget = QWidget()
        widget.setLayout(combo_layout)
        self.layout.addWidget(widget)

    def __choose_second_column(self):
        combo_layout = QHBoxLayout()
        label = QLabel('Choose second column: ', self)
        combo = QComboBox(self)
        combo.addItems(self.df.columns.tolist())
        combo.activated[str].connect(self.__choose_second_column_action)
        combo_layout.addWidget(label)
        combo_layout.addWidget(combo)
        widget = QWidget()
        widget.setLayout(combo_layout)
        self.layout.addWidget(widget)

    def __choose_first_column_action(self, column):
        self.first_column = column

    def __choose_second_column_action(self, column):
        self.second_column = column

    def __choose_method_action(self, method):
        self.method = method

    def __init_buttons(self):
        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        button_box = QDialogButtonBox(buttons)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self.layout.addWidget(button_box)

    def execute(self):
        if self.method == 'Jaccard':
            return self.__jaccard()
        elif self.method == 'Simple matching coefficient':
            return self.__simple_matching_coefficient()

    def __jaccard(self):
        list1 = list(self.df[self.first_column])
        list2 = list(self.df[self.second_column])
        intersection = 0
        for i, item in enumerate(list1):
            if list1[i] == list2[i]:
                intersection += 1
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union

    def __simple_matching_coefficient(self):
        list1 = list(self.df[self.first_column])
        list2 = list(self.df[self.second_column])
        similarity = 0
        for i, item in enumerate(list1):
            if list1[i] == list2[i]:
                similarity += 1
        return similarity / len(list1)
