from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QListWidget, QLabel, QFileDialog
from PyQt5.uic.properties import QtGui, QtWidgets


class Export(QDialog):

    def __init__(self, df, *args, **kwargs):
        super(Export, self).__init__(*args, **kwargs)

        self.df = df
        self.selected_columns = []

        self.setWindowTitle("Export")
        self.layout = QVBoxLayout()

        self.__init_list()
        self.__init_buttons()

        self.setLayout(self.layout)

    def __init_list(self):
        self.selected = QLabel('Export columns: ')
        self.listwidget = QListWidget()
        self.listwidget.setSelectionMode(2)
        self.listwidget.insertItems(0, self.df.columns.tolist())

        self.listwidget.clicked.connect(self.__add_columns)
        self.layout.addWidget(self.selected)
        self.layout.addWidget(self.listwidget)

    def __add_columns(self):
        self.selected_columns = [e.text() for e in self.listwidget.selectedItems()]
        self.selected.setText(f'Export columns: {", ".join(self.selected_columns)}')

    def __init_buttons(self):
        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        button_box = QDialogButtonBox(buttons)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self.layout.addWidget(button_box)

    def export_to_file(self):
        if self.selected_columns:
            dlg = QFileDialog()
            dlg.setFileMode(QFileDialog.AnyFile)

            if dlg.exec_():
                filenames = dlg.selectedFiles()
                self.df[self.selected_columns].to_csv(filenames[0], index=False)