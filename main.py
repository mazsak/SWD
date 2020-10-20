import sys

from PyQt5.QtWidgets import QApplication

from SWDMain import SWDMain

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SWDMain()
    sys.exit(app.exec_())
