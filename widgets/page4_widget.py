from PyQt5.QtWidgets import QVBoxLayout, QWidget

from TraditionalML_pyqt import TraditionalMLWindow


class Page4Widget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.page = TraditionalMLWindow()
        layout.addWidget(self.page)
