from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QMessageBox, QWidget, QHBoxLayout, QButtonGroup, QRadioButton
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt


class LoginWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("用户登录")
        self.resize(1400, 900)
        self.setStyleSheet(self._get_stylesheet())
        self.role = "用户"  # 默认身份为用户

        # 字体设置
        title_font = QFont("Arial", 36, QFont.Bold)
        input_font = QFont("Arial", 24)
        button_font = QFont("Arial", 18, QFont.Bold)

        # 标题
        self.title_label = QLabel("家庭经济困难学生认定系统")
        self.title_label.setFont(title_font)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("color: #2c4b7c; margin-bottom: 30px;")

        # 用户名输入框
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("请输入用户名")
        self.username_input.setFont(input_font)
        self.username_input.setFixedHeight(70)
        self.username_input.setFixedWidth(600)

        # 密码输入框
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setFont(input_font)
        self.password_input.setFixedHeight(70)
        self.password_input.setFixedWidth(600)

        # 身份选择按钮
        self.user_radio = QRadioButton("用户")
        self.user_radio.setFont(button_font)
        self.user_radio.setChecked(True)

        self.admin_radio = QRadioButton("管理员")
        self.admin_radio.setFont(button_font)

        self.role_group = QButtonGroup()
        self.role_group.addButton(self.user_radio)
        self.role_group.addButton(self.admin_radio)

        role_layout = QHBoxLayout()
        role_layout.setSpacing(50)
        role_layout.addStretch()
        role_layout.addWidget(self.user_radio)
        role_layout.addWidget(self.admin_radio)
        role_layout.addStretch()

        # 登录按钮
        self.login_button = QPushButton("登 录")
        self.login_button.setFont(button_font)
        self.login_button.setFixedHeight(70)
        self.login_button.setFixedWidth(300)
        self.login_button.clicked.connect(self.authenticate)

        # 布局设置
        form_layout = QVBoxLayout()
        form_layout.setAlignment(Qt.AlignCenter)
        form_layout.setSpacing(20)
        form_layout.addWidget(self.title_label)
        form_layout.addWidget(self.username_input, alignment=Qt.AlignCenter)
        form_layout.addWidget(self.password_input, alignment=Qt.AlignCenter)
        form_layout.addLayout(role_layout)
        form_layout.addWidget(self.login_button, alignment=Qt.AlignCenter)

        container = QWidget()
        container.setLayout(form_layout)

        outer_layout = QVBoxLayout(self)
        outer_layout.addStretch()
        outer_layout.addWidget(container, alignment=Qt.AlignCenter)
        outer_layout.addStretch()

    def authenticate(self):
        username = self.username_input.text()
        password = self.password_input.text()
        selected_role = "user" if self.user_radio.isChecked() else "admin"

        try:
            with open("users.txt", "r", encoding="utf-8") as f:
                for line in f:
                    u, p, r = line.strip().split("|")
                    if u == username and p == password and r == selected_role:
                        self.role = r
                        self.accept()
                        return
        except FileNotFoundError:
            pass

        QMessageBox.warning(self, "登录失败", "用户名、密码或身份错误")

    def _get_stylesheet(self):
        return """
        QDialog {
            background-color: #f0f4f8;
        }

        QLineEdit {
            border: 2px solid #cbd5e0;
            border-radius: 12px;
            padding: 10px 15px;
            font-size: 14px;
            background-color: #ffffff;
        }

        QLineEdit:focus {
            border: 2px solid #3f72af;
        }

        QPushButton {
            background-color: #3f72af;
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 16px;
        }

        QPushButton:hover {
            background-color: #2c4b7c;
        }

        QRadioButton {
            font-size: 18px;
        }
        """
