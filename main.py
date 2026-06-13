import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog

from widgets.page1_widget import Page1Widget
from widgets.page2_widget import Page2Widget
from widgets.page3_widget import Page3Widget

from login_window import LoginWindow

class IntegratedMainWindow(QMainWindow):
    def __init__(self, role='user'):
        super().__init__()
        self.setWindowTitle("家庭经济困难学生精准认定工具")
        self.resize(1400, 900)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("MainTabs")
        self.setCentralWidget(self.tabs)

        self.tabs.addTab(Page1Widget(), "① 指标体系选择与数据录入")
        self.tabs.addTab(Page2Widget(), "② 算法运行")
        if role == 'admin':
            self.tabs.addTab(Page3Widget(), "③ 指标体系优化")

        self.tabs.tabBar().setElideMode(Qt.ElideNone)
        self.setStyleSheet(self._get_stylesheet())

    def _get_stylesheet(self):
    # 主窗口Tab样式+Page2三个按钮样式，按钮宽度稍宽，字体大小适中
        return """
            QMainWindow {
                background-color: #f0f4f8;
            }

            #MainTabs::pane {
                border: 2px solid #3f72af;
                margin: 5px;
                background-color: #ffffff;
            }

            #MainTabs QTabBar::tab {
                background: #3f72af;
                color: #e0e6f3;
                padding: 12px 24px; /* 增加内边距以适应字体大小 */
                font-weight: 600;
                font-size: 16px; /* 字体大小从14px增加到16px */
                margin-right: 2px;
                transition: all 0.3s;
                min-width: 200px;
            }

            #MainTabs QTabBar::tab:selected, #MainTabs QTabBar::tab:hover {
                background: #2c4b7c;
                color: #ffffff;
            }

            #MainTabs QTabBar::tab:selected {
                font-weight: 700;
                font-size: 17px; /* 字体大小从15px增加到17px */
            }

            #MainTabs QScrollBar:vertical {
                background: #f0f4f8;
                width: 12px;
                margin: 0px;
            }

            #MainTabs QScrollBar::handle:vertical {
                background: #3f72af;
                min-height: 20px;
            }

            #MainTabs QScrollBar::handle:vertical:hover {
                background: #2c4b7c;
            }

            #MainTabs QScrollBar::add-line:vertical, #MainTabs QScrollBar::sub-line:vertical {
                height: 0px;
            }

            /* 仅针对Page2Widget里这三个按钮 */
            QPushButton#selectPageBtn {
                font-size: 16px; /* 字体大小从14px增加到16px */
                font-weight: 100;
                padding: 2px 2px; /* 调整内边距以适应字体大小 */
                min-width: 500px; /* 最小宽度保持不变 */
                max-width: none;
            }
            QPushButton#selectPageBtn:hover {
                background-color: #2c4b7c;
                color: white;
            }
            """


if __name__ == '__main__':
    app = QApplication(sys.argv)

    login = LoginWindow()
    if login.exec_() == QDialog.Accepted:
        win = IntegratedMainWindow(role=login.role)
        win.show()
        sys.exit(app.exec_())
    else:
        # 登录失败或点击关闭按钮都退出程序
        sys.exit(0)


