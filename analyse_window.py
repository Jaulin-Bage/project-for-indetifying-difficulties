# TODO 将右边的选择按钮弹出界面改为下拉框

# -*- coding: utf-8 -*-
"""
增强的分析界面，包含筛选控件和图表显示功能
"""

import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout,
                            QVBoxLayout, QFrame, QPushButton, QScrollArea,
                            QLabel, QGroupBox, QDialog, QDialogButtonBox,
                            QListWidget, QListWidgetItem, QAbstractItemView,
                            QComboBox, QLineEdit)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# 确保能导入本地模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from analyse_utils import (get_selected_info, get_available_schools_and_years,
                          plot_people_school, plot_people_year,
                          plot_poverty_distribution, plot_poverty_trend,
                          plot_poverty_school_dist)

PIC_FIXED_HEIGHT = 600
PIC_MAX_WIDTH = 800


def build_info_box_style(font_size=18, font_weight=400):
    return f"""
        QLabel {{
            font-size: {font_size}px;
            font-weight: {font_weight};
            line-height: 1.6;
            padding: 16px 20px;
            background-color: rgb(245, 248, 255);
            border: 1px solid rgb(205, 218, 255);
            border-radius: 8px;
            color: rgb(38, 50, 72);
        }}
    """


class SelectionDialog(QtWidgets.QDialog):
    """通用的多选/单选弹窗"""

    def __init__(self,
                 title,
                 items,
                 selected_items=None,
                 multi_select=True,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(420, 520)

        layout = QVBoxLayout(self)

        info_label = QLabel("可通过按住 Ctrl 或 Shift 进行多选")
        info_label.setStyleSheet("color: rgb(120, 120, 120); font-size: 18px;")
        info_label.setVisible(multi_select)
        layout.addWidget(info_label)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(
            QAbstractItemView.MultiSelection
            if multi_select else QAbstractItemView.SingleSelection)

        # 填充条目
        for item in items:
            list_item = QListWidgetItem(item)
            self.list_widget.addItem(list_item)

        # 恢复已选状态
        selected_items = selected_items or []
        selected_set = set(selected_items)
        for index in range(self.list_widget.count()):
            item_text = self.list_widget.item(index).text()
            if item_text in selected_set:
                self.list_widget.item(index).setSelected(True)

        # 如果是单选且当前没有选择，默认选中第一个
        if (not multi_select and self.list_widget.count() > 0
                and not self.list_widget.selectedItems()):
            self.list_widget.item(0).setSelected(True)

        layout.addWidget(self.list_widget)

        if multi_select:
            btn_layout = QHBoxLayout()
            select_all_btn = QPushButton("全选")
            clear_btn = QPushButton("全不选")
            select_all_btn.clicked.connect(self.list_widget.selectAll)
            clear_btn.clicked.connect(self.list_widget.clearSelection)
            btn_layout.addWidget(select_all_btn)
            btn_layout.addWidget(clear_btn)
            layout.addLayout(btn_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok
                                      | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_selected_items(self):
        items = self.list_widget.selectedItems()
        # 按照列表原始顺序排序
        sorted_items = sorted(items, key=lambda itm: self.list_widget.row(itm))
        return [item.text() for item in sorted_items]

class AnalyseWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("数据分析界面")
        self.setGeometry(100, 100, 1200, 800)

        app = QtWidgets.QApplication.instance()
        if app is not None:
            default_font = app.font()
            default_font.setFamily("Microsoft YaHei")
            app.setFont(default_font)

        # 当前选择的功能
        self.current_function = "概览"

        # 获取可用的学校和年份
        self.available_schools, self.available_years = get_available_schools_and_years(
        )

        # 当前选中的学校和年份
        self.selected_schools = list(self.available_schools)
        self.selected_years = list(self.available_years)
        self.selection_summary_label = None

        # 初始化界面
        self.init_ui()

        # 根据默认功能调整选择逻辑
        self.adjust_selection_for_function(self.current_function)
        self.update_selection_display()

    def init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)

        # 创建左侧面板
        self.create_left_panel()
        main_layout.addWidget(self.left_frame)

        # 创建右侧面板
        self.create_right_panel()
        main_layout.addWidget(self.right_frame)

    def create_left_panel(self):
        """创建左侧控制面板"""
        self.left_frame = QFrame()
        self.left_frame.setMinimumSize(QtCore.QSize(250, 100))
        self.left_frame.setMaximumSize(QtCore.QSize(300, 16777215))
        self.left_frame.setStyleSheet(
            "QFrame { background-color: white; border: 2px solid gray; }")

        left_layout = QVBoxLayout(self.left_frame)

        # 功能按钮组
        self.create_function_buttons(left_layout)

        # 填充间距
        left_layout.addStretch()

    def create_function_buttons(self, layout):
        """创建功能按钮组"""
        # 功能按钮组
        function_group = QGroupBox("分析功能")
        function_layout = QVBoxLayout(function_group)

        # 创建按钮
        self.function_buttons = {}
        functions = ["概览", "按年份对比", "按学校对比", "指标权重分析", "个体分析"]

        for func in functions:
            btn = QPushButton(func)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: white;
                    border: none;
                    height: 30px;
                    text-align: left;
                    padding-left: 10px;
                }
                QPushButton:hover {
                    background-color: rgb(222, 237, 255);
                }
                QPushButton:checked {
                    background-color: rgb(0, 120, 215);
                    color: white;
                }
            """)
            btn.setCheckable(True)
            btn.clicked.connect(
                lambda checked, f=func: self.on_function_selected(f))
            self.function_buttons[func] = btn
            function_layout.addWidget(btn)

        # 默认选中概览
        self.function_buttons["概览"].setChecked(True)

        layout.addWidget(function_group)

    def create_right_panel(self):
        """创建右侧内容面板"""
        self.right_frame = QFrame()
        self.right_frame.setStyleSheet(
            "QFrame { background-color: rgb(255, 255, 255); }")

        right_layout = QVBoxLayout(self.right_frame)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(16)

        selector_btn_style = """
            QPushButton {
                background-color: rgb(255, 255, 255);
                color: rgb(0, 120, 215);
                border: 1px solid rgb(0, 120, 215);
                padding: 0 18px;
                font-size: 18px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: rgb(230, 240, 255);
            }
            QPushButton:disabled {
                color: rgb(160, 160, 160);
                border-color: rgb(200, 200, 200);
                background-color: rgb(245, 245, 245);
            }
            QComboBox {
                background-color: rgb(245, 248, 255);
                color: rgb(0, 120, 215);
                border: none;
                padding: 0px 26px 0px 14px;
                font-size: 18px;
                border-radius: 5px;
                min-height: 26px;
            }
            QComboBox:hover {
                background-color: rgb(222, 237, 255);
            }
            QComboBox:focus {
                background-color: rgb(212, 232, 255);
            }
            QComboBox QLineEdit {
                border: none;
                padding: 0;
                margin: 0;
                background: transparent;
                selection-background-color: rgb(0, 120, 215);
                selection-color: white;
            }
            QComboBox:!enabled {
                color: rgb(160, 160, 160);
                background-color: rgb(245, 245, 245);
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 24px;
                border: none;
                background: transparent;
                margin: 0;
            }
            QComboBox::down-arrow {
                image: none;
                width: 0px;
                height: 0px;
            }
            QComboBox QAbstractItemView {
                padding: 4px 0;
                outline: none;
            }
            QComboBox QAbstractItemView::item {
                min-height: 28px;
                padding: 4px 4px;
            }
        """

        control_layout = QHBoxLayout()
        control_layout.setSpacing(12)

        selectors_widget = QWidget()
        selectors_layout = QHBoxLayout(selectors_widget)
        selectors_layout.setContentsMargins(0, 0, 0, 0)
        selectors_layout.setSpacing(12)

        self.school_combo = QComboBox()
        self.school_combo.setFixedHeight(36)
        self.school_combo.setMinimumWidth(200)
        self.school_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.school_combo.setStyleSheet(selector_btn_style)
        # 允许自定义展示文本（只读）以显示多选摘要
        self.school_combo.setEditable(True)
        school_line_edit = self.school_combo.lineEdit()
        school_line_edit.setReadOnly(True)
        school_line_edit.setFrame(False)
        school_line_edit.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        school_line_edit.setContentsMargins(0, 0, 0, 0)
        school_line_edit.setStyleSheet("padding: 0; margin: 0; border: none; background: transparent;")
        self.school_combo.setInsertPolicy(QComboBox.NoInsert)
        self._configure_combo_view(self.school_combo)
        self._ensure_combo_full_click(self.school_combo)

        self.year_combo = QComboBox()
        self.year_combo.setFixedHeight(36)
        self.year_combo.setMinimumWidth(200)
        self.year_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.year_combo.setStyleSheet(selector_btn_style)
        self.year_combo.setEditable(True)
        year_line_edit = self.year_combo.lineEdit()
        year_line_edit.setReadOnly(True)
        year_line_edit.setFrame(False)
        year_line_edit.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        year_line_edit.setContentsMargins(0, 0, 0, 0)
        year_line_edit.setStyleSheet("padding: 0; margin: 0; border: none; background: transparent;")
        self.year_combo.setInsertPolicy(QComboBox.NoInsert)
        self._configure_combo_view(self.year_combo)
        self._ensure_combo_full_click(self.year_combo)

        selectors_layout.addWidget(self.school_combo, 1)
        selectors_layout.addWidget(self.year_combo, 1)

        control_layout.addWidget(selectors_widget, 1)
        control_layout.addStretch()

        self.analyse_btn = QPushButton("分析")
        self.analyse_btn.setFixedHeight(40)
        self.analyse_btn.setMinimumWidth(140)
        self.analyse_btn.setStyleSheet("""
            QPushButton {
                background-color: rgb(0, 120, 215);
                color: white;
                border: none;
                height: 40px;
                font-size: 18px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: rgb(0, 100, 180);
            }
            QPushButton:disabled {
                background-color: rgb(200, 200, 200);
            }
        """)
        self.analyse_btn.clicked.connect(self.perform_analysis)
        control_layout.addWidget(self.analyse_btn)

        right_layout.addLayout(control_layout)
        right_layout.addSpacing(28)

        # 初始化下拉内容
        self._populate_selection_combos()

        self.content_scroll = QScrollArea()
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAsNeeded)
        self.content_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(24, 16, 24, 24)
        self.content_layout.setSpacing(18)
        self.content_scroll.setWidget(self.content_widget)

        right_layout.addWidget(self.content_scroll)

        # 显示初始说明
        self.show_function_description()

    def on_function_selected(self, function_name):
        """功能按钮选择事件"""
        for name, btn in self.function_buttons.items():
            btn.setChecked(name == function_name)

        self.current_function = function_name

        self.adjust_selection_for_function(function_name)
        # 根据新功能调整下拉框的模式（单选/多选）并刷新显示
        self._populate_selection_combos()
        self.update_selection_display()
        self.show_function_description()

    def adjust_selection_for_function(self, function_name):
        """根据功能调整选择逻辑"""
        if function_name == "按年份对比":
            if self.selected_schools:
                self.selected_schools = self.selected_schools[:1]
            elif self.available_schools:
                self.selected_schools = [self.available_schools[0]]

            if not self.selected_years and self.available_years:
                self.selected_years = list(self.available_years)

        elif function_name == "按学校对比":
            if self.selected_years:
                self.selected_years = self.selected_years[:1]
            elif self.available_years:
                self.selected_years = [self.available_years[0]]

            if not self.selected_schools and self.available_schools:
                self.selected_schools = list(self.available_schools)

        else:
            if not self.selected_schools and self.available_schools:
                self.selected_schools = list(self.available_schools)
            if not self.selected_years and self.available_years:
                self.selected_years = list(self.available_years)

    def update_selection_display(self):
        self.update_selection_button_texts()
        self.update_selection_summary_label()

    def update_selection_button_texts(self):
        # 更新下拉框右侧展示文本（摘要）和启用状态
        if hasattr(self, 'school_combo'):
            if not self.available_schools:
                self.school_combo.setEnabled(False)
                self._set_combo_caption(self.school_combo, True, text="暂无可选学校")
            else:
                self.school_combo.setEnabled(True)
                self._set_combo_caption(self.school_combo, True)

        if hasattr(self, 'year_combo'):
            if not self.available_years:
                self.year_combo.setEnabled(False)
                self._set_combo_caption(self.year_combo, False, text="暂无可选年份")
            else:
                self.year_combo.setEnabled(True)
                self._set_combo_caption(self.year_combo, False)

        self._update_selector_summaries()

    def _describe_selection(self, selected, available, unit):
        if not available:
            return f"暂无{unit}"
        if not selected:
            return f"未选择{unit}"
        if len(selected) == len(available):
            return f"全部{unit}"
        if len(selected) <= 3:
            return "、".join(selected)
        return f"{'、'.join(selected[:3])} 等{len(selected)}{unit}"

    def get_selection_summary_text(self):
        school_desc = self._describe_selection(self.selected_schools,
                                               self.available_schools, "学校")
        year_desc = self._describe_selection(self.selected_years,
                                             self.available_years, "年份")
        return f"已选学校：{school_desc}\n已选年份：{year_desc}"

    def update_selection_summary_label(self):
        if isinstance(self.selection_summary_label, QLabel):
            self.selection_summary_label.setText(
                self.get_selection_summary_text())

    @staticmethod
    def _sort_by_reference(items, reference):
        index_map = {value: idx for idx, value in enumerate(reference)}
        filtered = [item for item in items if item in index_map]
        return sorted(filtered, key=lambda value: index_map[value])

    def _create_info_label(self, text="", emphasis=False):
        font_size = 17 if emphasis else 15
        font_weight = 600 if emphasis else 400
        label = QLabel(text)
        label.setWordWrap(True)
        label.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                            QtWidgets.QSizePolicy.Minimum)
        label.setStyleSheet(build_info_box_style(font_size, font_weight))
        return label

    def _register_chart_canvas(self, canvas):
        canvas.setFixedHeight(PIC_FIXED_HEIGHT)
        canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                             QtWidgets.QSizePolicy.Fixed)
        self._chart_canvases.append(canvas)
        self._resize_chart_canvas(canvas)
        QtCore.QTimer.singleShot(0, self._resize_all_chart_canvases)

    def _resize_chart_canvas(self, canvas):
        if not hasattr(self, "content_scroll"):
            return
        viewport = self.content_scroll.viewport()
        viewport_width = viewport.width() if viewport is not None else 0
        if viewport_width <= 0 and hasattr(self, "right_frame"):
            viewport_width = self.right_frame.width()

        margins = self.content_layout.contentsMargins()
        available_width = viewport_width - margins.left() - margins.right()
        available_width = max(0, available_width)

        if available_width > 0:
            target_width = min(available_width, PIC_MAX_WIDTH)
        else:
            fallback_sources = [getattr(self.content_scroll, "width", lambda: 0)(),
                                getattr(self.right_frame, "width", lambda: 0)(),
                                self.width()]
            fallback_width = next((value for value in fallback_sources if value and value > 0), PIC_MAX_WIDTH)
            target_width = min(fallback_width, PIC_MAX_WIDTH)

        canvas.setMinimumWidth(target_width)
        canvas.setMaximumWidth(target_width)
        canvas.updateGeometry()

    def _resize_all_chart_canvases(self):
        for canvas in getattr(self, "_chart_canvases", []):
            self._resize_chart_canvas(canvas)

    def _populate_selection_combos(self):
        """根据当前 available 和 current_function 填充并配置下拉框（支持多选复选）"""
        # 学校：在“按年份对比”中为单选，否则多选
        school_multi = self.current_function != "按年份对比"
        year_multi = self.current_function != "按学校对比"

        self._setup_combo(self.school_combo, self.available_schools,
                          self.selected_schools, school_multi, is_school=True)
        self._setup_combo(self.year_combo, self.available_years,
                          self.selected_years, year_multi, is_school=False)

    def _set_combo_caption(self, combo, is_school, text=None):
        """统一设置下拉框的展示文本，默认显示固定提示语。"""
        caption = text if text is not None else ("选择学校" if is_school else "选择年份")
        if combo.isEditable():
            def apply_text():
                line_edit = combo.lineEdit()
                if line_edit is None:
                    return
                previous_state = line_edit.blockSignals(True)
                line_edit.setText(caption)
                line_edit.blockSignals(previous_state)
                line_edit.setCursorPosition(0)
            # 使用单次定时器，确保覆盖 QComboBox 默认的文本更新。
            QtCore.QTimer.singleShot(0, apply_text)
        else:
            combo.setCurrentText(caption)

    def _configure_combo_view(self, combo):
        """调整下拉列表项的间距和内边距。"""
        view = QtWidgets.QListView()
        view.setSpacing(8)
        view.setMouseTracking(True)
        view.setStyleSheet("QListView::item { padding: 10px 16px; }")
        combo.setView(view)

    def _ensure_combo_full_click(self, combo):
        line_edit = combo.lineEdit()
        if line_edit is None:
            return
        existing_filter = getattr(combo, "_line_click_filter", None)
        if existing_filter:
            line_edit.removeEventFilter(existing_filter)

        class _ComboClickFilter(QtCore.QObject):
            def __init__(self, combo_widget):
                super().__init__(combo_widget)
                self._combo = combo_widget

            def eventFilter(self, obj, event):
                if event.type() == QtCore.QEvent.MouseButtonPress:
                    self._combo.showPopup()
                    return True
                return super().eventFilter(obj, event)

        filter_obj = _ComboClickFilter(combo)
        combo._line_click_filter = filter_obj
        line_edit.installEventFilter(filter_obj)
        line_edit.setCursor(QtCore.Qt.PointingHandCursor)
        line_edit.setFocusPolicy(QtCore.Qt.NoFocus)

    def _attach_full_row_toggle(self, combo, model):
        view = combo.view()
        if view is None:
            return
        if hasattr(view, "_full_row_handler"):
            try:
                view.pressed.disconnect(view._full_row_handler)
            except (TypeError, RuntimeError):
                pass
        def handle_pressed(index, m=model):
            item = m.itemFromIndex(index)
            if item is None:
                return
            new_state = QtCore.Qt.Unchecked if item.checkState() == QtCore.Qt.Checked else QtCore.Qt.Checked
            item.setCheckState(new_state)
        view.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        view._full_row_handler = handle_pressed
        view.pressed.connect(handle_pressed)

    def _detach_full_row_toggle(self, combo):
        view = combo.view()
        if view is None:
            return
        if hasattr(view, "_full_row_handler"):
            try:
                view.pressed.disconnect(view._full_row_handler)
            except (TypeError, RuntimeError):
                pass
            delattr(view, "_full_row_handler")
        view.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

    def _update_selector_summaries(self):
        """刷新下拉框显示文本以反映当前勾选摘要。"""
        if hasattr(self, "school_combo") and self.school_combo.isEnabled():
            school_desc = self._describe_selection(self.selected_schools, self.available_schools, "学校")
            self._set_combo_caption(self.school_combo, True, text=school_desc)
        if hasattr(self, "year_combo") and self.year_combo.isEnabled():
            year_desc = self._describe_selection(self.selected_years, self.available_years, "年份")
            self._set_combo_caption(self.year_combo, False, text=year_desc)

    def _setup_combo(self, combo, items, selected, multi_select, is_school=True):
        """填充 combo，如果 multi_select=True 则使用可复选的 QStandardItemModel；否则使用普通项。
        is_school 用于区分信号处理（学校/年份）。"""
        # 断开信号，避免重复响应
        try:
            combo.blockSignals(True)
        except Exception:
            pass

        combo.clear()

        if not items:
            self._detach_full_row_toggle(combo)
            combo.setEnabled(False)
            empty_caption = "暂无可选学校" if is_school else "暂无可选年份"
            self._set_combo_caption(combo, is_school, text=empty_caption)
            try:
                combo.blockSignals(False)
            except Exception:
                pass
            self._update_selector_summaries()
            return

        combo.setEnabled(True)

        if multi_select:
            # 使用 QStandardItemModel + checkable items
            model = QtGui.QStandardItemModel()
            for it in items:
                it_item = QtGui.QStandardItem(it)
                it_item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
                if it in selected:
                    it_item.setData(QtCore.Qt.Checked, QtCore.Qt.CheckStateRole)
                else:
                    it_item.setData(QtCore.Qt.Unchecked, QtCore.Qt.CheckStateRole)
                model.appendRow(it_item)

            combo.setModel(model)
            self._set_combo_caption(combo, is_school)
            self._attach_full_row_toggle(combo, model)

            # 绑定模型变化
            # 先尝试断开旧连接，然后连接新信号
            if is_school:
                model.itemChanged.connect(self._on_school_model_item_changed)
            else:
                model.itemChanged.connect(self._on_year_model_item_changed)

        else:
            self._detach_full_row_toggle(combo)
            # 单选：普通 QComboBox 项
            combo.addItems(items)
            # 选择第一个已选择或第一个元素
            if selected:
                try:
                    idx = items.index(selected[0])
                except ValueError:
                    idx = 0
            else:
                idx = 0
            combo.setCurrentIndex(idx)
            self._set_combo_caption(combo, is_school)
            # 绑定单选变化
            # 避免重复连接，先尝试断开所有已连接的槽
            try:
                combo.currentIndexChanged.disconnect()
            except Exception:
                pass
            if is_school:
                combo.currentIndexChanged.connect(self._on_school_single_changed)
            else:
                combo.currentIndexChanged.connect(self._on_year_single_changed)

        try:
            combo.blockSignals(False)
        except Exception:
            pass

        self._set_combo_caption(combo, is_school)
        self._update_selector_summaries()

    def _read_checked_items_from_model(self, model):
        items = []
        for row in range(model.rowCount()):
            it = model.item(row)
            state = it.data(QtCore.Qt.CheckStateRole)
            if state == QtCore.Qt.Checked:
                items.append(it.text())
        return items

    def _on_school_model_item_changed(self, item):
        # 读取当前 model 中被选中的学校，按 available_schools 顺序排序
        model = item.model()
        checked = self._read_checked_items_from_model(model)
        self.selected_schools = self._sort_by_reference(checked, self.available_schools)
        self._set_combo_caption(self.school_combo, True)
        self._update_selector_summaries()
        self.update_selection_summary_label()

    def _on_year_model_item_changed(self, item):
        model = item.model()
        checked = self._read_checked_items_from_model(model)
        self.selected_years = self._sort_by_reference(checked, self.available_years)
        self._set_combo_caption(self.year_combo, False)
        self._update_selector_summaries()
        self.update_selection_summary_label()

    def _on_school_single_changed(self, index):
        txt = self.school_combo.itemText(index)
        self.selected_schools = [txt] if txt else []
        self._set_combo_caption(self.school_combo, True)
        self._update_selector_summaries()
        self.update_selection_summary_label()

    def _on_year_single_changed(self, index):
        txt = self.year_combo.itemText(index)
        self.selected_years = [txt] if txt else []
        self._set_combo_caption(self.year_combo, False)
        self._update_selector_summaries()
        self.update_selection_summary_label()

    def on_select_school_clicked(self):
        if not self.available_schools:
            QtWidgets.QMessageBox.information(self, "提示", "暂无可选学校。")
            return

        multi_select = self.current_function != "按年份对比"
        dialog = SelectionDialog("选择学校", self.available_schools,
                                 self.selected_schools, multi_select, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            selected = dialog.get_selected_items()
            if not selected:
                QtWidgets.QMessageBox.warning(self, "警告", "请至少选择一个学校！")
                return
            self.selected_schools = self._sort_by_reference(
                selected, self.available_schools)
            self.adjust_selection_for_function(self.current_function)
            self.update_selection_display()

    def on_select_year_clicked(self):
        if not self.available_years:
            QtWidgets.QMessageBox.information(self, "提示", "暂无可选年份。")
            return

        multi_select = self.current_function != "按学校对比"
        dialog = SelectionDialog("选择年份", self.available_years,
                                 self.selected_years, multi_select, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            selected = dialog.get_selected_items()
            if not selected:
                QtWidgets.QMessageBox.warning(self, "警告", "请至少选择一个年份！")
                return
            self.selected_years = self._sort_by_reference(
                selected, self.available_years)
            self.adjust_selection_for_function(self.current_function)
            self.update_selection_display()

    def show_function_description(self):
        """显示功能描述"""
        self.clear_content()

        descriptions = {
            "概览":
            "概览功能将显示以下分析图表：\n\n• 各学校参与认定的人员数量统计\n• 各年份参与认定的人员数量统计\n• 困难程度分布扇形图\n• 困难度分数随年份变化趋势\n• 各学校困难度分数统计对比\n\n请使用上方的“选择学校”和“选择年份”按钮设定范围，然后点击“分析”查看结果。",
            "按年份对比":
            "按年份对比功能将分析选定学校在不同时间段的变化趋势：\n\n• 只能选择1个学校进行分析\n• 默认选择所有年份进行对比\n• 显示该学校的认定人数统计信息\n• 展示年份维度的图表分析\n\n请先通过“选择学校”按钮确认单一学校，再点击“分析”查看结果。",
            "按学校对比":
            "按学校对比功能将分析选定年份下不同学校的差异：\n\n• 只能选择1个年份进行分析\n• 默认选择所有学校进行对比\n• 显示该年份的总体统计信息\n• 展示学校维度的图表分析\n\n请先通过“选择年份”按钮确认单一年份，再点击“分析”查看结果。",
            "指标权重分析": "指标权重分析功能将展示各项指标在困难度评估中的权重分布。\n\n请选择数据范围，然后点击“分析”按钮。",
            "个体分析": "个体分析功能将提供针对特定学生的详细分析。\n\n请选择数据范围，然后点击“分析”按钮。"
        }

        description = descriptions.get(self.current_function, "请选择一个分析功能")

        label = self._create_info_label(description)

        self.content_layout.addWidget(label)
        self.content_layout.addStretch()

    def clear_content(self):
        """清空内容区域并刷新选择摘要"""
        self._chart_canvases = []
        while self.content_layout.count():
            child = self.content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.selection_summary_label = self._create_info_label()
        self.content_layout.addWidget(self.selection_summary_label)
        self.update_selection_summary_label()
        self.content_layout.addSpacing(10)

    def perform_analysis(self):
        """执行分析"""
        # 确保选择符合当前功能的要求
        self.adjust_selection_for_function(self.current_function)
        self.update_selection_display()

        selected_schools = list(self.selected_schools)
        selected_years = list(self.selected_years)

        # 根据功能验证选择
        if self.current_function == "按年份对比":
            if len(selected_schools) != 1:
                QtWidgets.QMessageBox.warning(self, "警告", "按年份对比只能选择1个学校！")
                return
            if not selected_years:
                QtWidgets.QMessageBox.warning(self, "警告", "请至少选择一个年份！")
                return
        elif self.current_function == "按学校对比":
            if len(selected_years) != 1:
                QtWidgets.QMessageBox.warning(self, "警告", "按学校对比只能选择1个年份！")
                return
            if not selected_schools:
                QtWidgets.QMessageBox.warning(self, "警告", "请至少选择一个学校！")
                return
        else:
            if not selected_schools:
                QtWidgets.QMessageBox.warning(self, "警告", "请至少选择一个学校！")
                return
            if not selected_years:
                QtWidgets.QMessageBox.warning(self, "警告", "请至少选择一个年份！")
                return

        # 获取筛选后的数据
        try:
            data_result = get_selected_info("output", selected_schools,
                                            selected_years)
            info = data_result['info']

            if not info:
                QtWidgets.QMessageBox.information(self, "提示", "没有找到符合条件的数据！")
                return

            # 根据当前功能执行相应分析
            if self.current_function == "概览":
                self.show_overview_analysis(info)
            elif self.current_function == "按年份对比":
                self.show_time_comparison_analysis(info, selected_schools[0])
            elif self.current_function == "按学校对比":
                self.show_space_comparison_analysis(info, selected_years[0])
            else:
                # 其他功能暂时显示占位信息
                self.show_placeholder_analysis()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"分析过程中发生错误：{str(e)}")

    def show_overview_analysis(self, info):
        """显示概览分析结果"""
        self.clear_content()
        self._chart_canvases = []

        # 创建图表列表
        charts = [("各学校参与认定的人员数量统计", lambda: plot_people_school(info)),
                  ("各年份参与认定的人员数量统计", lambda: plot_people_year(info)),
                  ("困难程度分布扇形图", lambda: plot_poverty_distribution(info)),
                  ("困难度分数随年份变化趋势", lambda: plot_poverty_trend(info)),
                  ("各学校困难度分数统计对比", lambda: plot_poverty_school_dist(info))]

        # 为每个图表创建画布
        for title, plot_func in charts:
            try:
                # 创建标题
                title_label = QLabel(title)
                title_label.setStyleSheet("""
                    QLabel {
                        font-size: 18px;
                        font-weight: bold;
                        padding: 10px;
                        background-color: rgb(240, 240, 240);
                        border-bottom: 2px solid rgb(0, 120, 215);
                    }
                """)
                self.content_layout.addWidget(title_label)

                # 生成图表
                fig = plot_func()

                # 创建画布
                canvas = FigureCanvas(fig)
                self.content_layout.addWidget(canvas)
                self._register_chart_canvas(canvas)

                # 关闭matplotlib图形以释放内存
                plt.close(fig)

            except Exception as e:
                error_label = QLabel(f"生成图表时发生错误: {str(e)}")
                error_label.setStyleSheet(
                    "QLabel { color: red; padding: 10px; }")
                self.content_layout.addWidget(error_label)

        # 添加弹性空间
        self.content_layout.addStretch()

    def show_time_comparison_analysis(self, info, selected_school):
        """显示按年份对比分析结果（合并唯一版本，图片宽度自适应）"""
        self.clear_content()
        self._chart_canvases = []
        if selected_school not in info:
            error_label = QLabel(f"未找到学校 {selected_school} 的数据")
            error_label.setStyleSheet(
                "QLabel { color: red; padding: 10px; font-size: 18px; }")
            self.content_layout.addWidget(error_label)
            return

        school_info = info[selected_school]

        # 统计信息
        total_students = sum(year_data["学生总数"] for year_data in school_info.values())
        all_scores = []
        for year_data in school_info.values():
            if year_data["学生总数"] > 0:
                all_scores.extend([year_data["平均困难度分数"]] * year_data["学生总数"])

        stats_lines = [f"学校：{selected_school}", "", f"认定总人数：{total_students}"]

        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            max_score = max(year_data["最大困难度分数"] for year_data in school_info.values())
            min_score = min(year_data["最小困难度分数"] for year_data in school_info.values() if year_data["最小困难度分数"] != float('inf'))
            median_scores = [year_data["困难度分数中位数"] for year_data in school_info.values()]
            median_score = sum(median_scores) / len(median_scores) if median_scores else 0

            stats_lines.extend([
                f"平均困难度分数：{avg_score:.2f}",
                f"最大困难度分数：{max_score:.2f}",
                f"最小困难度分数：{min_score:.2f}",
                f"中位数困难度分数：{median_score:.2f}"
            ])
        else:
            stats_lines.append("无有效分数数据。")

        stats_label = self._create_info_label("\n".join(stats_lines), emphasis=True)
        self.content_layout.addWidget(stats_label)

        # 图表
        single_school_info = {selected_school: school_info}
        charts = [
            ("该学校各年份参与认定的人员数量统计", lambda: plot_people_year(single_school_info)),
            ("该学校困难程度分布", lambda: plot_poverty_distribution(single_school_info)),
            ("该学校困难度分数随年份变化趋势", lambda: plot_poverty_trend(single_school_info))
        ]
        for title, plot_func in charts:
            try:
                title_label = QLabel(title)
                title_label.setStyleSheet("""
                    QLabel {
                        font-size: 18px;
                        font-weight: bold;
                        padding: 10px;
                        background-color: rgb(240, 240, 240);
                        border-bottom: 2px solid rgb(0, 120, 215);
                    }
                """)
                self.content_layout.addWidget(title_label)
                fig = plot_func()
                canvas = FigureCanvas(fig)
                self.content_layout.addWidget(canvas)
                self._register_chart_canvas(canvas)
                plt.close(fig)
            except Exception as e:
                error_label = QLabel(f"生成图表时发生错误: {str(e)}")
                error_label.setStyleSheet("QLabel { color: red; padding: 10px; }")
                self.content_layout.addWidget(error_label)
        self.content_layout.addStretch()

    def show_space_comparison_analysis(self, info, selected_year):
        """显示按学校对比分析结果（修复问题）"""
        self.clear_content()
        self._chart_canvases = []

        # 筛选出指定年份的数据
        year_info = {}
        total_students = 0

        for school, school_data in info.items():
            if selected_year in school_data:
                year_info[school] = {selected_year: school_data[selected_year]}
                total_students += school_data[selected_year]["学生总数"]

        if not year_info:
            error_label = QLabel(f"未找到年份 {selected_year} 的数据")
            error_label.setStyleSheet(
                "QLabel { color: red; padding: 10px; font-size: 18px; }")
            self.content_layout.addWidget(error_label)
            return

        # 创建统计信息文本
        stats_lines = [f"年份：{selected_year}", "", f"参与学校数量：{len(year_info)}",
                       f"认定总人数：{total_students}"]

        stats_label = self._create_info_label("\n".join(stats_lines), emphasis=True)
        self.content_layout.addWidget(stats_label)

        # 创建图表
        charts = [
            ("各学校参与认定的人员数量统计", lambda: plot_people_school(year_info)),
            ("困难程度分布", lambda: plot_poverty_distribution(year_info)),
            ("各学校困难度分数统计对比", lambda: plot_poverty_school_dist(year_info))
        ]

        for title, plot_func in charts:
            try:
                title_label = QLabel(title)
                title_label.setStyleSheet("""
                    QLabel {
                        font-size: 18px;
                        font-weight: bold;
                        padding: 10px;
                        background-color: rgb(240, 240, 240);
                        border-bottom: 2px solid rgb(0, 120, 215);
                    }
                """)
                self.content_layout.addWidget(title_label)

                fig = plot_func()
                canvas = FigureCanvas(fig)
                self.content_layout.addWidget(canvas)
                self._register_chart_canvas(canvas)
                plt.close(fig)

            except Exception as e:
                error_label = QLabel(f"生成图表时发生错误: {str(e)}")
                error_label.setStyleSheet("QLabel { color: red; padding: 10px; }")
                self.content_layout.addWidget(error_label)

        self.content_layout.addStretch()

    def resizeEvent(self, event):
        """窗口大小变化时调整图表宽度（修复问题）"""
        super().resizeEvent(event)
        self._resize_all_chart_canvases()

    def show_placeholder_analysis(self):
        """显示占位分析（用于其他功能）"""
        self.clear_content()

        label = QLabel(f"{self.current_function}功能正在开发中，敬请期待...")
        label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                padding: 50px;
                text-align: center;
                color: rgb(100, 100, 100);
            }
        """)
        label.setAlignment(QtCore.Qt.AlignCenter)

        self.content_layout.addWidget(label)
        self.content_layout.addStretch()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnalyseWindow()
    window.show()
    sys.exit(app.exec_())
