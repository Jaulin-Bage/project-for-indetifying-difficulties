import textwrap
from pathlib import Path

import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from params import RESULT_VECTOR_MAPPING
from traditional_ml import (
    DEFAULT_TEST_DATASET,
    DEFAULT_TRAIN_DATASET,
    LABEL_COLUMN,
    train_traditional_model,
)


MODEL_NAMES = {
    "随机森林": "forest",
    #"极端随机树": "extra_trees",
    "逻辑回归": "logistic",
    #"线性SVM": "linear_svm",
    #"K近邻": "knn",
    #"梯度提升树": "gradient_boosting",
    #"决策树": "tree",
    "高斯朴素贝叶斯": "bayes",
}


class TrainingWorker(QThread):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            self.completed.emit(train_traditional_model(**self.params))
        except Exception as exc:
            self.failed.emit(str(exc))


class TraditionalMLWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("传统机器学习指标体系生成")
        self.worker = None
        self.result = None
        self.trained_model_label = None
        self.init_ui()

    def init_ui(self):
        root = QWidget()
        root_layout = QHBoxLayout(root)
        splitter = QSplitter()
        root_layout.addWidget(splitter)
        self.setCentralWidget(root)

        controls = QWidget()
        controls.setMaximumWidth(390)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setSpacing(12)

        file_group = QGroupBox("数据文件")
        file_form = QFormLayout(file_group)
        self.dataset_edit = self.create_path_row(
            file_form, "训练集", self.select_dataset)
        self.dataset_edit.setText(str(DEFAULT_TRAIN_DATASET))
        self.test_dataset_edit = self.create_path_row(
            file_form, "测试集", self.select_test_dataset)
        self.test_dataset_edit.setText(str(DEFAULT_TEST_DATASET))
        self.indicator_edit = self.create_path_row(
            file_form, "原指标体系", self.select_indicator)
        controls_layout.addWidget(file_group)

        model_group = QGroupBox("模型设置")
        model_form = QFormLayout(model_group)
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_NAMES.keys())
        self.min_importance_edit = QLineEdit("0.0")
        self.prior_strength_edit = QLineEdit("0.0")
        model_form.addRow("训练模型", self.model_combo)
        model_form.addRow("单项最低权重(%)", self.min_importance_edit)
        model_form.addRow("原权重保留比例", self.prior_strength_edit)
        constraint_tip = QLabel(
            "权重约束：所有模型都会保留每个指标的最低权重，并融合一部分原指标体系权重。"
        )
        constraint_tip.setWordWrap(True)
        model_form.addRow("", constraint_tip)
        controls_layout.addWidget(model_group)

        button_layout = QHBoxLayout()
        self.run_button = QPushButton("开始训练")
        self.run_button.clicked.connect(self.start_training)
        self.save_button = QPushButton("保存训练结果")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_result)
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.save_button)
        controls_layout.addLayout(button_layout)

        self.status_label = QLabel("请选择训练集、测试集和原指标体系")
        self.status_label.setWordWrap(True)
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch()

        self.tabs = QTabWidget()
        self.evaluation_scroll = QScrollArea()
        self.evaluation_scroll.setWidgetResizable(True)
        self.evaluation_content = QWidget()
        self.evaluation_layout = QVBoxLayout(self.evaluation_content)
        self.evaluation_layout.addWidget(QLabel("训练完成后显示模型评估结果"))
        self.evaluation_layout.addStretch()
        self.evaluation_scroll.setWidget(self.evaluation_content)
        self.tabs.addTab(self.evaluation_scroll, "传统机器学习评估")

        self.indicator_table = QTableWidget()
        self.indicator_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tabs.addTab(self.indicator_table, "新指标体系")

        splitter.addWidget(controls)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        self.setStyleSheet(self.stylesheet())

    def create_path_row(self, form, label, callback):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        edit.setReadOnly(True)
        button = QPushButton("浏览...")
        button.clicked.connect(callback)
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        form.addRow(label, container)
        return edit

    def select_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择训练集", "", "CSV Files (*.csv)")
        if path:
            self.dataset_edit.setText(path)

    def select_test_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择测试集", "", "CSV Files (*.csv)")
        if path:
            self.test_dataset_edit.setText(path)

    def select_indicator(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择原指标体系", "", "CSV Files (*.csv)")
        if path:
            self.indicator_edit.setText(path)

    def start_training(self):
        dataset = self.dataset_edit.text()
        test_dataset = self.test_dataset_edit.text()
        indicator = self.indicator_edit.text()
        if not all([dataset, test_dataset, indicator]):
            QMessageBox.warning(self, "提示", "请完整选择训练集、测试集和原指标体系。")
            return

        try:
            min_importance_percent = float(self.min_importance_edit.text())
            if min_importance_percent < 0:
                raise ValueError("单项最低权重不能小于 0。")
            prior_strength = float(self.prior_strength_edit.text())
            if not 0 <= prior_strength <= 1:
                raise ValueError("原权重保留比例必须在 0 和 1 之间。")
            columns = pd.read_csv(dataset, nrows=0).columns
            if LABEL_COLUMN not in columns:
                raise ValueError(f"训练集缺少“{LABEL_COLUMN}”列")
            test_columns = pd.read_csv(test_dataset, nrows=0).columns
            if LABEL_COLUMN not in test_columns:
                raise ValueError(f"测试集缺少“{LABEL_COLUMN}”列")
        except ValueError as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return
        except Exception as exc:
            QMessageBox.warning(self, "文件错误", str(exc))
            return

        model_name = MODEL_NAMES[self.model_combo.currentText()]
        self.trained_model_label = self.model_combo.currentText()
        params = {
            "dataset_path": dataset,
            "test_dataset_path": test_dataset,
            "indicator_path": indicator,
            "output_dir": Path("tmp"),
            "model_name": model_name,
            "new_indicator_path": Path("tmp") / f"traditional_{model_name}_indicator_system.csv",
            "min_importance_percent": min_importance_percent,
            "prior_strength": prior_strength,
        }
        self.result = None
        self.run_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.status_label.setText("正在训练模型...")
        self.worker = TrainingWorker(params, self)
        self.worker.completed.connect(self.training_completed)
        self.worker.failed.connect(self.training_failed)
        self.worker.finished.connect(lambda: self.run_button.setEnabled(True))
        self.worker.start()

    def training_completed(self, result):
        self.result = result
        self.save_button.setEnabled(True)
        self.status_label.setText("训练完成，可预览并保存训练结果")
        self.show_evaluation(result)
        self.show_indicator_table(result["new_indicator_df"])
        self.tabs.setCurrentIndex(0)
        QMessageBox.information(self, "完成", "模型训练完成，可点击“保存训练结果”。")

    def training_failed(self, message):
        self.status_label.setText("训练失败")
        self.save_button.setEnabled(False)
        QMessageBox.critical(self, "训练失败", message)

    def save_result(self):
        if self.result is None:
            QMessageBox.warning(self, "提示", "请先完成模型训练。")
            return

        source = Path(self.indicator_edit.text())
        default_name = source.with_name(
            f"{source.stem}_{self.trained_model_label}优化.csv")
        path, _ = QFileDialog.getSaveFileName(
            self, "保存训练结果", str(default_name), "CSV Files (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"

        try:
            self.result["new_indicator_df"].to_csv(
                path, index=False, encoding="utf-8-sig")
            self.status_label.setText(f"训练结果已保存：{path}")
            QMessageBox.information(self, "保存成功", f"新指标体系已保存至：\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def show_evaluation(self, result):
        self.clear_layout(self.evaluation_layout)
        model_name = self.trained_model_label
        first_row = QWidget()
        first_row_layout = QHBoxLayout(first_row)
        first_row_layout.setContentsMargins(0, 0, 0, 0)

        summary = QLabel(
            f"模型：{model_name}\n\n"
            f"训练样本：{result['train_samples']}\n\n"
            f"测试样本：{result['test_samples']}\n\n"
            f"训练准确率：{result['train_accuracy']:.2%}\n\n"
            f"测试准确率：{result['test_accuracy']:.2%}"
        )
        summary.setWordWrap(True)
        summary.setMinimumWidth(300)
        summary.setStyleSheet(
            "font-size: 16px; font-weight: bold; padding: 14px; "
            "background: white; border: 1px solid #ccd6e0;")
        first_row_layout.addWidget(summary, 1)
        first_row_layout.addWidget(self.create_confusion_canvas(result), 2)
        self.evaluation_layout.addWidget(first_row)

        importance_df = pd.read_csv(result["importance_path"])
        self.evaluation_layout.addWidget(self.create_importance_canvas(importance_df))

        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["编号", "完整指标名称", "重要性 (%)"])
        table.setRowCount(len(importance_df))
        for row, record in importance_df.reset_index(drop=True).iterrows():
            table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            table.setItem(row, 1, QTableWidgetItem(str(record["indicator"])))
            table.setItem(row, 2, QTableWidgetItem(f"{record['importance']:.4f}"))
        table.setMinimumHeight(360)
        table.horizontalHeader().setStretchLastSection(True)
        self.evaluation_layout.addWidget(table)
        self.evaluation_layout.addStretch()

    def create_confusion_canvas(self, result):
        figure = Figure(figsize=(8, 4.6), dpi=100)
        canvas = FigureCanvas(figure)
        axes = figure.add_subplot(111)
        matrix = result["confusion_matrix"]
        labels_by_value = {value: label for label, value in RESULT_VECTOR_MAPPING.items()}
        labels = [labels_by_value.get(value, str(value))
                  for value in result["class_labels"]]
        image = axes.imshow(matrix, cmap="Blues")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                axes.text(col, row, str(matrix[row, col]), ha="center", va="center")
        axes.set_xticks(range(len(labels)), labels)
        axes.set_yticks(range(len(labels)), labels)
        axes.set_xlabel("预测结果")
        axes.set_ylabel("实际结果")
        axes.set_title("混淆矩阵")
        figure.colorbar(image, ax=axes, fraction=0.03, pad=0.04)
        figure.tight_layout()
        canvas.setMinimumHeight(430)
        return canvas

    def create_importance_canvas(self, importance_df):
        ordered = importance_df.sort_values("importance")
        wrapped_labels = [
            self.wrap_indicator_label(label) for label in ordered["indicator"]
        ]
        max_lines = max(label.count("\n") + 1 for label in wrapped_labels)
        row_height = 0.34 + (max_lines - 1) * 0.11
        figure = Figure(
            figsize=(12, max(6, len(ordered) * row_height)), dpi=100)
        canvas = FigureCanvas(figure)
        axes = figure.add_subplot(111)
        bars = axes.barh(
            wrapped_labels,
            ordered["importance"],
            color="#3f72af",
            height=0.62,
        )
        axes.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8)
        axes.set_title("完整特征重要性")
        axes.set_xlabel("重要性 (%)")
        axes.set_ylabel("完整指标名称")
        axes.tick_params(axis="y", labelsize=8, length=0, pad=8)
        axes.margins(x=0.12)
        axes.grid(axis="x", linestyle="--", alpha=0.3)
        axes.set_axisbelow(True)
        figure.subplots_adjust(left=0.36, right=0.95, top=0.95, bottom=0.08)
        canvas.setMinimumHeight(max(600, int(len(ordered) * row_height * 100)))
        return canvas

    @staticmethod
    def wrap_indicator_label(label, width=18):
        text = str(label).strip()
        prefix = ""
        content = text
        if "." in text:
            possible_prefix, remainder = text.split(".", 1)
            if possible_prefix.isdigit():
                prefix = possible_prefix + "."
                content = remainder

        lines = textwrap.wrap(
            content,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [content]
        lines[0] = prefix + lines[0]
        return "\n".join(lines)

    def show_indicator_table(self, dataframe):
        self.indicator_table.clear()
        self.indicator_table.setRowCount(len(dataframe))
        self.indicator_table.setColumnCount(len(dataframe.columns))
        self.indicator_table.setHorizontalHeaderLabels(
            [str(column) for column in dataframe.columns])
        for row in range(len(dataframe)):
            for col, column in enumerate(dataframe.columns):
                self.indicator_table.setItem(
                    row, col, QTableWidgetItem(str(dataframe.iloc[row, col])))
        self.indicator_table.resizeColumnsToContents()

    @staticmethod
    def stylesheet():
        return """
            QWidget { background-color: #eaf0f6; font-family: 'Microsoft YaHei'; font-size: 10pt; }
            QPushButton { background-color: #3f72af; color: white; border: none; padding: 7px 12px; border-radius: 4px; }
            QPushButton:hover { background-color: #2b5d8c; }
            QPushButton:disabled { background-color: #b8c2cc; }
            QGroupBox { border: 1px solid #c8d0d8; border-radius: 6px; margin-top: 10px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QLineEdit, QComboBox { background: white; border: 1px solid #c8d0d8; border-radius: 4px; padding: 5px; }
            QTableWidget { background: white; gridline-color: #dfe5eb; }
            QHeaderView::section { background: #f3f6f9; padding: 6px; font-weight: bold; }
        """
