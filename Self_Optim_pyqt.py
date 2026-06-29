import sys
import os
import shutil

# 必须在导入 matplotlib 或使用 matplotlib 的模块之前设置后端
# Agg 是非交互式后端，适合在后台线程中生成图像，避免 GUI 冲突导致崩溃
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，线程安全

import pandas as pd
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QFileDialog,
    QComboBox,
    QGroupBox,
    QGridLayout,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QMessageBox,
    QTextEdit,
    QScrollArea,
    QSplitter,
)
from PyQt5.QtGui import QFont, QBrush, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QCoreApplication, QRunnable, QThreadPool, QObject, QMutex
from PyQt5.QtGui import QPixmap
import threading
from params import RRDA_AI_indicater_1
from selfOptimization import (  # 替换为您的实际模块名
    self_optimization,
    encode,
    read_csv_auto_encoding,
)
from util import get_AI_AII, sort_indicators_by_number, extract_indicator_number
from params import RESULT_VECTOR_MAPPING
import RRDA
import pandas as pd


class SortableTableWidgetItem(QTableWidgetItem):
    """支持数值排序的表格项"""
    def __lt__(self, other):
        try:
            # 尝试转换为浮点数进行比较
            return float(self.text()) < float(other.text())
        except ValueError:
            # 如果转换失败，回退到默认的字符串比较
            return super().__lt__(other)


# 工作线程信号类（因为 QRunnable 不能直接发送信号）
class WorkerSignals(QObject):
    """定义工作线程可以发出的信号"""
    update_progress = pyqtSignal(int, float, float, float, float, float, float, dict, object, object, object, object, dict)
    training_complete = pyqtSignal(object, str, list, list, list)
    error = pyqtSignal(str)
    finished = pyqtSignal()


# 算法运行器（使用 QRunnable 而不是 QThread）
class AlgorithmRunner(QRunnable):
    """使用 QRunnable + QThreadPool 实现的训练任务，比 QThread 更轻量且易于管理"""

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.signals = WorkerSignals()
        self.running = True
        self.epochs = params.get('epochs', 10)
        self.losses = []
        self.train_accs = []
        self.A_II_labels = []
        # 启用自动删除（任务完成后自动清理）
        self.setAutoDelete(True)
        # 绘图锁，确保 matplotlib 调用的线程安全（即使使用 Agg 后端也是好习惯）
        self._plot_lock = threading.Lock()

    def run(self):
        try:
            # 准备参数
            old_zbtx_file = self.params['old_zbtx_file']
            dataset_file = self.params['dataset']

            # 读取数据
            real_result = read_csv_auto_encoding(dataset_file)['实际认定结果']
            real_labels = real_result.map(RESULT_VECTOR_MAPPING).to_numpy()

            # 获取二级指标列表
            df_zbtx = read_csv_auto_encoding(old_zbtx_file)
            # A_I, A_II = get_AI_AII(df_zbtx, RRDA_AI_indicater_1)  # 假设RRDA_AI_indicater_1已定义
            A_I, A_II = get_AI_AII(df_zbtx)
            A_II_labels = []
            for ai in A_II.keys():
                for aii in A_II[ai].keys():
                    A_II_labels.append(aii)
            
            # 按照指标名称前面的数字排序
            A_II_labels = sort_indicators_by_number(A_II_labels)

            # 编码数据
            encode_data = encode(
                dataset_file,
                df_zbtx=read_csv_auto_encoding(old_zbtx_file),
                with_RRDA=False,
            )
            for i in range(len(encode_data)):
                encode_data[i] = {
                    k: v
                    for k, v in sorted(encode_data[i].items(),
                                       key=lambda item: A_II_labels.index(item[0]))
                }
            df = pd.DataFrame(encode_data)

            # 执行RRDA算法
            # full_df_std = RRDA.run_RRDA(dataset_file)
            full_df_std = read_csv_auto_encoding(dataset_file)

            # 准备real_data
            real_data = [df, real_labels]

            # 自定义训练回调函数
            def training_callback(epoch, loss, train_acc, test_acc, recall, AII_acc, FCEAII_acc, weights,
                                  loss_img, acc_img, acc_aii_fce_img, weight_img, color_mapping):
                if not self.running:
                    return False  # 停止训练

                try:
                    # 保存训练指标
                    self.losses.append(loss)
                    self.train_accs.append(train_acc)

                    # 发送更新信号（只在特定 epoch 发送以减少 UI 负担）
                    if epoch % 2 == 0 or epoch == self.epochs:
                        # 使用锁保护信号发送
                        with self._plot_lock:
                            self.signals.update_progress.emit(
                                epoch,
                                loss,
                                train_acc,
                                test_acc,
                                recall,
                                AII_acc,
                                FCEAII_acc,
                                weights,
                                loss_img,
                                acc_img,
                                acc_aii_fce_img,
                                weight_img,
                                color_mapping
                            )
                except Exception as e:
                    print(f"回调函数中发生错误: {e}")
                    import traceback
                    traceback.print_exc()
                
                return True  # 继续训练

            # 调用自优化算法
            optimized_df = self_optimization(
                full_df_std,
                real_data,
                self.params['old_zbtx_file'],
                self.params['new_zbtx_path'],
                None,
                self.params['lr'],
                self.params['epochs'],
                self.params['batch_size'],
                self.params['alpha'],
                self.params['beta'],
                A_I,
                A_II,
                A_II_labels,
                self.params['device'],
                recall_weight=self.params['recall_weight'],
                callback=training_callback,  # 添加回调函数
                work_dir=self.params['new_zbtx_path'],
            )

            # 发送训练完成信号，直接传递DataFrame
            self.signals.training_complete.emit(
                optimized_df,
                "训练完成！",
                self.losses,
                self.train_accs,
                []
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.signals.error.emit(f"训练出错: {str(e)}")
            self.signals.training_complete.emit(None, f"训练出错: {str(e)}", [], [], [])
        finally:
            # 确保任务完成时发出 finished 信号
            self.signals.finished.emit()

    def stop(self):
        self.running = False


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("指标体系优化工具")
        self.setGeometry(100, 100, 1200, 800)

        # 初始化参数
        self.params = {
            'old_zbtx_file': './tmp/整合版量化指标体系.csv',
            'new_zbtx_path': './tmp',
            'dataset': './tmp/新模拟数据.csv',
            'lr': 0.0005,
            'epochs': 500,
            'batch_size': 256,
            'alpha': 0.4,
            'beta': 0.,
            'recall_weight': 0.3,
            'device': 'cpu',  # 默认使用CPU
            # 'device': 'cuda:0'
        }

        # 线程池管理（使用全局线程池）
        self.thread_pool = QThreadPool.globalInstance()
        self.current_worker = None  # 当前运行的 worker
        self.optimized_df = None
        self.color_mapping = {}  # 存储颜色映射
        print(f"多线程配置: 最大线程数 = {self.thread_pool.maxThreadCount()}")

        # 创建主控件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 主布局
        main_layout = QHBoxLayout(self.central_widget)  # 改为水平布局

        # 创建左侧控制区（参数+进度）
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(10)  # 固定子栏目间距为10像素

        # 创建参数区域
        self.create_parameter_section(left_layout)

        # 进度条区域
        progress_group = QGroupBox("训练进度")
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setContentsMargins(10, 20, 10, 10)  # 增加内边距
        progress_layout.setSpacing(8)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        # 状态标签 (从详细信息栏移动过来)
        self.loss_label = QLabel("损失: -")
        self.loss_label.setStyleSheet("font-size: 18px; font-family: 'Microsoft YaHei';")
        self.acc_label = QLabel("训练/测试一致率: -")
        self.acc_label.setStyleSheet("font-size: 18px; font-family: 'Microsoft YaHei';")
        self.recall_label = QLabel("宏平均召回率: -")
        self.recall_label.setStyleSheet("font-size: 18px; font-family: 'Microsoft YaHei';")
        self.aii_label = QLabel("AII/FCEAII一致率: -")
        self.aii_label.setStyleSheet("font-size: 18px; font-family: 'Microsoft YaHei';")

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.loss_label)
        progress_layout.addWidget(self.acc_label)
        progress_layout.addWidget(self.recall_label)
        progress_layout.addWidget(self.aii_label)
        
        left_layout.addWidget(progress_group)
        
        # 添加颜色图例区域（与其他组同级）
        color_legend_group = QGroupBox("指标颜色对照")
        color_legend_group_layout = QVBoxLayout(color_legend_group)
        color_legend_group_layout.setSpacing(5)
        color_legend_group_layout.setContentsMargins(10, 20, 10, 10)  # 增加顶部边距，避免遮盖标题
        
        # 使用滚动区域显示颜色图例
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)  # 移除边框，更简洁
        
        self.color_legend_widget = QWidget()
        self.color_legend_layout = QVBoxLayout(self.color_legend_widget)
        self.color_legend_layout.setSpacing(5)  # 与详细信息保持一致的间距
        self.color_legend_layout.setContentsMargins(5, 5, 5, 5)
        
        scroll_area.setWidget(self.color_legend_widget)
        color_legend_group_layout.addWidget(scroll_area)
        
        left_layout.addWidget(color_legend_group)

        # 按钮区域
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("开始训练")
        self.run_button.clicked.connect(self.start_training)

        self.stop_button = QPushButton("停止训练")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)

        self.save_button = QPushButton("保存结果")
        # self.save_button.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)

        self.export_indicator_button = QPushButton("导出指标体系")
        self.export_indicator_button.clicked.connect(self.export_indicator_csv)
        self.export_indicator_button.setEnabled(False)

        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.export_indicator_button)
        left_layout.addLayout(button_layout)

        # 添加左侧控制区到主布局
        main_layout.addWidget(left_widget, 1)  # 左侧占1份空间

        # 创建右侧可视化区域（图表+表格）
        tab_widget = QTabWidget()  # 改名为 tab_widget 避免冲突

        # 图表标签页
        charts_tab = QWidget()
        charts_layout = QHBoxLayout(charts_tab)  # 水平布局

        # 中间列 - 纵向排列损失和一致率图
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)

        # center_layout.addWidget(QLabel("损失变化"))
        self.loss_canvas = self.create_image_widget()
        center_layout.addWidget(self.loss_canvas)

        # center_layout.addWidget(QLabel("一致率变化"))
        self.acc_canvas = self.create_image_widget()
        center_layout.addWidget(self.acc_canvas)

        self.aii_fceaii_canvas = self.create_image_widget()
        center_layout.addWidget(self.aii_fceaii_canvas)

        # center_layout.addStretch()  # 添加弹性空间使内容居中
        charts_layout.addWidget(center_widget, 1)  # 中间占2份空间

        # 右侧 - 指标权重变化图
        weight_widget = QWidget()  # 改名为 weight_widget 避免冲突
        weight_layout = QVBoxLayout(weight_widget)
        weight_layout.setContentsMargins(0, 0, 0, 0)

        weight_layout.addWidget(QLabel("指标权重变化"))
        self.weight_canvas = self.create_image_widget()
        weight_layout.addWidget(self.weight_canvas, 1)  # 占据所有可用空间

        charts_layout.addWidget(weight_widget, 3)  # 右侧占3份空间（更大的比例）

        # 表格标签页
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        
        # 创建分割器
        splitter = QSplitter(Qt.Vertical)
        
        # 表格部分
        self.result_table = QTableWidget()
        self.result_table.horizontalHeader().setStretchLastSection(True)
        # 关闭交替行背景，使用纯白背景以避免与颜色梯度叠加干扰
        self.result_table.setAlternatingRowColors(False)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        # 确保表格和单元格使用纯白背景
        # self.result_table.setStyleSheet("background-color: white;\nalternate-background-color: white;\nQTableWidget::item { background-color: white; }")
        splitter.addWidget(self.result_table)

        # 总结部分容器
        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        
        self.summary_label = QLabel("优化结果总结")
        self.summary_label.setStyleSheet("font-weight: bold; font-size: 18px;")
        summary_layout.addWidget(self.summary_label)

        self.summary_text = QTextEdit()
        self.summary_text.setMinimumHeight(100)
        self.summary_text.setFont(QFont("Microsoft YaHei", 12))
        self.summary_text.setReadOnly(True)
        self.summary_text.setPlaceholderText("优化结果将在训练完成后显示")
        summary_layout.addWidget(self.summary_text)
        
        splitter.addWidget(summary_widget)
        
        # 设置分割器初始比例
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 2)
        
        table_layout.addWidget(splitter)

        # 将标签页添加到选项卡控件
        tab_widget.addTab(charts_tab, "自优化可视化展板")
        tab_widget.addTab(table_tab, "优化结果分析")

        # 添加右侧可视化区到主布局
        main_layout.addWidget(tab_widget, 3)  # 右侧占3份空间（更大的比例）

        # splitter.addWidget(control_widget)
        # splitter.addWidget(viz_widget)
        # splitter.setSizes([300, 900])

        # 状态栏
        self.statusBar().showMessage("就绪")

        # 设置样式
        self.setStyleSheet("""
            QWidget {
                font-family: 'Microsoft YaHei';
                font-size: 10pt;
                background-color: #eaf0f6;
            }
            QLabel { color: #1f2d3d; }
            QPushButton {
                background-color: #3f72af; color: white; border: none;
                padding: 6px 12px; border-radius: 4px;
            }
            QPushButton:hover { background-color: #2b5d8c; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
            QGroupBox {
                border: 1px solid #ccc; border-radius: 6px;
                margin-top: 10px; font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 3px;
            }
            QTableWidget {
                background-color: white; border: 1px solid #ccc;
                gridline-color: #e0e0e0;
            }
            QHeaderView::section {
                background-color: #f3f6f9; padding: 6px; font-weight: bold;
                border: 1px solid #ccc;
            }
            QLineEdit, QComboBox, QTextEdit {
                padding: 5px; border: 1px solid #ccc; border-radius: 4px; background-color: white;
            }
            QProgressBar {
                border: 1px solid #ccc; border-radius: 4px; text-align: center; background-color: white;
            }
            QProgressBar::chunk {
                background-color: #3f72af;
            }
            QScrollArea {
                border: none; background-color: transparent;
            }
        """)

    def create_parameter_section(self, layout):
        """创建参数输入区域"""
        # 基本参数组
        basic_group = QGroupBox("基本设置")
        basic_layout = QVBoxLayout(basic_group)
        basic_layout.setContentsMargins(10, 20, 10, 10) # 增加内边距
        basic_layout.setSpacing(5)

        # 存储文件标签以便更新
        self.file_labels = {}

        # 1. 原始指标体系
        h_zbtx = QHBoxLayout()
        h_zbtx.addWidget(QLabel("原始指标体系:"))
        h_zbtx.addStretch()
        btn_zbtx = QPushButton("浏览...")
        btn_zbtx.setProperty('param', 'old_zbtx_file')
        btn_zbtx.clicked.connect(lambda: self.browse_file("open"))
        h_zbtx.addWidget(btn_zbtx)
        basic_layout.addLayout(h_zbtx)

        zbtx_name = os.path.basename(self.params['old_zbtx_file']) if self.params['old_zbtx_file'] else "未选择文件"
        self.label_zbtx_name = QLabel(zbtx_name)
        self.label_zbtx_name.setStyleSheet("font-size: 8pt; margin-bottom: 5px;")
        self.label_zbtx_name.setToolTip(self.params['old_zbtx_file'])
        basic_layout.addWidget(self.label_zbtx_name)
        self.file_labels['old_zbtx_file'] = self.label_zbtx_name

        # 2. 参与认定的学生数据
        h_dataset = QHBoxLayout()
        h_dataset.addWidget(QLabel("参与认定的学生数据:"))
        h_dataset.addStretch()
        btn_dataset = QPushButton("浏览...")
        btn_dataset.setProperty('param', 'dataset')
        btn_dataset.clicked.connect(lambda: self.browse_file("open"))
        h_dataset.addWidget(btn_dataset)
        basic_layout.addLayout(h_dataset)

        dataset_name = os.path.basename(self.params['dataset']) if self.params['dataset'] else "未选择文件"
        self.label_dataset_name = QLabel(dataset_name)
        self.label_dataset_name.setStyleSheet("font-size: 8pt; margin-bottom: 5px;")
        self.label_dataset_name.setToolTip(self.params['dataset'])
        basic_layout.addWidget(self.label_dataset_name)
        self.file_labels['dataset'] = self.label_dataset_name

        layout.addWidget(basic_group)

        # 创建高级设置组（可折叠）
        advanced_group = QGroupBox("高级设置")
        advanced_group.setCheckable(True)  # 设置为可折叠
        advanced_group.setChecked(False)  # 默认折叠
        advanced_layout = QGridLayout(advanced_group)
        advanced_layout.setContentsMargins(10, 20, 10, 10) # 增加内边距
        advanced_layout.setVerticalSpacing(8)  # 固定行间距
        advanced_layout.setHorizontalSpacing(5)  # 固定列间距

        # 高级参数标签
        advanced_labels = [
            "学习率(lr):", "训练轮数(epochs):", "批次大小(batch_size):", "AII权重(alpha):",
            "FCE权重(beta):", "召回率损失权重:", "训练设备(device):"
        ]

        # 高级参数对应的键
        advanced_keys = [
            'lr', 'epochs', 'batch_size', 'alpha', 'beta', 'recall_weight', 'device'
        ]
        self.param_edits = {}

        for i, label in enumerate(advanced_labels):
            advanced_layout.addWidget(QLabel(label), i, 0)
            param_name = advanced_keys[i]

            if param_name == 'device':
                combo = QComboBox()
                combo.addItems(['cpu', 'cuda'])
                combo.setCurrentText(self.params[param_name])
                advanced_layout.addWidget(combo, i, 1)
                self.param_edits[param_name] = combo
            else:
                edit = QLineEdit(str(self.params[param_name]))
                advanced_layout.addWidget(edit, i, 1)
                self.param_edits[param_name] = edit

        layout.addWidget(advanced_group)

    def create_image_widget(self):
        """创建用于显示图像的QLabel"""
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(300, 200)
        label.setStyleSheet(
            "background-color: white; border: 1px solid #c0c0c0;")
        return label

    def browse_file(self, dialog_type):
        """浏览文件或目录按钮点击事件"""
        btn = self.sender()
        param_name = btn.property('param')

        if dialog_type == "open":
            file_path, _ = QFileDialog.getOpenFileName(self, f"选择{param_name}")
            if file_path:
                self.params[param_name] = file_path
                if param_name in self.file_labels:
                    self.file_labels[param_name].setText(os.path.basename(file_path))
                    self.file_labels[param_name].setToolTip(file_path)

        elif dialog_type == "save" and param_name == "new_zbtx_path":
            # 使用保存对话框选择目录
            dir_path = QFileDialog.getExistingDirectory(self, "选择指标体系保存目录")
            if dir_path:
                self.params[param_name] = dir_path

    def save_results(self):
        """保存结果到指定文件夹"""
        try:
            # 选择保存目录
            dir_path = QFileDialog.getExistingDirectory(self, "选择保存结果的文件夹")
            if not dir_path:
                return

            # 1. 保存优化后的指标体系CSV
            if hasattr(self, 'optimized_df') and self.optimized_df is not None:
                save_path = os.path.join(dir_path, "optimized_indicators.csv")
                indicator_df = self.build_optimized_indicator_system()
                indicator_df.to_csv(save_path, index=False, encoding='utf-8-sig')
                summary_path = os.path.join(dir_path, "optimized_weight_summary.csv")
                self.optimized_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
            
            # 2. 保存四张图表（使用存储的高分辨率原始数据）
            images_to_save = {
                "loss_chart.png": getattr(self, 'current_loss_img_data', None),
                "acc_chart.png": getattr(self, 'current_acc_img_data', None),
                "aii_fceaii_chart.png": getattr(self, 'current_aii_fceaii_img_data', None),
                "weight_chart.png": getattr(self, 'current_weight_img_data', None)
            }
            
            for filename, img_data in images_to_save.items():
                if img_data:
                    try:
                        with open(os.path.join(dir_path, filename), 'wb') as f:
                            f.write(img_data)
                    except Exception as e:
                        print(f"保存图片 {filename} 失败: {e}")
            
            # 3. 保存总结文本
            summary = self.summary_text.toPlainText()
            with open(os.path.join(dir_path, "result_summary.txt"), 'w', encoding='utf-8') as f:
                f.write(summary)
                
            QMessageBox.information(self, "成功", f"结果已保存到: {dir_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存结果失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def export_indicator_csv(self):
        """直接导出优化后的指标体系CSV。"""
        if self.optimized_df is None or self.optimized_df.empty:
            QMessageBox.warning(self, "提示", "请先完成指标体系优化。")
            return

        source_path = self.params.get('old_zbtx_file') or "optimized_indicators.csv"
        source_dir = os.path.dirname(source_path)
        source_stem = os.path.splitext(os.path.basename(source_path))[0]
        default_name = os.path.join(source_dir, f"{source_stem}_自优化指标体系.csv")

        path, _ = QFileDialog.getSaveFileName(
            self, "导出指标体系", default_name, "CSV Files (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"

        try:
            indicator_df = self.build_optimized_indicator_system()
            indicator_df.to_csv(path, index=False, encoding='utf-8-sig')
            self.statusBar().showMessage(f"指标体系CSV已导出: {path}")
            QMessageBox.information(self, "导出成功", f"指标体系CSV已导出至:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def build_optimized_indicator_system(self):
        """生成与第四页一致的完整指标体系CSV结构。"""
        if self.optimized_df is None or self.optimized_df.empty:
            raise ValueError("请先完成指标体系优化。")

        old_zbtx_file = self.params.get('old_zbtx_file')
        if not old_zbtx_file or not os.path.exists(old_zbtx_file):
            raise ValueError("原始指标体系文件不存在，请重新选择。")

        indicator_df = pd.read_csv(old_zbtx_file)
        required_cols = {"indicator_2", "score_max", "normalized_score"}
        missing_cols = required_cols - set(indicator_df.columns)
        if missing_cols:
            raise ValueError(f"原始指标体系缺少列: {', '.join(sorted(missing_cols))}")

        name_col = self.find_result_column(["指标名称", "指标", "名称"], default_index=0)
        weight_col = self.find_result_column(["优化后权重"], default_index=None)
        if weight_col is None:
            raise ValueError("优化结果中缺少“优化后权重”列。")

        weight_df = self.optimized_df[[name_col, weight_col]].dropna(subset=[name_col])
        weight_map = (
            weight_df.assign(**{
                name_col: weight_df[name_col].astype(str).str.strip(),
                weight_col: pd.to_numeric(weight_df[weight_col], errors='coerce') * 100,
            })
            .dropna(subset=[weight_col])
            .set_index(name_col)[weight_col]
            .to_dict()
        )

        output_df = indicator_df.copy()
        indicator_names = output_df["indicator_2"].astype(str).str.strip()
        output_df["score_max"] = indicator_names.map(weight_map).fillna(output_df["score_max"])
        output_df["score"] = pd.to_numeric(output_df["normalized_score"], errors='coerce') * pd.to_numeric(
            output_df["score_max"], errors='coerce')
        return output_df

    def find_result_column(self, substrings, default_index=None):
        """按列名关键词查找优化结果表字段。"""
        for col in self.optimized_df.columns:
            col_text = str(col)
            if any(text in col_text for text in substrings):
                return col
        if default_index is not None and len(self.optimized_df.columns) > default_index:
            return self.optimized_df.columns[default_index]
        return None

    def start_training(self):
        """开始训练按钮点击事件"""
        # 更新参数
        # for key, edit in self.file_edits.items():
        #     self.params[key] = edit.text()

        for key, edit in self.param_edits.items():
            if isinstance(edit, QComboBox):
                self.params[key] = edit.currentText()
            else:
                try:
                    self.params[key] = float(edit.text(
                    )) if key != 'epochs' and key != 'batch_size' else int(
                        edit.text())
                except ValueError:
                    QMessageBox.warning(self, "输入错误", f"参数 {key} 输入无效，请检查！")
                    return

        if not 0 <= self.params['recall_weight'] <= 1:
            QMessageBox.warning(self, "输入错误", "召回率损失权重必须在 0 和 1 之间！")
            return
        if (self.params['alpha'] < 0 or self.params['beta'] < 0
                or self.params['alpha'] + self.params['beta'] > 1):
            QMessageBox.warning(
                self, "输入错误", "AII权重和FCE权重必须非负，并且两者之和不能超过 1！"
            )
            return

        # 重置UI状态
        self.progress_bar.setValue(0)
        self.loss_label.setText("当前损失: -")
        self.acc_label.setText("当前一致率: -")
        self.recall_label.setText("宏平均召回率: -")
        self.optimized_df = None
        self.save_button.setEnabled(False)
        self.export_indicator_button.setEnabled(False)
        self.statusBar().showMessage("训练中...")

        # 清空表格
        self.result_table.setRowCount(0)

        # 禁用开始按钮，启用停止按钮
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        # 创建并启动训练任务（使用 QRunnable + QThreadPool）
        self.current_worker = AlgorithmRunner(self.params)
        
        # 连接信号（使用 QueuedConnection 确保跨线程安全）
        self.current_worker.signals.update_progress.connect(
            self.update_training_progress, Qt.QueuedConnection)
        self.current_worker.signals.training_complete.connect(
            self.training_finished, Qt.QueuedConnection)
        self.current_worker.signals.error.connect(
            self.on_training_error, Qt.QueuedConnection)
        self.current_worker.signals.finished.connect(
            self.on_thread_finished, Qt.QueuedConnection)
        
        # 提交任务到线程池
        self.thread_pool.start(self.current_worker)
        print("训练任务已提交到线程池")

    def stop_training(self):
        """停止训练按钮点击事件"""
        if self.current_worker:
            self.current_worker.stop()
            self.statusBar().showMessage("正在停止训练...")
            # QThreadPool 会在任务完成时自动清理
            # 不需要像 QThread 那样调用 wait()

    def update_training_progress(self, epoch, loss, train_acc, test_acc,
                                 recall, AII_acc, FCEAII_acc, weights,
                                 loss_img_data, acc_img_data, aii_fceaii_img_data, weight_img_data, color_mapping):
        """更新训练进度"""
        total_epochs = self.params['epochs']
        progress = int((epoch / total_epochs) * 100)

        # 更新进度条
        self.progress_bar.setValue(progress)

        # 更新状态标签
        self.loss_label.setText(f"当前损失: {loss:.4f}")
        self.acc_label.setText(
            f"模型一致率: {train_acc:.2%}")
        self.recall_label.setText(
            f"宏平均召回率: {recall:.2%}")
        self.aii_label.setText(
            f"AII一致率: {AII_acc:.2%} | FCEAII一致率: {FCEAII_acc:.2%}")

        # 保存颜色映射
        self.color_mapping = color_mapping
        
        # 保存当前的高分辨率图像数据供保存使用
        self.current_loss_img_data = loss_img_data
        self.current_acc_img_data = acc_img_data
        self.current_aii_fceaii_img_data = aii_fceaii_img_data
        self.current_weight_img_data = weight_img_data
        
        # 更新颜色图例（只在第一次更新时或颜色映射改变时）
        if color_mapping:
            self.update_color_legend(color_mapping)

        # 更新图表
        self.update_chart(self.loss_canvas, loss_img_data)
        self.update_chart(self.acc_canvas, acc_img_data)
        self.update_chart(self.aii_fceaii_canvas, aii_fceaii_img_data)
        self.update_chart(self.weight_canvas, weight_img_data)

        # 更新状态栏
        self.statusBar().showMessage(f"训练中... Epoch {epoch}/{total_epochs}")

    def training_finished(self, optimized_df, message, losses, train_accs,
                          test_accs):
        """训练完成处理"""
        # 更新UI状态
        self.progress_bar.setValue(100)
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.statusBar().showMessage(message)
        
        # 保存优化后的DataFrame供后续使用
        self.optimized_df = optimized_df
        has_result = self.optimized_df is not None and not self.optimized_df.empty
        self.save_button.setEnabled(has_result)
        self.export_indicator_button.setEnabled(has_result)

        # 加载优化结果到表格
        if has_result:
            self.load_results_to_table(self.optimized_df)

        # 保存训练指标
        self.final_losses = losses
        self.final_train_accs = train_accs

    def update_chart(self, canvas, image_data):
        """更新单个图表显示"""
        if image_data is None:
            return
            
        try:
            # 验证图像数据
            if not isinstance(image_data, bytes) or len(image_data) == 0:
                print(f"警告: 图像数据无效或为空")
                return
                
            pixmap = QPixmap()
            if pixmap.loadFromData(image_data):
                if not pixmap.isNull():
                    # 使用 scaled 方法调整图像大小以适应画布
                    scaled_pixmap = pixmap.scaled(
                        canvas.width(),
                        canvas.height(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    canvas.setPixmap(scaled_pixmap)
                else:
                    print(f"警告: QPixmap 为空")
            else:
                print(f"警告: 无法从数据加载 QPixmap")
        except Exception as e:
            print(f"错误: 更新图表时发生异常: {e}")
            import traceback
            traceback.print_exc()

    def update_color_legend(self, color_mapping):
        """更新颜色图例显示（在训练进度栏下方）"""
        try:
            # 清空现有图例
            while self.color_legend_layout.count():
                child = self.color_legend_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            
            # 按照指标名称前面的数字排序
            sorted_items = sorted(color_mapping.items(), key=lambda x: extract_indicator_number(x[0]))
            
            # 为每个指标创建一行显示
            for indicator, color in sorted_items:
                item_widget = QWidget()
                item_layout = QHBoxLayout(item_widget)
                item_layout.setContentsMargins(0, 0, 0, 0)
                item_layout.setSpacing(8)  # 与其他组件间距保持一致
                
                # 创建颜色方块
                color_label = QLabel()
                color_label.setFixedSize(20, 15)
                color_label.setStyleSheet(f"background-color: {color}; border: 1px solid #000;")
                
                # 创建指标名称标签
                text_label = QLabel(indicator)
                text_label.setStyleSheet("font-size: 10pt; font-family: 'Microsoft YaHei';")
                
                item_layout.addWidget(color_label)
                item_layout.addWidget(text_label)
                item_layout.addStretch()
                
                self.color_legend_layout.addWidget(item_widget)
            
            # 添加弹性空间
            self.color_legend_layout.addStretch()
            
        except Exception as e:
            print(f"更新颜色图例时出错: {e}")
            import traceback
            traceback.print_exc()

    def load_results_to_table(self, df):
        """加载优化结果到表格"""
        try:
            if df is not None and not df.empty:
                # 设置表格行列数
                self.result_table.setRowCount(len(df))
                self.result_table.setColumnCount(len(df.columns))

                # 设置表头
                headers = df.columns.tolist()
                print("表头:", headers)
                self.result_table.setHorizontalHeaderLabels(headers)
                
                # 识别关键列索引
                pct_col_idx = -1
                change_col_idx = -1
                trend_col_idx = -1
                
                for i, col in enumerate(headers):
                    if "百分比" in col:
                        pct_col_idx = i
                    if "变化量" in col:
                        change_col_idx = i
                    if "趋势" in col:
                        trend_col_idx = i
                
                # 获取最大百分比用于归一化
                max_pct = 0
                if pct_col_idx != -1:
                    try:
                        # 确保数据是数值型
                        max_pct = pd.to_numeric(df.iloc[:, pct_col_idx], errors='coerce').abs().max()
                    except:
                        pass

                # 填充表格数据
                self.result_table.setSortingEnabled(False)  # 填充数据前关闭排序
                for row in range(len(df)):
                    for col in range(len(df.columns)):
                        value = df.iloc[row, col]
                        # 使用自定义的 SortableTableWidgetItem
                        item = SortableTableWidgetItem(str(value))
                        item.setTextAlignment(Qt.AlignCenter)

                        # 颜色梯度处理：对权重变化量和变化百分比列应用颜色
                        if (col == change_col_idx or col == pct_col_idx) and trend_col_idx != -1 and pct_col_idx != -1:
                            try:
                                # 获取趋势和百分比
                                trend = df.iloc[row, trend_col_idx]
                                pct_val = df.iloc[row, pct_col_idx]
                                pct = float(pct_val) if pd.notnull(pct_val) else 0
                                
                                # 计算颜色强度 (0-1)
                                ratio = 0
                                if max_pct > 0:
                                    ratio = abs(pct) / max_pct
                                
                                # 限制 ratio 在 0-1 之间
                                ratio = min(max(ratio, 0), 1)
                                
                                # 使用 Alpha 通道实现深浅变化，这样可以保留表格的交替行背景色
                                # 最小 Alpha 设为 0 (无色)，最大设为 200 (避免文字看不清)
                                alpha = int(ratio * 200)
                                
                                if trend == '上升':
                                    # 红色
                                    item.setBackground(QBrush(QColor(255, 0, 0, alpha)))
                                elif trend == '下降':
                                    # 蓝色
                                    item.setBackground(QBrush(QColor(0, 0, 255, alpha)))
                                
                                # 如果背景色较深（ratio大），将文字设为白色以保证对比度
                                # 这里阈值设高一点，因为我们最大alpha只有200
                                if ratio > 0.6:
                                    item.setForeground(QBrush(QColor(255, 255, 255))) # 白字
                                else:
                                    item.setForeground(QBrush(QColor(0, 0, 0))) # 黑字
                                    
                            except Exception as e:
                                # print(f"设置颜色出错: {e}")
                                pass

                        # 关键修改：将item添加到表格中
                        self.result_table.setItem(row, col, item)

                # 调整列宽
                self.result_table.resizeColumnsToContents()
                # 调整行高
                self.result_table.resizeRowsToContents()
                
                self.result_table.setSortingEnabled(True)  # 填充完成后开启排序

                # 更新总结文本
                summary = self.get_summary_text(df)
                self.summary_text.setText(summary)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载结果失败: {str(e)}")

    def get_summary_text(self, df):
        """生成优化结果总结文本"""
        if df.empty:
            return "无优化结果数据"

        # 尝试识别常用列名（兼容不同CSV导出格式）
        def find_col(substrings):
            for col in df.columns:
                for s in substrings:
                    if s in col:
                        return col
            return None

        name_col = find_col(['指标名称', '指标', '名称']) or df.columns[0]
        change_col = find_col(['变化量', '权重变化量'])
        pct_col = find_col(['百分比', '%', '变化百分比'])
        trend_col = find_col(['趋势', '权重变化趋势'])

        # 统计上升/下降（若没有趋势列则跳过统计）
        total_weight_change = {}
        if trend_col and trend_col in df.columns:
            try:
                total_weight_change = df[trend_col].value_counts().to_dict()
            except Exception:
                total_weight_change = {}

        up_indicators = []
        down_indicators = []
        if trend_col and trend_col in df.columns and name_col in df.columns:
            try:
                up_indicators = df[df[trend_col] == '上升'][name_col].astype(str).tolist()
                down_indicators = df[df[trend_col] == '下降'][name_col].astype(str).tolist()
            except Exception:
                up_indicators = []
                down_indicators = []

        format_up_indicators = '\n      '.join(up_indicators)
        format_down_indicators = '\n      '.join(down_indicators)

        summary_lines = [
            f"指标体系包含 {len(df)} 个指标。",
            f"总权重变化：\n{total_weight_change.get('上升', 0)} 个指标权重上升:",
            f"      {format_up_indicators}",
            f"\n{total_weight_change.get('下降', 0)} 个指标权重下降:",
            f"      {format_down_indicators}",
        ]

        # ----- 联合判定法：基于变化量与变化百分比同时超过阈值判定显著变化 -----
        try:
            if change_col and pct_col and change_col in df.columns and pct_col in df.columns:
                # 转换为数值并取绝对值（关注变化幅度）
                change_vals = pd.to_numeric(df[change_col], errors='coerce').fillna(0).astype(float).abs()
                pct_vals = pd.to_numeric(df[pct_col], errors='coerce').fillna(0).astype(float).abs()

                # 计算阈值：均值 + 1 个标准差
                change_mean = change_vals.mean()
                change_std = change_vals.std()
                pct_mean = pct_vals.mean()
                pct_std = pct_vals.std()

                change_thresh = float(change_mean + change_std) if not pd.isna(change_mean) else 0.0
                pct_thresh = float(pct_mean + pct_std) if not pd.isna(pct_mean) else 0.0

                # 符合两个阈值的指标为显著变化
                mask = (change_vals > change_thresh) | (pct_vals > pct_thresh)

                sig_df = df.loc[mask]

                # 准备阐述计算过程和结果
                summary_lines.append('\n显著变化判定（联合判定法）：')
                summary_lines.append(f"- 变化量阈值 = 均值({change_mean:.4f}) + 标准差({change_std:.4f}) = {change_thresh:.4f}")
                summary_lines.append(f"- 变化百分比阈值 = 均值({pct_mean:.4f}) + 标准差({pct_std:.4f}) = {pct_thresh:.4f}")

                if sig_df is None or sig_df.empty:
                    summary_lines.append("- 未检测到同时超过两项阈值的显著变化指标。")
                else:
                    summary_lines.append(f"- 检测到 {len(sig_df)} 个显著变化指标：")
                    # 列出每个显著指标的关键信息
                    for _, row in sig_df.iterrows():
                        name = str(row.get(name_col, ''))
                        ch = row.get(change_col, '')
                        pc = row.get(pct_col, '')
                        trend = row.get(trend_col, '') if trend_col in df.columns else ''
                        summary_lines.append(f"      {name}：变化量：{ch}，变化百分比：{pc}，趋势：{trend}；")
            else:
                summary_lines.append('\n显著变化判定（联合判定法）：输入数据中未同时包含"变化量"和"百分比"列，无法进行联合判定。')
        except Exception as e:
            summary_lines.append(f"\n显著变化判定出错：{e}")

        return "\n".join(summary_lines)

    def on_training_error(self, error_msg):
        """处理训练错误"""
        print(f"训练错误: {error_msg}")
        try:
            QMessageBox.critical(self, "训练错误", error_msg)
            self.statusBar().showMessage(f"错误: {error_msg}")
        except Exception as e:
            print(f"显示错误消息时出错: {e}")

    def on_thread_finished(self):
        """线程结束时的清理，确保 UI 状态被正确恢复。"""
        try:
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            # 如果线程结束且进度未到100%，将进度设置为完成状态
            if self.progress_bar.value() < 100:
                self.progress_bar.setValue(100)
            if self.statusBar().currentMessage() == "训练中..." or "正在停止" in self.statusBar().currentMessage():
                self.statusBar().showMessage("训练任务已结束")
            self.current_worker = None  # 清空当前 worker 引用
            print("训练任务已完成并清理")
        except Exception as e:
            print(f"清理时发生异常: {e}")

    def closeEvent(self, event):
        """在窗口关闭时确保后台训练线程被安全停止，避免发信号到已销毁的对象导致崩溃。"""
        if self.current_worker and self.thread_pool.activeThreadCount() > 0:
            reply = QMessageBox.question(
                self,
                "正在训练",
                "训练正在进行，是否停止训练并退出？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                try:
                    # 停止当前任务
                    if self.current_worker:
                        self.current_worker.stop()
                    # 等待线程池中的所有任务完成（最长等待 5 秒）
                    self.thread_pool.waitForDone(5000)
                    print("所有训练任务已停止")
                except Exception as e:
                    print(f"停止训练任务时发生异常: {e}")
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()



if __name__ == "__main__":

    # 启用高 DPI 缩放与高质量像素图支持，使得在 Windows 系统缩放下界面自适应
    # QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    # QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # 统一应用字体为微软雅黑，适当增大字体以提高可读性（在高 DPI 下会自动缩放）
    try:
        app.setFont(QFont("Microsoft YaHei", 9))
    except Exception:
        pass

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
