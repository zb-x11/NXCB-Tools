import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog, QMenuBar, QMenu, QAction, QLabel
from PyQt5.QtCore import Qt
from scipy.fft import fft
from scipy.signal import welch

class SignalAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Signal Analysis Tool')
        self.setGeometry(100, 100, 800, 600)

        # 主布局
        self.layout = QVBoxLayout()

        # 显示图片的标签
        self.image_label = QLabel('图像显示区域')
        self.layout.addWidget(self.image_label)

        # 按钮和菜单设置
        self.button_load_data = QPushButton('加载数据文件')
        self.button_load_data.clicked.connect(self.load_data)
        self.layout.addWidget(self.button_load_data)

        # 创建中央widget
        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

        # 创建菜单
        self.menu_bar = self.menuBar()
        file_menu = self.menu_bar.addMenu('文件')
        load_action = QAction('加载数据', self)
        load_action.triggered.connect(self.load_data)
        file_menu.addAction(load_action)

        # 初始化数据
        self.data = None
        self.fs = 1e3  # 假设采样频率为1000Hz

    def load_data(self):
        """加载txt数据文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, '打开文件', '', 'Text Files (*.txt);;All Files (*)')
        if file_path:
            self.data = np.loadtxt(file_path)
            self.process_data()

    def process_data(self):
        """处理数据：FFT、计算SNR和SNDR等"""
        if self.data is None:
            return

        # FFT
        n = len(self.data)
        freq = np.fft.fftfreq(n, 1/self.fs)
        fft_data = fft(self.data)
        magnitude = np.abs(fft_data)

        # 计算SNR和SNDR (此处仅为示例公式)
        signal_power = np.mean(np.square(self.data))
        noise_power = np.var(self.data)
        snr = 10 * np.log10(signal_power / noise_power)

        # 绘制频谱图
        self.plot_spectrum(freq, magnitude)

        # 输出性能指标
        self.display_metrics(snr)

    def plot_spectrum(self, freq, magnitude):
        """绘制频谱图"""
        plt.figure(figsize=(8, 6))
        plt.plot(freq, magnitude)
        plt.title('Frequency Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.tight_layout()

        # 将图像嵌入到PyQt界面中
        plt.savefig('spectrum.png')
        self.image_label.setPixmap('spectrum.png')

    def display_metrics(self, snr):
        """显示性能指标"""
        metrics = f"SNR: {snr:.2f} dB"
        self.image_label.setText(metrics)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SignalAnalysisApp()
    window.show()
    sys.exit(app.exec_())
