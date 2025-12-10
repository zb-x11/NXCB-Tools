import tkinter as tk
from tkinter import ttk, messagebox
import serial
import serial.tools.list_ports
import threading
import time
import subprocess
import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageTk
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]          # 这个 GUI 文件就放在 YOLOv5 根目录（和 detect.py 同级）
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from ultralytics.utils.plotting import Annotator, colors  # 你 detect.py 里就是这么导的
from models.common import DetectMultiBackend
from utils.general import (
    cv2,
    check_img_size,
    non_max_suppression,
    scale_boxes,
)

from tkinter import filedialog

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.torch_utils import select_device
from utils.augmentations import letterbox


# ================= 数据标注窗口 =================
class ImageAnnotatorWindow:
    """
    简单 YOLO 格式图像标注工具：
    - 选择图片文件夹
    - 上一张 / 下一张
    - 鼠标拖拽绘制矩形框
    - 输入 Class ID
    - 保存为 YOLO txt：class x_center y_center w h (归一化)
    """

    def __init__(self, root: tk.Toplevel):
        self.root = root
        self.root.title("YOLO Image Annotation")
        self.root.geometry("1000x800")

        # 图像列表 & 当前索引
        self.image_dir = None
        self.image_files = []
        self.current_index = 0

        # 当前图像 & 显示相关
        self.current_image = None        # PIL 原图
        self.current_photo = None        # Tk 版本
        self.orig_w = None
        self.orig_h = None
        self.scale_x = 1.0
        self.scale_y = 1.0

        # 标注数据：列表，每个元素为 dict：
        # {"x1":..., "y1":..., "x2":..., "y2":..., "class_id":..., "rect_id":...}
        self.boxes = []

        # 画框状态
        self.drawing = False
        self.start_x = None
        self.start_y = None
        self.temp_rect_id = None

        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ----------- UI 布局 -----------
    def create_widgets(self):
        # 顶部：文件夹选择和图片信息
        top_frame = ttk.LabelFrame(self.root, text="Dataset")
        top_frame.pack(fill="x", padx=10, pady=10)

        btn_choose_dir = ttk.Button(top_frame, text="Select Image Folder", command=self.choose_dir)
        btn_choose_dir.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.dir_var = tk.StringVar(value="No folder selected")
        lbl_dir = ttk.Label(top_frame, textvariable=self.dir_var, width=60)
        lbl_dir.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.img_info_var = tk.StringVar(value="Image: 0 / 0")
        lbl_info = ttk.Label(top_frame, textvariable=self.img_info_var)
        lbl_info.grid(row=0, column=2, padx=5, pady=5, sticky="e")

        # 中部：左画布显示图像，右边控制区域
        middle_frame = ttk.Frame(self.root)
        middle_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # 画布
        self.canvas = tk.Canvas(middle_frame, bg="grey", width=800, height=600)
        self.canvas.pack(side="left", fill="both", expand=True)

        # 绑定鼠标事件（画框）
        self.canvas.bind("<Button-1>", self.on_canvas_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_canvas_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_mouse_up)

        # 右侧控制区域
        ctrl_frame = ttk.LabelFrame(middle_frame, text="Annotation Control", width=200)
        ctrl_frame.pack(side="right", fill="y", padx=10)

        # Class ID 输入
        ttk.Label(ctrl_frame, text="Class ID:").pack(anchor="w", padx=5, pady=(10, 2))
        self.class_id_var = tk.StringVar(value="0")
        entry_class = ttk.Entry(ctrl_frame, textvariable=self.class_id_var, width=10)
        entry_class.pack(anchor="w", padx=5, pady=2)

        # 盒子列表
        ttk.Label(ctrl_frame, text="Boxes:").pack(anchor="w", padx=5, pady=(10, 2))
        self.box_listbox = tk.Listbox(ctrl_frame, height=10)
        self.box_listbox.pack(fill="x", padx=5, pady=2)

        btn_delete_box = ttk.Button(ctrl_frame, text="Delete Selected Box", command=self.delete_selected_box)
        btn_delete_box.pack(fill="x", padx=5, pady=5)

        btn_clear_boxes = ttk.Button(ctrl_frame, text="Clear All Boxes", command=self.clear_boxes)
        btn_clear_boxes.pack(fill="x", padx=5, pady=5)

        # 导航按钮
        nav_frame = ttk.Frame(ctrl_frame)
        nav_frame.pack(fill="x", padx=5, pady=(20, 5))

        btn_prev = ttk.Button(nav_frame, text="<< Prev", command=self.prev_image)
        btn_prev.grid(row=0, column=0, padx=2, pady=2)

        btn_next = ttk.Button(nav_frame, text="Next >>", command=self.next_image)
        btn_next.grid(row=0, column=1, padx=2, pady=2)

        # 保存按钮
        btn_save = ttk.Button(ctrl_frame, text="Save Labels", command=self.save_labels)
        btn_save.pack(fill="x", padx=5, pady=(10, 5))

        # 日志简单显示
        self.log_var = tk.StringVar(value="")
        lbl_log = ttk.Label(ctrl_frame, textvariable=self.log_var, wraplength=180, foreground="blue")
        lbl_log.pack(fill="x", padx=5, pady=(10, 5))

    # ----------- 文件夹 & 图片加载 -----------
    def choose_dir(self):
        folder = filedialog.askdirectory(title="Select image folder")
        if not folder:
            return
        self.image_dir = folder
        self.dir_var.set(folder)

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
        files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
        files.sort()
        self.image_files = files
        self.current_index = 0

        if not self.image_files:
            self.img_info_var.set("Image: 0 / 0")
            self.log("No image files found in this folder.")
            return

        self.log(f"Found {len(self.image_files)} images.")
        self.load_image(self.current_index)

    def load_image(self, index: int):
        if not self.image_files:
            return

        if index < 0:
            index = 0
        if index >= len(self.image_files):
            index = len(self.image_files) - 1

        self.current_index = index
        img_name = self.image_files[self.current_index]
        img_path = os.path.join(self.image_dir, img_name)

        # 加载原图
        img = Image.open(img_path).convert("RGB")
        self.current_image = img
        self.orig_w, self.orig_h = img.size

        # 按画布大小缩放
        canvas_w = self.canvas.winfo_width() or 800
        canvas_h = self.canvas.winfo_height() or 600

        ratio = min(canvas_w / self.orig_w, canvas_h / self.orig_h)
        display_w = int(self.orig_w * ratio)
        display_h = int(self.orig_h * ratio)
        self.scale_x = display_w / self.orig_w
        self.scale_y = display_h / self.orig_h

        img_resized = img.resize((display_w, display_h), Image.BILINEAR)
        self.current_photo = ImageTk.PhotoImage(img_resized)

        # 清空画布并绘制图像到中心
        self.canvas.delete("all")
        offset_x = (canvas_w - display_w) // 2
        offset_y = (canvas_h - display_h) // 2
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.current_photo)
        self.canvas.image = self.current_photo  # 防止被回收

        # 每次加载新图，清空旧标注，重新加载对应 txt
        self.boxes.clear()
        self.box_listbox.delete(0, tk.END)
        self.load_labels_for_image()

        # 更新信息
        self.img_info_var.set(f"Image: {self.current_index + 1} / {len(self.image_files)}  ({img_name})")
        self.log(f"Loaded image: {img_name}")

    # ----------- YOLO 标注读写 -----------
    def labels_path_for_current_image(self):
        if not self.image_files:
            return None
        img_name = self.image_files[self.current_index]
        base, _ = os.path.splitext(img_name)
        # 这里简单起见，直接存在同一目录下 base.txt
        return os.path.join(self.image_dir, base + ".txt")

    def load_labels_for_image(self):
        path = self.labels_path_for_current_image()
        if not path or not os.path.exists(path):
            return

        try:
            with open(path, "r") as f:
                lines = f.readlines()
        except Exception as e:
            self.log(f"Failed to read label file: {e}")
            return

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = parts[0]
            xc, yc, w, h = map(float, parts[1:])

            # YOLO 归一化 -> 原图坐标
            x_center = xc * self.orig_w
            y_center = yc * self.orig_h
            bw = w * self.orig_w
            bh = h * self.orig_h

            x1 = x_center - bw / 2
            y1 = y_center - bh / 2
            x2 = x_center + bw / 2
            y2 = y_center + bh / 2

            # 转换到显示坐标
            dx1 = x1 * self.scale_x
            dy1 = y1 * self.scale_y
            dx2 = x2 * self.scale_x
            dy2 = y2 * self.scale_y

            rect_id = self.canvas.create_rectangle(dx1, dy1, dx2, dy2, outline="red", width=2)
            box = {"x1": dx1, "y1": dy1, "x2": dx2, "y2": dy2, "class_id": class_id, "rect_id": rect_id}
            self.boxes.append(box)
            self.box_listbox.insert(tk.END, f"{class_id}: ({int(dx1)}, {int(dy1)})-({int(dx2)}, {int(dy2)})")

        self.log(f"Loaded {len(self.boxes)} boxes from label file.")

    def save_labels(self):
        if not self.image_files:
            return
        path = self.labels_path_for_current_image()
        if not path:
            return

        lines = []
        for box in self.boxes:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            class_id = box["class_id"]

            # 先转回原图坐标
            rx1 = x1 / self.scale_x
            ry1 = y1 / self.scale_y
            rx2 = x2 / self.scale_x
            ry2 = y2 / self.scale_y

            bw = rx2 - rx1
            bh = ry2 - ry1
            x_center = rx1 + bw / 2
            y_center = ry1 + bh / 2

            xc = x_center / self.orig_w
            yc = y_center / self.orig_h
            w = bw / self.orig_w
            h = bh / self.orig_h

            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        try:
            with open(path, "w") as f:
                f.writelines(lines)
            self.log(f"Saved {len(self.boxes)} boxes to {os.path.basename(path)}")
        except Exception as e:
            self.log(f"Failed to save labels: {e}")

    # ----------- 画框（鼠标事件） -----------
    def on_canvas_mouse_down(self, event):
        if not self.current_image:
            return
        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y
        # 临时矩形
        self.temp_rect_id = self.canvas.create_rectangle(
            self.start_x,
            self.start_y,
            self.start_x,
            self.start_y,
            outline="yellow",
            width=2,
        )

    def on_canvas_mouse_move(self, event):
        if not self.drawing or self.temp_rect_id is None:
            return
        # 更新临时矩形
        self.canvas.coords(self.temp_rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_canvas_mouse_up(self, event):
        if not self.drawing:
            return
        self.drawing = False

        end_x, end_y = event.x, event.y
        if self.temp_rect_id is None:
            return

        x1, y1 = self.start_x, self.start_y
        x2, y2 = end_x, end_y

        # 保证左上右下
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        # 最小尺寸限制，避免误点
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            self.canvas.delete(self.temp_rect_id)
            self.temp_rect_id = None
            return

        # 固定最终矩形（改成红色）
        self.canvas.itemconfig(self.temp_rect_id, outline="red")
        rect_id = self.temp_rect_id
        self.temp_rect_id = None

        class_id = self.class_id_var.get().strip()
        if class_id == "":
            class_id = "0"

        box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "class_id": class_id, "rect_id": rect_id}
        self.boxes.append(box)
        self.box_listbox.insert(tk.END, f"{class_id}: ({int(x1)}, {int(y1)})-({int(x2)}, {int(y2)})")

    # ----------- Box 管理 -----------
    def delete_selected_box(self):
        sel = self.box_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        box = self.boxes[idx]
        # 删除画布中的矩形
        self.canvas.delete(box["rect_id"])
        # 删除数据 & 列表项
        del self.boxes[idx]
        self.box_listbox.delete(idx)

    def clear_boxes(self):
        for box in self.boxes:
            self.canvas.delete(box["rect_id"])
        self.boxes.clear()
        self.box_listbox.delete(0, tk.END)

    # ----------- 图片导航 -----------
    def prev_image(self):
        if not self.image_files:
            return
        # 切换前保存当前标注
        self.save_labels()
        new_index = self.current_index - 1
        if new_index < 0:
            new_index = 0
        self.load_image(new_index)

    def next_image(self):
        if not self.image_files:
            return
        # 切换前保存当前标注
        self.save_labels()
        new_index = self.current_index + 1
        if new_index >= len(self.image_files):
            new_index = len(self.image_files) - 1
        self.load_image(new_index)

    # ----------- 小工具 -----------
    def log(self, text):
        self.log_var.set(text)

    def on_close(self):
        # 关闭前保存当前图片的标注
        if self.image_files:
            self.save_labels()
        self.root.destroy()

# ================= 眼图窗口 =================

class EyeDiagramWindow:
    def __init__(self, root: tk.Toplevel):
        self.root = root
        self.root.title("Eye Diagram")
        self.root.geometry("840x640")

        self.file_path = None  # 当前选择的文件路径

        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        # ===== 顶部：配置区 =====
        frame_top = ttk.LabelFrame(self.root, text="Config")
        frame_top.pack(fill="x", padx=10, pady=10)

        # 文件路径显示
        ttk.Label(frame_top, text="File:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.path_var = tk.StringVar(value="No file selected")
        lbl_path = ttk.Label(frame_top, textvariable=self.path_var, width=60)
        lbl_path.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        btn_choose = ttk.Button(frame_top, text="Browse", width=10, command=self.choose_file)
        btn_choose.grid(row=0, column=2, padx=5, pady=5)

        # 每符号采样点数 Samples / symbol
        ttk.Label(frame_top, text="Samples / symbol:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.sps_var = tk.StringVar(value="64")  # 默认 64 点/符号
        entry_sps = ttk.Entry(frame_top, textvariable=self.sps_var, width=10)
        entry_sps.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # 调制方式 NRZ / PAM4
        ttk.Label(frame_top, text="Modulation:").grid(row=1, column=2, padx=5, pady=5, sticky="e")
        self.mod_var = tk.StringVar(value="NRZ")
        combo_mod = ttk.Combobox(
            frame_top,
            textvariable=self.mod_var,
            values=["NRZ", "PAM4"],
            width=8,
            state="readonly",
        )
        combo_mod.grid(row=1, column=3, padx=5, pady=5, sticky="w")

        # 显示多少个符号（用于波形绘制）
        ttk.Label(frame_top, text="Symbols to show:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.symbols_var = tk.StringVar(value="10")  # 默认 10 个符号
        entry_symbols = ttk.Entry(frame_top, textvariable=self.symbols_var, width=10)
        entry_symbols.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # Generate Eye 按钮
        btn_eye = ttk.Button(frame_top, text="Generate Eye", width=14, command=self.generate_eye_diagram)
        btn_eye.grid(row=2, column=2, padx=5, pady=5, sticky="w")

        # Generate Waveform 按钮
        btn_wave = ttk.Button(frame_top, text="Generate Waveform", width=18, command=self.generate_waveform)
        btn_wave.grid(row=2, column=3, padx=5, pady=5, sticky="w")

        # ===== 中部：日志区（小一点） =====
        frame_log = ttk.LabelFrame(self.root, text="Log")
        frame_log.pack(fill="x", padx=10, pady=(0, 5))

        self.text_log = tk.Text(frame_log, wrap="none", height=6)
        self.text_log.pack(side="left", fill="x", expand=True)

        scroll_log = ttk.Scrollbar(frame_log, command=self.text_log.yview)
        scroll_log.pack(side="right", fill="y")
        self.text_log.config(yscrollcommand=scroll_log.set)

        # ===== 底部：绘图区（眼图 / 波形） =====
        frame_plot = ttk.LabelFrame(self.root, text="Plot")
        frame_plot.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        # 初始不画网格，只是空白背景
        self.fig.patch.set_facecolor("white")
        self.ax.set_facecolor("white")

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_plot)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)
        self.canvas.draw()

    # ===== 工具函数 =====
    def log(self, text: str):
        """写日志到上面的 Log 区。"""
        self.text_log.insert("end", text + "\n")
        self.text_log.see("end")

    def choose_file(self):
        filetypes = [
            ("Text files", "*.txt"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Select ADC data file", filetypes=filetypes)
        if path:
            self.file_path = path
            self.path_var.set(path)
            self.log(f"[INFO] Selected file: {path}")

    def _load_data(self):
        """从 txt 文件读取 ADC 数据，一维 numpy 数组。"""
        if not self.file_path:
            messagebox.showwarning("Warning", "Please select a data file first.")
            return None

        try:
            with open(self.file_path, "r") as f:
                text = f.read()
            # 把换行替换成空格，然后按空格切分
            parts = text.replace("\n", " ").split()
            data = np.array([float(x) for x in parts], dtype=float)
            if data.size < 2:
                messagebox.showwarning("Warning", "Data too short.")
                return None
            return data
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read file:\n{e}")
            return None

    def _get_sps(self):
        """读取每符号采样点数 Samples / symbol。"""
        try:
            sps = int(self.sps_var.get())
            if sps <= 0:
                raise ValueError
            return sps
        except ValueError:
            messagebox.showwarning("Warning", "Samples / symbol must be a positive integer.")
            return None

    def _get_symbols_to_show(self):
        """读取要显示多少个符号（用于波形生成）。"""
        try:
            symbols_to_show = int(self.symbols_var.get())
            if symbols_to_show <= 0:
                raise ValueError
            return symbols_to_show
        except ValueError:
            messagebox.showwarning("Warning", "Symbols to show must be a positive integer.")
            return None

    # ===== 眼图生成 =====
    def generate_eye_diagram(self):
        data = self._load_data()
        if data is None:
            return

        sps = self._get_sps()
        if sps is None:
            return

        num_symbols = data.size // sps
        if num_symbols < 2:
            messagebox.showwarning(
                "Warning",
                f"Not enough data to form multiple symbols (current symbols: {num_symbols}).",
            )
            return

        data_use = data[: num_symbols * sps]
        segments = data_use.reshape(num_symbols, sps)

        mod = self.mod_var.get()

        self.ax.clear()
        t = np.arange(sps)

        # 叠加每个符号的轨迹
        for seg in segments:
            self.ax.plot(t, seg, alpha=0.3)

        self.ax.set_title(f"Eye Diagram ({mod})")
        self.ax.set_xlabel("Sample Index")
        self.ax.set_ylabel("Amplitude")
        # 网格和曲线一起出现
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()
        self.log(f"[INFO] Eye diagram generated. Modulation = {mod}, symbols = {num_symbols}, sps = {sps}")

    # ===== 波形生成 =====
    def generate_waveform(self):
        data = self._load_data()
        if data is None:
            return

        sps = self._get_sps()
        if sps is None:
            return

        symbols_to_show = self._get_symbols_to_show()
        if symbols_to_show is None:
            return

        mod = self.mod_var.get()

        max_points = symbols_to_show * sps
        if data.size < max_points:
            max_points = data.size

        t = np.arange(max_points)

        self.ax.clear()
        self.ax.plot(t, data[:max_points], linewidth=1.0)

        self.ax.set_title(f"Waveform ({mod})")
        self.ax.set_xlabel("Sample Index")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, alpha=0.3)

        self.canvas.draw()
        self.log(
            f"[INFO] Waveform generated. Points = {max_points}, symbols = {symbols_to_show}, sps = {sps}, modulation = {mod}"
        )

    def on_close(self):
        self.root.destroy()



# ================= 物体检测窗口 =================
class DetectionWindow:
    def __init__(self, root: tk.Toplevel):
        self.root = root
        self.root.title("物体检测（NXCB）")
        self.root.geometry("800x600")  # 整体窗口大小固定

        # YOLO / 摄像头状态
        self.running = False
        self.model = None
        self.device = None
        self.names = None
        self.stride = None
        self.cap = None
        self.detect_thread = None

        # 一些检测参数（可按你 detect.py 的设置调整）
        self.imgsz = (640, 640)
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.max_det = 1000

        self.create_widgets()

        # 窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        # ===== 顶部控制区域（状态 + 按钮） =====
        frame_top = ttk.Frame(self.root)
        frame_top.pack(fill="x", padx=10, pady=10)

        self.status_var = tk.StringVar(value="状态：未运行")
        lbl_status = ttk.Label(frame_top, textvariable=self.status_var, foreground="blue")
        lbl_status.pack(side="left", padx=5)

        btn_run = ttk.Button(frame_top, text="运行检测", width=12, command=self.run_detection)
        btn_run.pack(side="right", padx=5)

        btn_stop = ttk.Button(frame_top, text="停止检测", width=12, command=self.stop_detection)
        btn_stop.pack(side="right", padx=5)

        # ===== 日志区域（在视频上方） =====
        frame_log = ttk.LabelFrame(self.root, text="日志")
        frame_log.pack(fill="x", padx=10, pady=(0, 5))  # 横向占满，上下不太高

        # 日志不需要太大，高度调小一点，例如 6 行
        self.text_log = tk.Text(frame_log, wrap="none", height=6)
        self.text_log.pack(side="left", fill="x", expand=True)

        scroll_log = ttk.Scrollbar(frame_log, command=self.text_log.yview)
        scroll_log.pack(side="right", fill="y")
        self.text_log.config(yscrollcommand=scroll_log.set)

        # ===== 视频显示区域（在最下面，占据大部分空间） =====
        frame_video = ttk.LabelFrame(self.root, text="检测画面")
        frame_video.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # 视频 Label，背景白色
        self.video_label = tk.Label(frame_video, bg="white")
        self.video_label.pack(fill="both", expand=True)

        # # 初始化一张白色画面（未播放时显示）
        # self._init_white_frame()


    # def _init_white_frame(self):
    #     """在视频区域先显示一张白板图，表示还未开始播放。"""
    #     # 创建纯白图片
    #     white_img = Image.new("RGB", (self.video_width, self.video_height), color="white")
    #     imgtk = ImageTk.PhotoImage(white_img)
    #
    #     # 保存引用 + 显示
    #     self.video_label.imgtk = imgtk
    #     self.video_label.configure(image=imgtk)



    # ===== 工具：日志输出 =====
    def append_log(self, text: str):
        self.text_log.insert("end", text)
        self.text_log.see("end")

    # ===== 按钮：运行检测 =====
    def run_detection(self):
        if self.running:
            messagebox.showinfo("提示", "检测已经在运行中。")
            return

        self.status_var.set("状态：等待中（加载模型、打开摄像头中...）")
        self.append_log("[INFO] 等待中，准备启动 NXCB 检测...\n")

        self.running = True
        self.detect_thread = threading.Thread(target=self._detect_loop, daemon=True)
        self.detect_thread.start()

    # ===== 按钮：停止检测 =====
    def stop_detection(self):
        if not self.running:
            self.append_log("[INFO] 检测未在运行，无需停止。\n")
            return

        self.append_log("[INFO] 请求停止检测...\n")
        self.running = False  # 循环会自动退出

    # ===== 检测线程：真正跑 YOLOv5 + 摄像头 =====
    def _detect_loop(self):
        try:
            # 1. 加载模型（只加载一次）
            if self.model is None:
                self.append_log("[INFO] 正在加载 NXCB 模型参数...\n")
                weights = ROOT / "yolov5s.pt"
                data = ROOT / "data/coco128.yaml"

                self.device = select_device("")  # 自动选择 CPU/GPU
                self.model = DetectMultiBackend(weights, device=self.device, dnn=False, data=data, fp16=False)
                self.stride, self.names, pt = self.model.stride, self.model.names, self.model.pt
                self.imgsz = check_img_size(self.imgsz, s=self.stride)

                self.append_log(f"[INFO] 模型加载完成，使用设备：{self.device}\n")

            # 2. 打开摄像头（等价于 source=0）
            self.append_log("[INFO] 正在打开摄像头...\n")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.append_log("[ERROR] 无法打开摄像头。\n")
                self.running = False
                return

            self.status_var.set("状态：正在运行（按“停止检测”可结束）")
            self.append_log("[INFO] 摄像头已打开，开始检测。\n")

            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    self.append_log("[WARN] 读取摄像头帧失败。\n")
                    break

                # ============= 这里是简化版 detect.py 推理流程 =============
                im0 = frame.copy()
                # letterbox 缩放 + 填充
                im = letterbox(im0, self.imgsz, stride=self.stride, auto=True)[0]
                # BGR->RGB, HWC->CHW
                im = im.transpose((2, 0, 1))
                im = np.ascontiguousarray(im)

                im_tensor = torch.from_numpy(im).to(self.device)
                im_tensor = im_tensor.float()
                im_tensor /= 255.0
                if im_tensor.ndim == 3:
                    im_tensor = im_tensor[None]

                # 推理
                pred = self.model(im_tensor, augment=False, visualize=False)
                # NMS
                pred = non_max_suppression(
                    pred,
                    self.conf_thres,
                    self.iou_thres,
                    classes=None,
                    agnostic=False,
                    max_det=self.max_det,
                )

                # 画框
                for det in pred:
                    annotator = Annotator(im0, line_width=2, example=str(self.names))
                    if len(det):
                        det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im0.shape).round()
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)
                            label = f"{self.names[c]} {float(conf):.2f}"
                            annotator.box_label(xyxy, label, color=colors(c, True))
                    im0 = annotator.result()

                # ============= 显示到 Tkinter 窗口中 =============
                img_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                pil_img = pil_img.resize((800, 600))
                imgtk = ImageTk.PhotoImage(image=pil_img)
                # 防止被回收：
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

                # 控制刷新频率
                time.sleep(0.03)

            self.append_log("[INFO] 检测循环结束。\n")

        except Exception as e:
            self.append_log(f"[ERROR] 检测线程异常: {e}\n")
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.running = False
            self.status_var.set("状态：未运行")

    def on_close(self):
        # 关闭窗口时，先请求停止检测
        if self.running:
            self.append_log("[INFO] 窗口关闭，请求停止检测...\n")
            self.running = False
            time.sleep(0.2)

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.root.destroy()


# ================= 串口助手窗口 =================
class SerialGUI:
    def __init__(self, root: tk.Toplevel):
        self.root = root
        self.root.title("串口助手")
        self.ser = None
        self.running = False

        self.create_widgets()
        self.refresh_ports()

        # 当用户点关闭按钮时，先关串口再销毁窗口
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        # ======= 上方配置区域 =======
        frame_cfg = ttk.LabelFrame(self.root, text="串口配置")
        frame_cfg.pack(fill="x", padx=10, pady=5)

        # 串口选择
        ttk.Label(frame_cfg, text="串口：").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.port_var = tk.StringVar()
        self.port_cb = ttk.Combobox(frame_cfg, textvariable=self.port_var, width=12, state="readonly")
        self.port_cb.grid(row=0, column=1, padx=5, pady=5)

        btn_refresh = ttk.Button(frame_cfg, text="刷新", width=6, command=self.refresh_ports)
        btn_refresh.grid(row=0, column=2, padx=5, pady=5)

        # 波特率
        ttk.Label(frame_cfg, text="波特率：").grid(row=0, column=3, padx=5, pady=5, sticky="e")
        self.baud_var = tk.StringVar(value="115200")
        self.baud_cb = ttk.Combobox(
            frame_cfg, textvariable=self.baud_var,
            values=["9600", "19200", "38400", "57600", "115200", "230400"],
            width=10, state="readonly"
        )
        self.baud_cb.grid(row=0, column=4, padx=5, pady=5)

        # DTR / RTS 选项
        self.dtr_var = tk.BooleanVar(value=False)
        self.rts_var = tk.BooleanVar(value=False)
        chk_dtr = ttk.Checkbutton(frame_cfg, text="启用 DTR", variable=self.dtr_var)
        chk_rts = ttk.Checkbutton(frame_cfg, text="启用 RTS", variable=self.rts_var)
        chk_dtr.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        chk_rts.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # 连接 / 断开
        self.btn_connect = ttk.Button(frame_cfg, text="连接", width=10, command=self.toggle_connect)
        self.btn_connect.grid(row=1, column=3, padx=5, pady=5)
        self.status_var = tk.StringVar(value="未连接")
        ttk.Label(frame_cfg, textvariable=self.status_var, foreground="blue").grid(
            row=1, column=4, padx=5, pady=5
        )

        # ======= 接收区域 =======
        frame_rx = ttk.LabelFrame(self.root, text="接收区")
        frame_rx.pack(fill="both", expand=True, padx=10, pady=5)

        self.text_rx = tk.Text(frame_rx, wrap="none", height=15)
        self.text_rx.pack(fill="both", expand=True, side="left")
        scroll_rx = ttk.Scrollbar(frame_rx, command=self.text_rx.yview)
        scroll_rx.pack(side="right", fill="y")
        self.text_rx.config(yscrollcommand=scroll_rx.set)

        # ======= 发送区域 =======
        frame_tx = ttk.LabelFrame(self.root, text="发送区")
        frame_tx.pack(fill="x", padx=10, pady=5)

        self.entry_tx = tk.Entry(frame_tx)
        self.entry_tx.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        self.entry_tx.bind("<Return>", lambda event: self.send_data())

        btn_send = ttk.Button(frame_tx, text="发送", width=8, command=self.send_data)
        btn_send.pack(side="left", padx=5, pady=5)

    # -------- 串口相关 --------
    def refresh_ports(self):
        ports = serial.tools.list_ports.comports()
        names = [p.device for p in ports]
        self.port_cb["values"] = names
        if names and not self.port_var.get():
            self.port_var.set(names[0])

    def toggle_connect(self):
        if self.ser and self.ser.is_open:
            self.close_serial()
        else:
            self.open_serial()

    def open_serial(self):
        port = self.port_var.get()
        baud = self.baud_var.get()

        if not port:
            messagebox.showwarning("提示", "请先选择串口号")
            return

        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=int(baud),
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False,
            )

            # 根据勾选设置 DTR / RTS（不勾选时就是你之前的 ser.setDTR(False) / setRTS(False)）
            self.ser.setDTR(self.dtr_var.get())
            self.ser.setRTS(self.rts_var.get())

        except Exception as e:
            messagebox.showerror("错误", f"打开串口失败：\n{e}")
            return

        self.running = True
        self.btn_connect.config(text="断开")
        self.status_var.set(f"已连接：{port} @ {baud}")
        self.text_rx.insert("end", f"[连接] {port} @ {baud}\n")
        self.text_rx.see("end")

        threading.Thread(target=self.read_loop, daemon=True).start()

    def close_serial(self):
        self.running = False
        time.sleep(0.2)
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
        self.btn_connect.config(text="连接")
        self.status_var.set("未连接")
        self.text_rx.insert("end", "[断开]\n")
        self.text_rx.see("end")

    def read_loop(self):
        while self.running and self.ser and self.ser.is_open:
            try:
                n = self.ser.in_waiting
                if n:
                    data = self.ser.read(n)
                    text = data.decode("utf-8", errors="ignore")
                    self.text_rx.after(0, self.append_rx, text)
            except Exception as e:
                self.text_rx.after(0, self.append_rx, f"\n[接收错误] {e}\n")
                break
            time.sleep(0.02)

    def append_rx(self, text):
        self.text_rx.insert("end", text)
        self.text_rx.see("end")

    def send_data(self):
        if not (self.ser and self.ser.is_open):
            messagebox.showwarning("提示", "串口未连接")
            return

        msg = self.entry_tx.get()
        if not msg:
            return

        try:
            data = (msg + "\r\n").encode("utf-8")
            self.ser.write(data)
            self.text_rx.insert("end", f"[TX] {msg}\n")
            self.text_rx.see("end")
            self.entry_tx.delete(0, "end")
        except Exception as e:
            messagebox.showerror("错误", f"发送失败：\n{e}")

    def on_close(self):
        # 副窗口关闭时，顺便关串口
        if self.ser and self.ser.is_open:
            self.close_serial()
        self.root.destroy()


# ================= 主窗口 =================
class MainApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("NXCB software Tools")
        self.root.geometry("340x220")
        self.serial_window = None  # 保存串口助手窗口的引用
        self.detect_window = None  # 保存物体检测窗口的引用
        self.eye_window = None     # 保存物体检测窗口的引用
        self.annot_window = None   # 保存图像标注窗口的引用
        self.create_menu()
        self.create_main_ui()

    def create_menu(self):
        menubar = tk.Menu(self.root)

        # 文件菜单（示例）
        menu_file = tk.Menu(menubar, tearoff=0)
        menu_file.add_command(label="退出", command=self.root.quit)
        menubar.add_cascade(label="文件", menu=menu_file)

        # 工具菜单：这里放“串口助手”
        menu_tools = tk.Menu(menubar, tearoff=0)
        menu_tools.add_command(label="串口助手", command=self.open_serial_helper)
        menubar.add_cascade(label="工具", menu=menu_tools)

        self.root.config(menu=menubar)

    def create_main_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        # 主提示标签放在 row=0, col=1
        # lbl = ttk.Label(main_frame, text="这里是主窗口内容，可以放你的其它功能~", font=("微软雅黑", 16))
        # lbl.grid(row=0, column=1, padx=20, pady=20, sticky="w")

        # 按钮放在 row=0, col=0
        btn_open_serial = ttk.Button(main_frame, text="打开串口助手", width=16, command=self.open_serial_helper)
        btn_open_serial.grid(row=0, column=0, padx=20, pady=20, sticky="w")
        btn_open_serial.configure(padding=(2, 5))
        # 物体检测按钮（在同一行第 1 列，按你需要改 row/column）
        btn_open_detect = ttk.Button(
            main_frame,
            text="物体检测",
            command=self.open_detect_window,
            width=16
        )
        btn_open_detect.grid(row=0, column=1, padx=20, pady=20, sticky="w")

        btn_open_detect.configure(padding=(2, 5))

        btn_open_eye = ttk.Button(
            main_frame,
            text="波形工具",
            command=self.open_eye_window,
            width=16
        )
        btn_open_eye.grid(row=1, column=0, padx=20, pady=20, sticky="w")
        btn_open_eye.configure(padding=(2, 5))

        btn_open_annot = ttk.Button(main_frame, text="图像标注", command=self.open_annot_window, width=16)
        btn_open_annot.grid(row=1, column=1, padx=20, pady=20, sticky="w")
        btn_open_annot.configure(padding=(2, 5))


        # 防止 grid “缩成一团”
        # main_frame.columnconfigure(0, weight=0)  # 第一列不扩展
        # main_frame.columnconfigure(1, weight=1)  # 第二列自动扩展
        # main_frame.rowconfigure(0, weight=1)



    def open_serial_helper(self):
        # 如果已经打开了，就把窗口提到最前
        if self.serial_window and tk.Toplevel.winfo_exists(self.serial_window):
            self.serial_window.lift()
            return

        # 新建一个 Toplevel 作为“副窗口”
        win = tk.Toplevel(self.root)
        SerialGUI(win)
        self.serial_window = win

    def open_detect_window(self):
        # 如果已经打开了，就把它提到前面
        if self.detect_window and tk.Toplevel.winfo_exists(self.detect_window):
            self.detect_window.lift()
            return

        win = tk.Toplevel(self.root)
        DetectionWindow(win)
        self.detect_window = win

    def open_eye_window(self):
        # 如果已经打开，就激活
        if self.eye_window and tk.Toplevel.winfo_exists(self.eye_window):
            self.eye_window.lift()
            return

        win = tk.Toplevel(self.root)
        EyeDiagramWindow(win)
        self.eye_window = win

    def open_annot_window(self):
        if self.annot_window and tk.Toplevel.winfo_exists(self.annot_window):
            self.annot_window.lift()
            return

        win = tk.Toplevel(self.root)
        ImageAnnotatorWindow(win)
        self.annot_window = win

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
