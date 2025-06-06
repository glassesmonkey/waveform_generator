import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox, colorchooser
import librosa
import numpy as np
import cv2
import os
import tempfile
import threading
import subprocess
from scipy import signal
import time
import math

# --- FFmpeg Check ---
def check_ffmpeg_installed():
    """检查系统是否安装FFmpeg"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, 
                      startupinfo=subprocess.STARTUPINFO(dwFlags=subprocess.STARTF_USESHOWWINDOW) if os.name == 'nt' else None)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

# --- 颜色工具函数 ---
def hex_to_bgr(hex_color):
    """将十六进制颜色转换为BGR格式"""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return [rgb[2], rgb[1], rgb[0]]  # BGR格式

def bgr_to_hex(bgr_color):
    """将BGR颜色转换为十六进制格式"""
    return f"#{bgr_color[2]:02x}{bgr_color[1]:02x}{bgr_color[0]:02x}"

# --- 音频特征提取 ---
def extract_audio_features(y, sr, n_bands=64, hop_length=512):
    """
    提取音频的频谱特征，用于驱动能量条
    
    参数:
    - y: 音频信号
    - sr: 采样率
    - n_bands: 频率段数量（能量条数量）
    - hop_length: 跳跃长度，影响时间分辨率
    
    返回:
    - 频谱特征矩阵 (n_bands, n_frames)
    """
    # 计算短时傅里叶变换
    stft = librosa.stft(y, hop_length=hop_length, n_fft=2048)
    magnitude = np.abs(stft)
    
    # 将频率轴映射到指定数量的频率段
    mel_basis = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=n_bands, fmin=0, fmax=sr//2)
    mel_spectrogram = np.dot(mel_basis, magnitude)
    
    # 转换为对数刻度
    log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # 归一化到 0-1 范围
    normalized = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min())
    
    return normalized

# --- 不同样式的能量条绘制函数 ---
def draw_rectangle_bars(frame, x, y_start, y_end, x_end, bar_color, highlight_color, style_params):
    """绘制经典矩形能量条"""
    cv2.rectangle(frame, (x, y_start), (x_end, y_end), bar_color, -1)
    if style_params.get('gradient_effect', True):
        bar_height = y_end - y_start
        cv2.rectangle(frame, (x, y_start), (x_end, y_start + max(1, bar_height // 4)), 
                     highlight_color, -1)

def draw_rounded_bars(frame, x, y_start, y_end, x_end, bar_color, highlight_color, style_params):
    """绘制圆角矩形能量条"""
    bar_width = x_end - x
    bar_height = y_end - y_start
    radius = min(bar_width // 4, 8)  # 圆角半径
    
    # 绘制主体矩形
    cv2.rectangle(frame, (x, y_start + radius), (x_end, y_end - radius), bar_color, -1)
    cv2.rectangle(frame, (x + radius, y_start), (x_end - radius, y_end), bar_color, -1)
    
    # 绘制四个圆角
    cv2.circle(frame, (x + radius, y_start + radius), radius, bar_color, -1)
    cv2.circle(frame, (x_end - radius, y_start + radius), radius, bar_color, -1)
    cv2.circle(frame, (x + radius, y_end - radius), radius, bar_color, -1)
    cv2.circle(frame, (x_end - radius, y_end - radius), radius, bar_color, -1)
    
    # 高亮效果
    if style_params.get('gradient_effect', True):
        cv2.rectangle(frame, (x + radius, y_start), (x_end - radius, y_start + max(1, bar_height // 4)), 
                     highlight_color, -1)

def draw_circle_bars(frame, x, y_start, y_end, x_end, bar_color, highlight_color, style_params):
    """绘制圆形点状能量条"""
    bar_width = x_end - x
    bar_height = y_end - y_start
    center_x = x + bar_width // 2
    
    # 计算圆圈数量和间距
    circle_radius = max(2, bar_width // 4)
    circle_spacing = circle_radius * 2 + 2
    num_circles = max(1, bar_height // circle_spacing)
    
    for i in range(num_circles):
        circle_y = y_end - (i + 1) * circle_spacing + circle_radius
        if circle_y >= y_start:
            # 渐变颜色效果
            alpha = 1.0 - (i / max(1, num_circles - 1)) * 0.5
            circle_color = [int(c * alpha) for c in bar_color]
            cv2.circle(frame, (center_x, circle_y), circle_radius, circle_color, -1)
            
            # 高亮圆心
            if style_params.get('gradient_effect', True):
                cv2.circle(frame, (center_x, circle_y), max(1, circle_radius // 2), highlight_color, -1)

def draw_triangle_bars(frame, x, y_start, y_end, x_end, bar_color, highlight_color, style_params):
    """绘制三角形尖峰能量条"""
    bar_width = x_end - x
    center_x = x + bar_width // 2
    
    # 三角形顶点
    points = np.array([
        [center_x, y_start],  # 顶点
        [x, y_end],           # 左下
        [x_end, y_end]        # 右下
    ], np.int32)
    
    cv2.fillPoly(frame, [points], bar_color)
    
    # 高亮边缘
    if style_params.get('gradient_effect', True):
        cv2.polylines(frame, [points], True, highlight_color, 2)

def draw_symmetric_bars(frame, x, y_start, y_end, x_end, bar_color, highlight_color, style_params, center_y):
    """绘制对称双向能量条"""
    bar_height = y_end - y_start
    half_height = bar_height // 2
    
    # 上半部分
    cv2.rectangle(frame, (x, center_y - half_height), (x_end, center_y), bar_color, -1)
    # 下半部分
    cv2.rectangle(frame, (x, center_y), (x_end, center_y + half_height), bar_color, -1)
    
    # 高亮效果
    if style_params.get('gradient_effect', True):
        cv2.rectangle(frame, (x, center_y - half_height), (x_end, center_y - half_height + max(1, half_height // 3)), 
                     highlight_color, -1)
        cv2.rectangle(frame, (x, center_y + half_height - max(1, half_height // 3)), (x_end, center_y + half_height), 
                     highlight_color, -1)

def draw_waterfall_bars(frame, x, y_start, y_end, x_end, bar_color, highlight_color, style_params):
    """绘制瀑布式能量条（从底部向上，有重力感）"""
    bar_width = x_end - x
    bar_height = y_end - y_start
    
    # 创建渐变效果
    steps = max(5, bar_height // 5)
    for i in range(steps):
        step_height = bar_height // steps
        current_y = y_end - (i + 1) * step_height
        alpha = 0.3 + 0.7 * (i / max(1, steps - 1))  # 底部更亮
        step_color = [int(c * alpha) for c in bar_color]
        cv2.rectangle(frame, (x, current_y), (x_end, current_y + step_height), step_color, -1)
    
    # 顶部高亮
    if style_params.get('gradient_effect', True):
        cv2.rectangle(frame, (x, y_start), (x_end, y_start + max(1, bar_height // 6)), highlight_color, -1)

def draw_pulse_bars(frame, x, y_start, y_end, x_end, bar_color, highlight_color, style_params, frame_idx, fps):
    """绘制脉冲式能量条（有呼吸效果）"""
    # 脉冲周期
    pulse_period = 2.0  # 2秒一个周期
    time_in_cycle = (frame_idx / fps) % pulse_period
    pulse_factor = 0.8 + 0.4 * math.sin(2 * math.pi * time_in_cycle / pulse_period)
    
    # 调整大小
    bar_width = x_end - x
    bar_height = y_end - y_start
    adjusted_height = int(bar_height * pulse_factor)
    adjusted_y_start = y_end - adjusted_height
    
    # 绘制脉冲效果
    cv2.rectangle(frame, (x, adjusted_y_start), (x_end, y_end), bar_color, -1)
    
    # 添加外发光效果
    if style_params.get('gradient_effect', True):
        glow_radius = max(1, int(bar_width * 0.2 * pulse_factor))
        for i in range(glow_radius):
            alpha = 0.3 * (1 - i / glow_radius)
            glow_color = [int(c * alpha) for c in highlight_color]
            cv2.rectangle(frame, (x - i, adjusted_y_start - i), (x_end + i, y_end + i), glow_color, 1)

def draw_neon_bars(frame, x, y_start, y_end, x_end, bar_color, highlight_color, style_params):
    """绘制霓虹边框能量条"""
    # 主体
    cv2.rectangle(frame, (x + 2, y_start + 2), (x_end - 2, y_end - 2), bar_color, -1)
    
    # 霓虹边框效果（多层）
    if style_params.get('gradient_effect', True):
        # 外层发光
        cv2.rectangle(frame, (x, y_start), (x_end, y_end), highlight_color, 2)
        # 中层发光
        cv2.rectangle(frame, (x + 1, y_start + 1), (x_end - 1, y_end - 1), 
                     [min(255, c + 50) for c in highlight_color], 1)

# --- 高性能视频生成 ---
def create_energy_bar_frame(features, frame_idx, width, height, style_params):
    """
    创建单个能量条帧
    
    参数:
    - features: 音频特征矩阵
    - frame_idx: 当前帧索引
    - width, height: 视频尺寸
    - style_params: 样式参数
    
    返回:
    - BGR格式的图像数组
    """
    # 创建黑色背景
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 背景颜色
    bg_color = style_params.get('background_color', [0, 50, 0])
    frame[:] = bg_color
    
    n_bands = features.shape[0]
    if frame_idx >= features.shape[1]:
        return frame
    
    current_features = features[:, frame_idx]
    
    # 能量条参数
    bar_width = max(1, width // (n_bands + 1))
    spacing = max(1, width // (n_bands * 2))
    max_bar_height = height * 0.8
    
    # 颜色
    bar_color = style_params.get('bar_color', [0, 255, 0])
    highlight_color = style_params.get('highlight_color', [min(255, c + 50) for c in bar_color])
    
    # 获取样式类型
    bar_style = style_params.get('bar_style', 'rectangle')
    fps = style_params.get('fps', 25)
    
    # 绘制能量条
    for i, energy in enumerate(current_features):
        # 计算条的位置和高度
        x = i * (bar_width + spacing) + spacing
        bar_height = int(energy * max_bar_height)
        
        if bar_height > 0:
            if bar_style == 'symmetric':
                # 对称式需要特殊处理
                center_y = height // 2
                half_height = bar_height // 2
                y_start = center_y - half_height
                y_end = center_y + half_height
                x_end = min(width, x + bar_width)
                draw_symmetric_bars(frame, x, y_start, y_end, x_end, bar_color, highlight_color, 
                                  style_params, center_y)
            else:
                # 其他样式从底部向上（修复坐标计算）
                bottom_margin = height * 0.1  # 底部留10%边距
                y_end = int(height - bottom_margin)
                y_start = max(int(bottom_margin), y_end - bar_height)
                
                # 确保不越界
                y_start = max(0, y_start)
                y_end = min(height, y_end)
                x_end = min(width, x + bar_width)
                
                # 根据样式选择绘制函数
                if bar_style == 'rectangle':
                    draw_rectangle_bars(frame, x, y_start, y_end, x_end, bar_color, highlight_color, style_params)
                elif bar_style == 'rounded':
                    draw_rounded_bars(frame, x, y_start, y_end, x_end, bar_color, highlight_color, style_params)
                elif bar_style == 'circle':
                    draw_circle_bars(frame, x, y_start, y_end, x_end, bar_color, highlight_color, style_params)
                elif bar_style == 'triangle':
                    draw_triangle_bars(frame, x, y_start, y_end, x_end, bar_color, highlight_color, style_params)
                elif bar_style == 'waterfall':
                    draw_waterfall_bars(frame, x, y_start, y_end, x_end, bar_color, highlight_color, style_params)
                elif bar_style == 'pulse':
                    draw_pulse_bars(frame, x, y_start, y_end, x_end, bar_color, highlight_color, 
                                   style_params, frame_idx, fps)
                elif bar_style == 'neon':
                    draw_neon_bars(frame, x, y_start, y_end, x_end, bar_color, highlight_color, style_params)
    
    return frame

def generate_energy_bar_video(audio_path, output_video_path, style_params, progress_callback):
    """
    生成能量条风格的音频可视化视频
    
    主要优化:
    1. 使用OpenCV替代matplotlib，提升渲染速度
    2. 预先计算所有音频特征，避免重复计算
    3. 批量处理帧，减少I/O操作
    4. 优化内存使用
    """
    try:
        progress_callback(f"开始处理: {os.path.basename(audio_path)}")
        
        # 1. 加载音频
        start_time = time.time()
        y, sr = librosa.load(audio_path, sr=22050)  # 降低采样率提升速度
        duration_sec = librosa.get_duration(y=y, sr=sr)
        
        if duration_sec == 0:
            progress_callback(f"音频文件 {os.path.basename(audio_path)} 时长为0，跳过。")
            return
            
        progress_callback(f"  音频加载完成 ({duration_sec:.1f}秒), 耗时: {time.time() - start_time:.1f}秒")
        
        # 2. 提取音频特征
        start_time = time.time()
        fps = style_params.get('fps', 25)  # 降低FPS提升速度
        n_bands = style_params.get('n_bands', 64)
        hop_length = int(sr * (1.0 / fps))  # 确保帧数匹配
        
        features = extract_audio_features(y, sr, n_bands=n_bands, hop_length=hop_length)
        total_frames = features.shape[1]
        
        progress_callback(f"  特征提取完成 ({total_frames}帧), 耗时: {time.time() - start_time:.1f}秒")
        
        # 3. 视频参数
        width = style_params.get('width', 1280)
        height = style_params.get('height', 720)
        
        # 4. 创建临时视频文件
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        
        # 5. 初始化OpenCV视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise Exception("无法创建视频写入器")
        
        progress_callback(f"  开始生成视频帧...")
        start_time = time.time()
        
        # 6. 批量生成帧
        batch_size = 100  # 批处理大小，平衡内存和速度
        for batch_start in range(0, total_frames, batch_size):
            batch_end = min(batch_start + batch_size, total_frames)
            
            # 生成批次内的所有帧
            for frame_idx in range(batch_start, batch_end):
                frame = create_energy_bar_frame(features, frame_idx, width, height, style_params)
                out.write(frame)
            
            # 报告进度
            progress = (batch_end / total_frames) * 100
            if batch_start % (batch_size * 5) == 0:  # 每500帧报告一次
                elapsed = time.time() - start_time
                eta = elapsed * (total_frames - batch_end) / batch_end if batch_end > 0 else 0
                progress_callback(f"  视频生成进度: {progress:.1f}% (预计剩余: {eta:.1f}秒)")
        
        out.release()
        
        generation_time = time.time() - start_time
        progress_callback(f"  视频帧生成完成, 耗时: {generation_time:.1f}秒")
        
        # 7. 合并音频
        progress_callback(f"  正在合并音频...")
        start_time = time.time()
        
        output_dir = os.path.dirname(output_video_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 使用ffmpeg合并音视频
        cmd = [
            'ffmpeg', '-y',  # 覆盖输出文件
            '-i', temp_video,  # 视频输入
            '-i', audio_path,  # 音频输入
            '-c:v', 'libx264',  # 视频编码器
            '-c:a', 'aac',      # 音频编码器
            '-shortest',        # 以较短的流为准
            '-crf', '23',       # 质量控制
            '-preset', 'medium', # 编码速度预设
            output_video_path
        ]
        
        # 在Windows下隐藏控制台窗口
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        result = subprocess.run(cmd, capture_output=True, text=True, startupinfo=startupinfo)
        
        if result.returncode != 0:
            raise Exception(f"FFmpeg错误: {result.stderr}")
        
        merge_time = time.time() - start_time
        progress_callback(f"  音频合并完成, 耗时: {merge_time:.1f}秒")
        progress_callback(f"成功创建: {output_video_path}")
        
    except Exception as e:
        progress_callback(f"处理 {os.path.basename(audio_path)} 时发生错误: {e}")
    finally:
        # 清理临时文件
        if 'temp_video' in locals() and os.path.exists(temp_video):
            try:
                os.remove(temp_video)
            except PermissionError:
                progress_callback(f"无法立即删除临时文件: {temp_video}。可以稍后手动删除。")

# --- GUI Application ---
class WaveformApp:
    def __init__(self, root):
        self.root = root
        root.title("音频能量条视频生成器 v2.1")
        root.geometry("900x800")

        if not check_ffmpeg_installed():
            messagebox.showerror("FFmpeg 错误", "未检测到 FFmpeg 或其未在系统 PATH 中。请安装 FFmpeg 并确保其在 PATH 中。")

        # 预定义的样式模板（包含完整配置）
        self.style_templates = {
            "经典矩形": {
                "bar_style": "rectangle",
                "background_color": [0, 30, 0],
                "bar_color": [0, 255, 0],
                "gradient_effect": True,
                "description": "传统的矩形能量条，适合电子音乐"
            },
            "圆角现代": {
                "bar_style": "rounded",
                "background_color": [20, 20, 50],
                "bar_color": [100, 150, 255],
                "gradient_effect": True,
                "description": "圆角矩形，现代感十足"
            },
            "圆点科技": {
                "bar_style": "circle",
                "background_color": [30, 30, 30],
                "bar_color": [0, 255, 255],
                "gradient_effect": True,
                "description": "圆形点状，科技感强烈"
            },
            "尖峰摇滚": {
                "bar_style": "triangle",
                "background_color": [50, 0, 0],
                "bar_color": [255, 100, 0],
                "gradient_effect": True,
                "description": "三角形尖峰，适合摇滚音乐"
            },
            "对称双向": {
                "bar_style": "symmetric",
                "background_color": [20, 0, 40],
                "bar_color": [255, 0, 255],
                "gradient_effect": True,
                "description": "从中心向上下扩展，对称美感"
            },
            "瀑布渐变": {
                "bar_style": "waterfall",
                "background_color": [0, 20, 50],
                "bar_color": [0, 200, 255],
                "gradient_effect": True,
                "description": "瀑布式渐变效果，动感十足"
            },
            "脉冲呼吸": {
                "bar_style": "pulse",
                "background_color": [40, 0, 40],
                "bar_color": [255, 50, 150],
                "gradient_effect": True,
                "description": "脉冲呼吸效果，有生命力"
            },
            "霓虹发光": {
                "bar_style": "neon",
                "background_color": [0, 0, 0],
                "bar_color": [0, 255, 0],
                "gradient_effect": True,
                "description": "霓虹边框发光，夜店风格"
            }
        }

        # 当前选择的颜色
        self.current_bg_color = [0, 30, 0]
        self.current_bar_color = [0, 255, 0]
        self.current_highlight_color = [50, 255, 50]

        self.init_ui()

    def init_ui(self):
        """初始化用户界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # 文件选择区域
        file_frame = ttk.LabelFrame(main_frame, text="文件选择", padding=(10, 5))
        file_frame.pack(fill="x", pady=(0, 10))
        
        tk.Label(file_frame, text="选择音频文件夹:", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.folder_path_var = tk.StringVar()
        self.folder_entry = tk.Entry(file_frame, textvariable=self.folder_path_var, width=60)
        self.folder_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        tk.Button(file_frame, text="浏览...", command=self.select_folder).grid(row=0, column=2, padx=5, pady=5)
        file_frame.grid_columnconfigure(1, weight=1)

        # 样式选择区域
        style_frame = ttk.LabelFrame(main_frame, text="样式模板", padding=(10, 5))
        style_frame.pack(fill="x", pady=(0, 10))
        
        tk.Label(style_frame, text="选择样式模板:", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.template_var = tk.StringVar(value=list(self.style_templates.keys())[0])
        template_menu = ttk.Combobox(style_frame, textvariable=self.template_var, 
                                   values=list(self.style_templates.keys()), state="readonly", width=25)
        template_menu.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        template_menu.bind("<<ComboboxSelected>>", self.on_template_change)
        
        # 样式描述
        self.style_desc_label = tk.Label(style_frame, text="", fg="blue", wraplength=300)
        self.style_desc_label.grid(row=0, column=2, padx=(20, 5), pady=5, sticky="w")

        # 颜色自定义区域
        color_frame = ttk.LabelFrame(main_frame, text="颜色自定义", padding=(10, 5))
        color_frame.pack(fill="x", pady=(0, 10))
        
        # 背景色
        tk.Label(color_frame, text="背景颜色:", font=("Arial", 9, "bold")).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.bg_color_button = tk.Button(color_frame, text="选择背景色", width=12, height=2,
                                        command=self.choose_bg_color)
        self.bg_color_button.grid(row=0, column=1, padx=5, pady=5)
        self.bg_color_preview = tk.Label(color_frame, text="预览", width=10, height=2, relief="sunken")
        self.bg_color_preview.grid(row=0, column=2, padx=5, pady=5)
        
        # 能量条主色
        tk.Label(color_frame, text="能量条颜色:", font=("Arial", 9, "bold")).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.bar_color_button = tk.Button(color_frame, text="选择条颜色", width=12, height=2,
                                         command=self.choose_bar_color)
        self.bar_color_button.grid(row=1, column=1, padx=5, pady=5)
        self.bar_color_preview = tk.Label(color_frame, text="预览", width=10, height=2, relief="sunken")
        self.bar_color_preview.grid(row=1, column=2, padx=5, pady=5)
        
        # 高亮色
        tk.Label(color_frame, text="高亮颜色:", font=("Arial", 9, "bold")).grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.highlight_color_button = tk.Button(color_frame, text="选择高亮色", width=12, height=2,
                                               command=self.choose_highlight_color)
        self.highlight_color_button.grid(row=2, column=1, padx=5, pady=5)
        self.highlight_color_preview = tk.Label(color_frame, text="预览", width=10, height=2, relief="sunken")
        self.highlight_color_preview.grid(row=2, column=2, padx=5, pady=5)
        
        # 重置颜色按钮
        tk.Button(color_frame, text="重置为模板颜色", command=self.reset_colors).grid(row=3, column=1, padx=5, pady=5)

        # 视频设置区域
        settings_frame = ttk.LabelFrame(main_frame, text="视频设置", padding=(10, 5))
        settings_frame.pack(fill="x", pady=(0, 10))
        
        # 第一行：FPS和能量条数量
        tk.Label(settings_frame, text="帧率 (FPS):").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.fps_var = tk.IntVar(value=25)
        tk.Entry(settings_frame, textvariable=self.fps_var, width=8).grid(row=0, column=1, padx=5, pady=2, sticky="w")
        
        tk.Label(settings_frame, text="能量条数量:").grid(row=0, column=2, padx=(20,5), pady=2, sticky="w")
        self.n_bands_var = tk.IntVar(value=64)
        tk.Entry(settings_frame, textvariable=self.n_bands_var, width=8).grid(row=0, column=3, padx=5, pady=2, sticky="w")
        
        # 第二行：视频尺寸
        tk.Label(settings_frame, text="视频宽度:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.width_var = tk.IntVar(value=1280)
        tk.Entry(settings_frame, textvariable=self.width_var, width=8).grid(row=1, column=1, padx=5, pady=2, sticky="w")
        
        tk.Label(settings_frame, text="视频高度:").grid(row=1, column=2, padx=(20,5), pady=2, sticky="w")
        self.height_var = tk.IntVar(value=720)
        tk.Entry(settings_frame, textvariable=self.height_var, width=8).grid(row=1, column=3, padx=5, pady=2, sticky="w")

        # 性能优化提示
        tips_frame = ttk.LabelFrame(main_frame, text="新功能与优化提示", padding=(10, 5))
        tips_frame.pack(fill="x", pady=(0, 10))
        
        tips_text = """✨ 新功能: 8种能量条样式 + 自定义取色器
• 尖峰摇滚: 三角形设计，适合重金属音乐
• 脉冲呼吸: 动态缩放效果，有生命力
• 霓虹发光: 边框发光效果，夜店风格
• 性能提示: 较小的视频尺寸处理更快（720p比1080p快约40%）"""
        
        tk.Label(tips_frame, text=tips_text, justify="left", fg="blue", font=("Arial", 9)).pack(anchor="w")

        # 开始处理按钮
        self.start_button = tk.Button(main_frame, text="🎵 开始生成能量条视频 🎬", command=self.start_processing, 
                                    bg="lightblue", width=30, height=2, font=("Arial", 12, "bold"))
        self.start_button.pack(pady=15)

        # 日志区域
        tk.Label(main_frame, text="处理日志:", font=("Arial", 10, "bold")).pack(anchor="w", padx=5)
        self.log_area = scrolledtext.ScrolledText(main_frame, width=100, height=15, wrap=tk.WORD, state=tk.DISABLED)
        self.log_area.pack(fill="both", expand=True, padx=5, pady=5)

        # 初始化颜色显示
        self.update_color_previews()
        self.on_template_change()

    def choose_bg_color(self):
        """选择背景颜色"""
        current_hex = bgr_to_hex(self.current_bg_color)
        color = colorchooser.askcolor(color=current_hex, title="选择背景颜色")
        if color[1]:  # 如果用户选择了颜色
            self.current_bg_color = hex_to_bgr(color[1])
            self.update_color_previews()

    def choose_bar_color(self):
        """选择能量条颜色"""
        current_hex = bgr_to_hex(self.current_bar_color)
        color = colorchooser.askcolor(color=current_hex, title="选择能量条颜色")
        if color[1]:
            self.current_bar_color = hex_to_bgr(color[1])
            # 自动更新高亮色
            self.current_highlight_color = [min(255, c + 50) for c in self.current_bar_color]
            self.update_color_previews()

    def choose_highlight_color(self):
        """选择高亮颜色"""
        current_hex = bgr_to_hex(self.current_highlight_color)
        color = colorchooser.askcolor(color=current_hex, title="选择高亮颜色")
        if color[1]:
            self.current_highlight_color = hex_to_bgr(color[1])
            self.update_color_previews()

    def reset_colors(self):
        """重置为当前模板的颜色"""
        template_name = self.template_var.get()
        template = self.style_templates[template_name]
        self.current_bg_color = template["background_color"].copy()
        self.current_bar_color = template["bar_color"].copy()
        self.current_highlight_color = [min(255, c + 50) for c in self.current_bar_color]
        self.update_color_previews()

    def update_color_previews(self):
        """更新颜色预览"""
        # 背景色预览
        bg_hex = bgr_to_hex(self.current_bg_color)
        self.bg_color_preview.config(bg=bg_hex, text="")
        
        # 能量条色预览
        bar_hex = bgr_to_hex(self.current_bar_color)
        self.bar_color_preview.config(bg=bar_hex, text="")
        
        # 高亮色预览
        highlight_hex = bgr_to_hex(self.current_highlight_color)
        self.highlight_color_preview.config(bg=highlight_hex, text="")

    def on_template_change(self, event=None):
        """当模板改变时更新描述和颜色"""
        template_name = self.template_var.get()
        template = self.style_templates[template_name]
        
        # 更新描述
        self.style_desc_label.config(text=template["description"])
        
        # 更新颜色为模板默认值
        self.current_bg_color = template["background_color"].copy()
        self.current_bar_color = template["bar_color"].copy()
        self.current_highlight_color = [min(255, c + 50) for c in self.current_bar_color]
        self.update_color_previews()

    def log_message(self, message):
        """在日志区域添加消息"""
        if self.root.winfo_exists():
            self.log_area.configure(state=tk.NORMAL)
            self.log_area.insert(tk.END, message + "\n")
            self.log_area.configure(state=tk.DISABLED)
            self.log_area.see(tk.END)
            self.root.update_idletasks()

    def select_folder(self):
        """选择音频文件夹"""
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.folder_path_var.set(folder_selected)
            self.log_message(f"已选择文件夹: {folder_selected}")

    def start_processing(self):
        """开始处理音频文件"""
        folder = self.folder_path_var.get()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("错误", "请选择一个有效的文件夹。")
            return

        if not check_ffmpeg_installed():
            messagebox.showerror("FFmpeg 错误", "未检测到 FFmpeg。请确保已安装并配置在系统 PATH。")
            return

        self.start_button.config(state=tk.DISABLED, text="正在处理...")
        self.log_message("=" * 80)
        self.log_message("🎵 开始批量处理任务...")
        self.log_message("=" * 80)

        # 在后台线程中处理
        thread = threading.Thread(target=self._process_folder_thread, args=(folder,), daemon=True)
        thread.start()

    def _process_folder_thread(self, folder_path):
        """后台线程处理文件夹中的音频文件"""
        try:
            # 支持的音频格式
            audio_extensions = ('.wav', '.mp3', '.flac', '.aac', '.m4a', '.ogg')
            audio_files = [f for f in os.listdir(folder_path) if f.lower().endswith(audio_extensions)]

            if not audio_files:
                self.log_message("在选定文件夹中未找到支持的音频文件。")
                return

            # 创建输出文件夹
            output_base_folder = os.path.join(folder_path, "energy_bar_videos_output")
            os.makedirs(output_base_folder, exist_ok=True)

            # 获取样式参数
            selected_template = self.template_var.get()
            template = self.style_templates[selected_template]
            
            # 合并模板和用户自定义设置
            style_params = template.copy()
            style_params.update({
                "fps": self.fps_var.get(),
                "n_bands": self.n_bands_var.get(),
                "width": self.width_var.get(),
                "height": self.height_var.get(),
                "background_color": self.current_bg_color.copy(),
                "bar_color": self.current_bar_color.copy(),
                "highlight_color": self.current_highlight_color.copy()
            })

            self.log_message(f"找到 {len(audio_files)} 个音频文件")
            self.log_message(f"使用样式模板: {selected_template} ({template['description']})")
            self.log_message(f"视频设置: {style_params['width']}x{style_params['height']}@{style_params['fps']}fps, {style_params['n_bands']}条")
            self.log_message(f"样式类型: {style_params['bar_style']}")
            self.log_message("")

            # 处理每个音频文件
            total_start_time = time.time()
            for i, audio_file_name in enumerate(audio_files, 1):
                if not self.root.winfo_exists():
                    self.log_message("GUI已关闭，处理中止。")
                    break
                
                self.log_message(f"[{i}/{len(audio_files)}] 处理文件: {audio_file_name}")
                
                full_audio_path = os.path.join(folder_path, audio_file_name)
                base_name, _ = os.path.splitext(audio_file_name)
                output_video_name = f"{base_name}_{selected_template}_energy_bars.mp4"
                output_video_path = os.path.join(output_base_folder, output_video_name)

                file_start_time = time.time()
                generate_energy_bar_video(full_audio_path, output_video_path, style_params, self.log_message)
                file_time = time.time() - file_start_time
                
                self.log_message(f"  文件处理完成，耗时: {file_time:.1f}秒")
                self.log_message("")
            
            total_time = time.time() - total_start_time
            self.log_message("=" * 80)
            self.log_message(f"🎉 所有文件处理完毕！总耗时: {total_time:.1f}秒")
            self.log_message(f"📁 输出文件夹: {output_base_folder}")
            self.log_message("=" * 80)

        except Exception as e:
            self.log_message(f"❌ 处理过程中发生严重错误: {e}")
        finally:
            if self.root.winfo_exists():
                self.start_button.config(state=tk.NORMAL, text="🎵 开始生成能量条视频 🎬")

if __name__ == "__main__":
    main_root = tk.Tk()
    app = WaveformApp(main_root)
    main_root.mainloop()
