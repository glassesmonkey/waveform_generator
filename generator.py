import argparse
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox, colorchooser
import os
import sys
import tempfile
import threading
import subprocess
import time
import math

try:
    import librosa
    import numpy as np
    import cv2
    from scipy import signal
    LIBRARY_IMPORT_ERROR = None
except ImportError as exc:
    librosa = None
    np = None
    cv2 = None
    signal = None
    LIBRARY_IMPORT_ERROR = exc

AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.aac', '.m4a', '.ogg')
DEFAULT_STYLE_NAME = "Classic Rectangles"
DEFAULT_SETTINGS = {
    "fps": 25,
    "n_bands": 64,
    "width": 1280,
    "height": 720,
    "sensitivity": 1.5,
}
SETTING_LIMITS = {
    "fps": (1, 60, "FPS"),
    "n_bands": (8, 256, "Bar count"),
    "width": (320, 3840, "Video width"),
    "height": (240, 2160, "Video height"),
    "sensitivity": (0.5, 3.0, "Sensitivity"),
}

STYLE_TEMPLATES = {
    "Classic Rectangles": {
        "bar_style": "rectangle",
        "background_color": [0, 30, 0],
        "bar_color": [0, 255, 0],
        "gradient_effect": True,
        "description": "Traditional rectangular bars; fast and reliable for electronic music."
    },
    "Modern Rounded": {
        "bar_style": "rounded",
        "background_color": [20, 20, 50],
        "bar_color": [100, 150, 255],
        "gradient_effect": True,
        "description": "Rounded bars with a softer, modern visual style."
    },
    "Tech Dots": {
        "bar_style": "circle",
        "background_color": [30, 30, 30],
        "bar_color": [0, 255, 255],
        "gradient_effect": True,
        "description": "Stacked dots for a futuristic, technical look."
    },
    "Rock Peaks": {
        "bar_style": "triangle",
        "background_color": [50, 0, 0],
        "bar_color": [255, 100, 0],
        "gradient_effect": True,
        "description": "Sharp triangular peaks for rock and heavy music."
    },
    "Symmetric Bars": {
        "bar_style": "symmetric",
        "background_color": [20, 0, 40],
        "bar_color": [255, 0, 255],
        "gradient_effect": True,
        "description": "Bars expand from the center line for a balanced composition."
    },
    "Waterfall Gradient": {
        "bar_style": "waterfall",
        "background_color": [0, 20, 50],
        "bar_color": [0, 200, 255],
        "gradient_effect": True,
        "description": "Layered vertical gradients with a flowing feel."
    },
    "Pulse Breathing": {
        "bar_style": "pulse",
        "background_color": [40, 0, 40],
        "bar_color": [255, 50, 150],
        "gradient_effect": True,
        "description": "A breathing pulse animation for vocals and organic tracks."
    },
    "Neon Glow": {
        "bar_style": "neon",
        "background_color": [0, 0, 0],
        "bar_color": [0, 255, 0],
        "gradient_effect": True,
        "description": "Layered neon outlines for club and dance music."
    },
    "CRT Oscilloscope": {
        "bar_style": "symmetric",
        "background_color": [4, 12, 4],
        "bar_color": [35, 255, 90],
        "highlight_color": [180, 255, 210],
        "gradient_effect": True,
        "grid_effect": True,
        "scanline_effect": True,
        "grid_color": [18, 88, 34],
        "description": "Black-grid CRT look with green oscilloscope energy bars."
    },
    "Cyber Grid": {
        "bar_style": "neon",
        "background_color": [20, 8, 28],
        "bar_color": [255, 245, 40],
        "highlight_color": [255, 55, 210],
        "gradient_effect": True,
        "grid_effect": True,
        "perspective_grid": True,
        "grid_color": [130, 80, 255],
        "description": "Neon cyan and magenta bars over a retro-futuristic perspective grid."
    },
    "Signal Glitch": {
        "bar_style": "rectangle",
        "background_color": [10, 10, 18],
        "bar_color": [245, 255, 60],
        "highlight_color": [255, 45, 220],
        "gradient_effect": True,
        "scanline_effect": True,
        "glitch_effect": True,
        "description": "Audio bars with subtle horizontal jitter and signal-break artifacts."
    },
}


def sanitize_filename_part(value):
    """Return a filesystem-safe name fragment for generated video files."""
    safe_chars = []
    for char in value.lower().replace(" ", "_"):
        if char.isalnum() or char in ("-", "_"):
            safe_chars.append(char)
    return "".join(safe_chars) or "style"


def validate_generation_settings(settings):
    """Return a list of human-readable validation errors for render settings."""
    validation_errors = []
    for key, (minimum, maximum, label) in SETTING_LIMITS.items():
        value = settings[key]
        if not minimum <= value <= maximum:
            validation_errors.append(f"{label} must be between {minimum} and {maximum}.")
    return validation_errors


def get_audio_files(folder_path):
    """List supported audio files in a folder in stable alphabetical order."""
    return sorted(f for f in os.listdir(folder_path) if f.lower().endswith(AUDIO_EXTENSIONS))


def build_style_params(style_name, settings, color_overrides=None):
    """Merge a named visual preset with render settings and optional GUI colors."""
    template = STYLE_TEMPLATES[style_name]
    style_params = template.copy()
    style_params.update({
        "fps": settings["fps"],
        "n_bands": settings["n_bands"],
        "width": settings["width"],
        "height": settings["height"],
        "sensitivity": settings["sensitivity"],
    })
    if color_overrides:
        style_params.update(color_overrides)
    return style_params


def get_missing_dependency_message():
    """Explain missing Python dependencies without hiding the original import name."""
    if LIBRARY_IMPORT_ERROR is None:
        return None
    missing_name = getattr(LIBRARY_IMPORT_ERROR, "name", None) or str(LIBRARY_IMPORT_ERROR)
    return (
        f"Missing or incompatible Python dependency: {missing_name}. "
        "Install project dependencies with: pip install -r requirements.txt"
    )


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
def extract_audio_features(y, sr, n_bands=64, hop_length=512, sensitivity=1.5):
    """
    提取音频的频谱特征，用于驱动能量条
    
    参数:
    - y: 音频信号
    - sr: 采样率
    - n_bands: 频率段数量（能量条数量）
    - hop_length: 跳跃长度，影响时间分辨率
    - sensitivity: 音量敏感度 (0.5-3.0)，值越大对小音量越敏感
    
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
    
    # 使用固定动态范围归一化，更准确反映音量变化
    # 设定合理的dB范围：-60dB到0dB
    db_min = -60.0  # 最小音量阈值
    db_max = 0.0    # 最大音量阈值
    
    # 将dB值限制在合理范围内
    log_mel_clipped = np.clip(log_mel, db_min, db_max)
    
    # 归一化到 0-1 范围
    normalized = (log_mel_clipped - db_min) / (db_max - db_min)
    
    # 添加音量敏感度调整
    normalized = np.power(normalized, 1.0 / sensitivity)
    
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


def draw_grid_effect(frame, width, height, style_params):
    """Draw a low-contrast grid that makes Cyber/CRT presets feel technical."""
    grid_color = style_params.get('grid_color', [35, 70, 35])
    if style_params.get('perspective_grid', False):
        horizon_y = int(height * 0.58)
        bottom_y = height - 1
        center_x = width // 2

        for offset in range(-width, width + 1, max(80, width // 12)):
            cv2.line(frame, (center_x, horizon_y), (center_x + offset, bottom_y), grid_color, 1)

        y = horizon_y
        step = 14
        while y < height:
            cv2.line(frame, (0, y), (width, y), grid_color, 1)
            step = int(step * 1.18) + 1
            y += step
        return

    grid_step = max(32, width // 24)
    for x in range(0, width, grid_step):
        cv2.line(frame, (x, 0), (x, height), grid_color, 1)
    for y in range(0, height, grid_step):
        cv2.line(frame, (0, y), (width, y), grid_color, 1)


def draw_scanline_effect(frame, height):
    """Darken every few rows to mimic a CRT display."""
    frame[0:height:4] = (frame[0:height:4] * 0.55).astype(np.uint8)


def apply_glitch_effect(frame, frame_idx, style_params):
    """Add deterministic horizontal slices so glitch renders are repeatable."""
    height, width = frame.shape[:2]
    if frame_idx % 11 not in (0, 1, 7):
        return frame

    rng = np.random.default_rng(frame_idx)
    glitch_color = style_params.get('highlight_color', [255, 45, 220])
    for _ in range(4):
        y = int(rng.integers(0, max(1, height - 8)))
        slice_height = int(rng.integers(2, max(3, height // 45)))
        shift = int(rng.integers(-width // 30, width // 30 + 1))
        frame[y:y + slice_height] = np.roll(frame[y:y + slice_height], shift, axis=1)
        cv2.line(frame, (0, y), (width, y), glitch_color, 1)
    return frame


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

    if style_params.get('grid_effect', False):
        draw_grid_effect(frame, width, height, style_params)
    
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
        if style_params.get('glitch_effect', False) and frame_idx % 9 == 0:
            x += int(math.sin(frame_idx * 0.73 + i * 1.9) * max(1, spacing))
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
    
    if style_params.get('scanline_effect', False):
        draw_scanline_effect(frame, height)
    if style_params.get('glitch_effect', False):
        frame = apply_glitch_effect(frame, frame_idx, style_params)

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
        progress_callback(f"Starting: {os.path.basename(audio_path)}")
        
        # 1. 加载音频
        start_time = time.time()
        y, sr = librosa.load(audio_path, sr=22050)  # 降低采样率提升速度
        duration_sec = librosa.get_duration(y=y, sr=sr)
        
        if duration_sec == 0:
            progress_callback(f"Audio file {os.path.basename(audio_path)} has zero duration; skipping.")
            return False

        progress_callback(f"  Audio loaded ({duration_sec:.1f}s) in {time.time() - start_time:.1f}s.")
        
        # 2. 提取音频特征
        start_time = time.time()
        fps = style_params.get('fps', 25)  # 降低FPS提升速度
        n_bands = style_params.get('n_bands', 64)
        sensitivity = style_params.get('sensitivity', 1.5)  # 获取敏感度设置
        hop_length = int(sr * (1.0 / fps))  # 确保帧数匹配
        
        features = extract_audio_features(y, sr, n_bands=n_bands, hop_length=hop_length, sensitivity=sensitivity)
        total_frames = features.shape[1]
        
        progress_callback(f"  Audio features extracted ({total_frames} frames) in {time.time() - start_time:.1f}s.")
        
        # 3. 视频参数
        width = style_params.get('width', 1280)
        height = style_params.get('height', 720)
        
        # 4. 创建临时视频文件
        temp_fd, temp_video = tempfile.mkstemp(suffix='.avi')
        os.close(temp_fd)
        
        # 5. 初始化OpenCV视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise Exception("Could not create the temporary video writer.")
        
        progress_callback("  Rendering video frames...")
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
                progress_callback(f"  Render progress: {progress:.1f}% (ETA: {eta:.1f}s)")
        
        out.release()
        
        generation_time = time.time() - start_time
        progress_callback(f"  Video frames rendered in {generation_time:.1f}s.")
        
        # 7. 合并音频
        progress_callback("  Merging audio with video...")
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
            raise Exception(f"FFmpeg error: {result.stderr}")
        
        merge_time = time.time() - start_time
        progress_callback(f"  Audio merged in {merge_time:.1f}s.")
        progress_callback(f"Created: {output_video_path}")
        return True

    except Exception as e:
        progress_callback(f"Error while processing {os.path.basename(audio_path)}: {e}")
        return False
    finally:
        # 清理临时文件
        if 'temp_video' in locals() and os.path.exists(temp_video):
            try:
                os.remove(temp_video)
            except PermissionError:
                progress_callback(f"Could not delete temporary file right away: {temp_video}. You can remove it manually later.")


def process_audio_folder(folder_path, output_folder, selected_template, settings, color_overrides, progress_callback, should_continue=None):
    """Render every supported audio file in a folder and return processing counts."""
    audio_files = get_audio_files(folder_path)
    if not audio_files:
        progress_callback("No supported audio files were found in the selected folder.")
        return {"found": 0, "created": 0, "failed": 0}

    os.makedirs(output_folder, exist_ok=True)

    style_params = build_style_params(selected_template, settings, color_overrides)
    template = STYLE_TEMPLATES[selected_template]

    progress_callback(f"Found {len(audio_files)} audio file(s).")
    progress_callback(f"Using style: {selected_template} ({template['description']})")
    progress_callback(f"Video settings: {style_params['width']}x{style_params['height']} @ {style_params['fps']} FPS, {style_params['n_bands']} bars")
    progress_callback(f"Renderer: {style_params['bar_style']}")
    progress_callback("")

    total_start_time = time.time()
    created_count = 0
    failed_count = 0

    for i, audio_file_name in enumerate(audio_files, 1):
        if should_continue and not should_continue():
            progress_callback("Processing stopped before the remaining files were rendered.")
            break

        progress_callback(f"[{i}/{len(audio_files)}] Processing file: {audio_file_name}")

        full_audio_path = os.path.join(folder_path, audio_file_name)
        base_name, _ = os.path.splitext(audio_file_name)
        output_video_name = f"{base_name}_{sanitize_filename_part(selected_template)}_energy_bars.mp4"
        output_video_path = os.path.join(output_folder, output_video_name)

        file_start_time = time.time()
        created = generate_energy_bar_video(full_audio_path, output_video_path, style_params, progress_callback)
        file_time = time.time() - file_start_time

        if created:
            created_count += 1
        else:
            failed_count += 1
        progress_callback(f"  File finished in {file_time:.1f} seconds.")
        progress_callback("")

    total_time = time.time() - total_start_time
    progress_callback("=" * 80)
    progress_callback(f"All files finished in {total_time:.1f} seconds.")
    progress_callback(f"Output folder: {output_folder}")
    progress_callback(f"Created: {created_count}; Failed or skipped: {failed_count}")
    progress_callback("=" * 80)

    return {"found": len(audio_files), "created": created_count, "failed": failed_count}


# --- GUI Application ---
class WaveformApp:
    def __init__(self, root):
        self.root = root
        self.ui_thread = threading.current_thread()
        root.title("Audio Energy Bar Video Generator v2.2")
        root.geometry("900x800")

        if not check_ffmpeg_installed():
            messagebox.showerror("FFmpeg Error", "FFmpeg was not found in your system PATH. Please install FFmpeg before generating videos.")
        dependency_message = get_missing_dependency_message()
        if dependency_message:
            messagebox.showerror("Missing Dependency", dependency_message)

        self.style_templates = STYLE_TEMPLATES

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
        file_frame = ttk.LabelFrame(main_frame, text="Audio Source", padding=(10, 5))
        file_frame.pack(fill="x", pady=(0, 10))
        
        tk.Label(file_frame, text="Audio folder:", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.folder_path_var = tk.StringVar()
        self.folder_entry = tk.Entry(file_frame, textvariable=self.folder_path_var, width=60)
        self.folder_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        tk.Button(file_frame, text="Browse...", command=self.select_folder).grid(row=0, column=2, padx=5, pady=5)
        file_frame.grid_columnconfigure(1, weight=1)

        # 样式选择区域
        style_frame = ttk.LabelFrame(main_frame, text="Visual Style", padding=(10, 5))
        style_frame.pack(fill="x", pady=(0, 10))
        
        tk.Label(style_frame, text="Style preset:", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.template_var = tk.StringVar(value=list(self.style_templates.keys())[0])
        template_menu = ttk.Combobox(style_frame, textvariable=self.template_var, 
                                   values=list(self.style_templates.keys()), state="readonly", width=25)
        template_menu.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        template_menu.bind("<<ComboboxSelected>>", self.on_template_change)
        
        # 样式描述
        self.style_desc_label = tk.Label(style_frame, text="", fg="blue", wraplength=300)
        self.style_desc_label.grid(row=0, column=2, padx=(20, 5), pady=5, sticky="w")

        # 颜色自定义区域
        color_frame = ttk.LabelFrame(main_frame, text="Colors", padding=(10, 5))
        color_frame.pack(fill="x", pady=(0, 10))
        
        # 背景色
        tk.Label(color_frame, text="Background:", font=("Arial", 9, "bold")).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.bg_color_button = tk.Button(color_frame, text="Choose", width=12, height=2,
                                        command=self.choose_bg_color)
        self.bg_color_button.grid(row=0, column=1, padx=5, pady=5)
        self.bg_color_preview = tk.Label(color_frame, text="Preview", width=10, height=2, relief="sunken")
        self.bg_color_preview.grid(row=0, column=2, padx=5, pady=5)
        
        # 能量条主色
        tk.Label(color_frame, text="Bar color:", font=("Arial", 9, "bold")).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.bar_color_button = tk.Button(color_frame, text="Choose", width=12, height=2,
                                         command=self.choose_bar_color)
        self.bar_color_button.grid(row=1, column=1, padx=5, pady=5)
        self.bar_color_preview = tk.Label(color_frame, text="Preview", width=10, height=2, relief="sunken")
        self.bar_color_preview.grid(row=1, column=2, padx=5, pady=5)
        
        # 高亮色
        tk.Label(color_frame, text="Highlight:", font=("Arial", 9, "bold")).grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.highlight_color_button = tk.Button(color_frame, text="Choose", width=12, height=2,
                                               command=self.choose_highlight_color)
        self.highlight_color_button.grid(row=2, column=1, padx=5, pady=5)
        self.highlight_color_preview = tk.Label(color_frame, text="Preview", width=10, height=2, relief="sunken")
        self.highlight_color_preview.grid(row=2, column=2, padx=5, pady=5)
        
        # 重置颜色按钮
        tk.Button(color_frame, text="Reset preset colors", command=self.reset_colors).grid(row=3, column=1, padx=5, pady=5)

        # 视频设置区域
        settings_frame = ttk.LabelFrame(main_frame, text="Video Settings", padding=(10, 5))
        settings_frame.pack(fill="x", pady=(0, 10))
        
        # 第一行：FPS和能量条数量
        tk.Label(settings_frame, text="Frame rate (FPS):").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.fps_var = tk.IntVar(value=25)
        tk.Entry(settings_frame, textvariable=self.fps_var, width=8).grid(row=0, column=1, padx=5, pady=2, sticky="w")
        
        tk.Label(settings_frame, text="Bar count:").grid(row=0, column=2, padx=(20,5), pady=2, sticky="w")
        self.n_bands_var = tk.IntVar(value=64)
        tk.Entry(settings_frame, textvariable=self.n_bands_var, width=8).grid(row=0, column=3, padx=5, pady=2, sticky="w")
        
        # 第二行：视频尺寸
        tk.Label(settings_frame, text="Video width:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.width_var = tk.IntVar(value=1280)
        tk.Entry(settings_frame, textvariable=self.width_var, width=8).grid(row=1, column=1, padx=5, pady=2, sticky="w")
        
        tk.Label(settings_frame, text="Video height:").grid(row=1, column=2, padx=(20,5), pady=2, sticky="w")
        self.height_var = tk.IntVar(value=720)
        tk.Entry(settings_frame, textvariable=self.height_var, width=8).grid(row=1, column=3, padx=5, pady=2, sticky="w")

        # 第三行：音量敏感度
        tk.Label(settings_frame, text="Sensitivity:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.sensitivity_var = tk.DoubleVar(value=1.5)
        sensitivity_scale = tk.Scale(settings_frame, from_=0.5, to=3.0, resolution=0.1, 
                                   variable=self.sensitivity_var, orient="horizontal", length=100)
        sensitivity_scale.grid(row=2, column=1, padx=5, pady=2, sticky="w")
        
        tk.Label(settings_frame, text="0.5 = lower response, 1.5 = default, 3.0 = higher response", fg="gray", font=("Arial", 8)).grid(row=2, column=2, columnspan=2, padx=5, pady=2, sticky="w")

        # 性能优化提示
        tips_frame = ttk.LabelFrame(main_frame, text="Tips", padding=(10, 5))
        tips_frame.pack(fill="x", pady=(0, 10))
        
        tips_text = """Features: 11 bar styles, Cyber/CRT effects, custom colors, and sensitivity control
• Sensitivity controls how strongly quiet passages move the bars.
• CRT Oscilloscope adds grid lines and scanlines for an old monitor look.
• Cyber Grid adds a neon perspective floor for retro-futuristic clips.
• Signal Glitch adds subtle jitter and signal-break artifacts.
• Rock Peaks uses triangular bars for heavier tracks.
• Performance tip: 720p renders much faster than 1080p."""
        
        tk.Label(tips_frame, text=tips_text, justify="left", fg="blue", font=("Arial", 9)).pack(anchor="w")

        # 开始处理按钮
        self.start_button = tk.Button(main_frame, text="Generate Energy Bar Videos", command=self.start_processing, 
                                    bg="lightblue", width=30, height=2, font=("Arial", 12, "bold"))
        self.start_button.pack(pady=15)

        # 日志区域
        tk.Label(main_frame, text="Processing log:", font=("Arial", 10, "bold")).pack(anchor="w", padx=5)
        self.log_area = scrolledtext.ScrolledText(main_frame, width=100, height=15, wrap=tk.WORD, state=tk.DISABLED)
        self.log_area.pack(fill="both", expand=True, padx=5, pady=5)

        # 初始化颜色显示
        self.update_color_previews()
        self.on_template_change()

    def choose_bg_color(self):
        """选择背景颜色"""
        current_hex = bgr_to_hex(self.current_bg_color)
        color = colorchooser.askcolor(color=current_hex, title="Choose background color")
        if color[1]:  # 如果用户选择了颜色
            self.current_bg_color = hex_to_bgr(color[1])
            self.update_color_previews()

    def choose_bar_color(self):
        """选择能量条颜色"""
        current_hex = bgr_to_hex(self.current_bar_color)
        color = colorchooser.askcolor(color=current_hex, title="Choose bar color")
        if color[1]:
            self.current_bar_color = hex_to_bgr(color[1])
            # 自动更新高亮色
            self.current_highlight_color = [min(255, c + 50) for c in self.current_bar_color]
            self.update_color_previews()

    def choose_highlight_color(self):
        """选择高亮颜色"""
        current_hex = bgr_to_hex(self.current_highlight_color)
        color = colorchooser.askcolor(color=current_hex, title="Choose highlight color")
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
        if threading.current_thread() is not self.ui_thread:
            if self.root.winfo_exists():
                self.root.after(0, self.log_message, message)
            return

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
            self.log_message(f"Selected folder: {folder_selected}")

    def get_validated_settings(self):
        """Read and validate UI settings before starting slow video work."""
        try:
            fps = self.fps_var.get()
            n_bands = self.n_bands_var.get()
            width = self.width_var.get()
            height = self.height_var.get()
            sensitivity = self.sensitivity_var.get()
        except tk.TclError:
            messagebox.showerror("Invalid Settings", "FPS, bar count, width, height, and sensitivity must be numbers.")
            return None

        validation_errors = validate_generation_settings({
            "fps": fps,
            "n_bands": n_bands,
            "width": width,
            "height": height,
            "sensitivity": sensitivity,
        })

        if validation_errors:
            messagebox.showerror("Invalid Settings", "\n".join(validation_errors))
            return None

        return {
            "fps": fps,
            "n_bands": n_bands,
            "width": width,
            "height": height,
            "sensitivity": sensitivity,
            "selected_template": self.template_var.get(),
            "background_color": self.current_bg_color.copy(),
            "bar_color": self.current_bar_color.copy(),
            "highlight_color": self.current_highlight_color.copy(),
        }

    def start_processing(self):
        """开始处理音频文件"""
        folder = self.folder_path_var.get()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Invalid Folder", "Please choose a valid folder that contains audio files.")
            return

        if not check_ffmpeg_installed():
            messagebox.showerror("FFmpeg Error", "FFmpeg was not found. Please install it and make sure it is available in your system PATH.")
            return

        dependency_message = get_missing_dependency_message()
        if dependency_message:
            messagebox.showerror("Missing Dependency", dependency_message)
            return

        settings = self.get_validated_settings()
        if settings is None:
            return

        self.start_button.config(state=tk.DISABLED, text="Processing...")
        self.log_message("=" * 80)
        self.log_message("Starting batch processing...")
        self.log_message("=" * 80)

        # 在后台线程中处理
        thread = threading.Thread(target=self._process_folder_thread, args=(folder, settings), daemon=True)
        thread.start()

    def _process_folder_thread(self, folder_path, settings):
        """后台线程处理文件夹中的音频文件"""
        try:
            output_base_folder = os.path.join(folder_path, "energy_bar_videos_output")
            color_overrides = {
                "background_color": settings["background_color"],
                "bar_color": settings["bar_color"],
                "highlight_color": settings["highlight_color"],
            }
            process_audio_folder(
                folder_path,
                output_base_folder,
                settings["selected_template"],
                settings,
                color_overrides,
                self.log_message,
                should_continue=self.root.winfo_exists,
            )

        except Exception as e:
            self.log_message(f"A serious error occurred while processing: {e}")
        finally:
            if self.root.winfo_exists():
                self.root.after(0, self.start_button.config, {"state": tk.NORMAL, "text": "Generate Energy Bar Videos"})


def create_cli_parser():
    """Build the command-line interface without changing the default GUI launch."""
    parser = argparse.ArgumentParser(
        description="Generate Cyber/CRT audio energy-bar videos from a folder of audio files."
    )
    parser.add_argument("--input", dest="input_folder", required=True, help="Folder containing audio files.")
    parser.add_argument("--style", default=DEFAULT_STYLE_NAME, choices=list(STYLE_TEMPLATES.keys()), help="Visual preset to render.")
    parser.add_argument("--output", help="Output folder. Defaults to energy_bar_videos_output inside the input folder.")
    parser.add_argument("--fps", type=int, default=DEFAULT_SETTINGS["fps"], help="Video frame rate, 1-60.")
    parser.add_argument("--bars", dest="n_bands", type=int, default=DEFAULT_SETTINGS["n_bands"], help="Number of energy bars, 8-256.")
    parser.add_argument("--width", type=int, default=DEFAULT_SETTINGS["width"], help="Video width in pixels, 320-3840.")
    parser.add_argument("--height", type=int, default=DEFAULT_SETTINGS["height"], help="Video height in pixels, 240-2160.")
    parser.add_argument("--sensitivity", type=float, default=DEFAULT_SETTINGS["sensitivity"], help="Audio sensitivity, 0.5-3.0.")
    return parser


def run_cli(argv):
    """Run batch rendering from the terminal and return a process exit code."""
    parser = create_cli_parser()
    args = parser.parse_args(argv)

    input_folder = os.path.abspath(args.input_folder)
    if not os.path.isdir(input_folder):
        parser.error(f"input folder does not exist: {input_folder}")

    dependency_message = get_missing_dependency_message()
    if dependency_message:
        print(f"Error: {dependency_message}", file=sys.stderr)
        return 1

    if not check_ffmpeg_installed():
        print("Error: FFmpeg was not found in your system PATH.", file=sys.stderr)
        return 1

    settings = {
        "fps": args.fps,
        "n_bands": args.n_bands,
        "width": args.width,
        "height": args.height,
        "sensitivity": args.sensitivity,
    }
    validation_errors = validate_generation_settings(settings)
    if validation_errors:
        parser.error("\n".join(validation_errors))

    output_folder = os.path.abspath(args.output) if args.output else os.path.join(input_folder, "energy_bar_videos_output")

    result = process_audio_folder(
        input_folder,
        output_folder,
        args.style,
        settings,
        color_overrides=None,
        progress_callback=print,
    )

    if result["found"] == 0 or result["failed"] > 0:
        return 1
    return 0


def run_gui():
    """Start the original Tkinter desktop app."""
    main_root = tk.Tk()
    app = WaveformApp(main_root)
    main_root.mainloop()
    return 0


def main(argv=None):
    """Use CLI mode when arguments are provided; otherwise open the GUI."""
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        return run_cli(argv)
    return run_gui()


if __name__ == "__main__":
    sys.exit(main())
