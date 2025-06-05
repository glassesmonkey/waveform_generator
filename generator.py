import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg') # <-- 添加这一行，在导入 pyplot 之前设置后端
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import ffmpeg as ffmpeg_python # To avoid conflict with other ffmpeg usage
import os
import tempfile
import threading
import subprocess

# --- FFmpeg Check ---
def check_ffmpeg_installed():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, startupinfo=subprocess.STARTUPINFO(dwFlags=subprocess.STARTF_USESHOWWINDOW) if os.name == 'nt' else None)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

# --- Core Waveform Video Generation Logic ---
def generate_waveform_video(audio_path, output_video_path, style_params, progress_callback):
    try:
        progress_callback(f"开始处理: {os.path.basename(audio_path)}")
        
        y, sr = librosa.load(audio_path, sr=None)
        duration_sec = librosa.get_duration(y=y, sr=sr)
        if duration_sec == 0:
            progress_callback(f"音频文件 {os.path.basename(audio_path)} 时长为0，跳过。")
            return

        fps = style_params.get('fps', 30)
        total_frames = int(duration_sec * fps)
        if total_frames == 0: # Handle very short audio resulting in 0 frames
            progress_callback(f"音频文件 {os.path.basename(audio_path)} 太短，无法生成视频帧，跳过。")
            return

        if np.max(np.abs(y)) > 0:
            y_normalized = y / np.max(np.abs(y))
        else:
            y_normalized = y # Silence

        window_duration_sec = style_params.get('window_duration_sec', 3.0)
        samples_in_window = int(window_duration_sec * sr)
        if samples_in_window == 0: # Ensure samples_in_window is not zero
            progress_callback(f"计算得到的窗口采样数为0，可能采样率或窗口时长设置有问题。跳过 {os.path.basename(audio_path)}。")
            return
            
        waveform_color = style_params.get('waveform_color', 'lime')
        background_color = style_params.get('background_color', 'green')
        line_width = style_params.get('line_width', 1.5)
        template_type = style_params.get('template_type', 'line') # 'line' or 'fill'

        fig, ax = plt.subplots(figsize=style_params.get('figsize', (12, 2.5))) # Smaller height
        fig.patch.set_facecolor(background_color)
        ax.set_facecolor(background_color)
        ax.axis('off')
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlim(0, window_duration_sec)

        x_window_time = np.linspace(0, window_duration_sec, samples_in_window, endpoint=False)
        
        # Artists to animate
        line_artist = None
        fill_artist = None

        if template_type == 'line':
            line_artist, = ax.plot([], [], color=waveform_color, lw=line_width)
        elif template_type == 'fill':
            # Path for fill_between consists of (x, y1) followed by (x_reversed, y2_reversed)
            # Initialize with a dummy path that will be updated
            verts = [np.array([[0,0],[0,0],[0,0]])] # Minimal valid path data
            fill_artist = ax.fill_between(x_window_time[:1], [-0.0, -0.0], [0.0, 0.0], color=waveform_color, alpha=0.7) # Create collection
            # To update fill_between, we need to update the paths of its PolyCollection
            # The actual path data will be set in update_frame.
            # For blitting, the fill_artist needs to be a single artist or we need to manage its components.
            # The object returned by fill_between is a PolyCollection.

        def update_frame(frame_num):
            current_sample_pos = int((frame_num / fps) * sr)
            start_sample = current_sample_pos
            end_sample = start_sample + samples_in_window

            y_segment = np.zeros(samples_in_window) # Initialize with silence
            
            # Valid audio part for the current window
            actual_audio_segment = y_normalized[start_sample : min(end_sample, len(y_normalized))]
            len_to_copy = len(actual_audio_segment)
            y_segment[:len_to_copy] = actual_audio_segment
            
            artists_returned = []

            if template_type == 'line' and line_artist:
                line_artist.set_data(x_window_time, y_segment)
                artists_returned.append(line_artist)
            elif template_type == 'fill' and fill_artist:
                # To update fill_between, we modify the paths of the PolyCollection
                # Path: (x_bottom_left -> x_bottom_right -> x_top_right -> x_top_left)
                y_abs_segment = np.abs(y_segment)
                new_path_verts = np.column_stack((np.concatenate([x_window_time, x_window_time[::-1]]), 
                                                  np.concatenate([-y_abs_segment, y_abs_segment[::-1]])))
                fill_artist.get_paths()[0].vertices = new_path_verts
                artists_returned.append(fill_artist)

            if frame_num % (fps * 5) == 0: # Log progress every 5 seconds of video
                 progress_callback(f"  生成帧: {frame_num}/{total_frames} for {os.path.basename(audio_path)}")
            return artists_returned

        # Use blit=True for performance if artists are correctly managed.
        # For fill_between, if direct path update is tricky, blit=False is safer but slower.
        # With path updates for PolyCollection, blit=True should be feasible.
        ani = FuncAnimation(fig, update_frame, frames=total_frames, blit=True, interval=1000/fps)

        temp_silent_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        
        writer_options = {
            'fps': fps,
            'codec': 'libx264',
            'extra_args': ['-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2'] # Ensure even dimensions
        }
        mov_writer = FFMpegWriter(**writer_options)
        
        progress_callback(f"  正在生成静音视频: {os.path.basename(temp_silent_video)}")
        ani.save(temp_silent_video, writer=mov_writer)
        plt.close(fig)

        progress_callback(f"  正在合并音频...")
        input_video_stream = ffmpeg_python.input(temp_silent_video)
        input_audio_stream = ffmpeg_python.input(audio_path)

        output_dir = os.path.dirname(output_video_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        ffmpeg_python.concat(input_video_stream, input_audio_stream, v=1, a=1)\
            .output(output_video_path, vcodec='copy', acodec='aac', strict='experimental', loglevel="quiet")\
            .overwrite_output()\
            .run()
        
        progress_callback(f"成功创建: {output_video_path}")

    except Exception as e:
        progress_callback(f"处理 {os.path.basename(audio_path)} 时发生错误: {e}")
    finally:
        if 'temp_silent_video' in locals() and os.path.exists(temp_silent_video):
            try:
                os.remove(temp_silent_video)
            except PermissionError: # On Windows, file might still be locked briefly
                progress_callback(f"无法立即删除临时文件: {temp_silent_video}。可以稍后手动删除。")


# --- GUI Application ---
class WaveformApp:
    def __init__(self, root):
        self.root = root
        root.title("音频波形视频生成器")
        root.geometry("700x650")

        if not check_ffmpeg_installed():
            messagebox.showerror("FFmpeg 错误", "未检测到 FFmpeg 或其未在系统 PATH 中。请安装 FFmpeg 并确保其在 PATH 中。")
            # root.destroy() # Optionally close app if ffmpeg is critical from the start
            # return

        self.templates = {
            "滚动线条": {"type": "line", "preview": "scrolling_line_preview.png"},
            "滚动对称填充": {"type": "fill", "preview": "scrolling_fill_preview.png"}
        }
        self.colors = {
            "石灰绿 (Lime)": "lime", "绿色 (Green)": "green", "蓝色 (Blue)": "blue",
            "黑色 (Black)": "black", "白色 (White)": "white", "黄色 (Yellow)": "yellow",
            "青色 (Cyan)": "cyan", "品红 (Magenta)": "magenta"
        }
        self.bg_colors = {
            "绿色 (Green)": "green", "蓝色 (Blue)": "blue", "黑色 (Black)": "black",
            "白色 (White)": "white", "灰色 (Gray)": "gray"
        }

        # Folder Selection
        tk.Label(root, text="选择音频文件夹:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.folder_path_var = tk.StringVar()
        self.folder_entry = tk.Entry(root, textvariable=self.folder_path_var, width=60)
        self.folder_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        tk.Button(root, text="浏览...", command=self.select_folder).grid(row=0, column=2, padx=5, pady=5)

        # Template Selection
        tk.Label(root, text="选择波形模板:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.template_var = tk.StringVar(value=list(self.templates.keys())[0])
        template_menu = ttk.Combobox(root, textvariable=self.template_var, values=list(self.templates.keys()), state="readonly", width=25)
        template_menu.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        template_menu.bind("<<ComboboxSelected>>", self.update_template_preview)

        self.preview_label = tk.Label(root, text="模板预览区") # Placeholder for image
        self.preview_label.grid(row=1, column=2, padx=5, pady=5, rowspan=2, sticky="nsew")
        self.current_preview_image = None # To hold reference to PhotoImage

        # Waveform Color
        tk.Label(root, text="波形颜色:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.wave_color_var = tk.StringVar(value=list(self.colors.keys())[0])
        wave_color_menu = ttk.Combobox(root, textvariable=self.wave_color_var, values=list(self.colors.keys()), state="readonly", width=25)
        wave_color_menu.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        # Background Color
        tk.Label(root, text="背景颜色:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.bg_color_var = tk.StringVar(value=list(self.bg_colors.keys())[0])
        bg_color_menu = ttk.Combobox(root, textvariable=self.bg_color_var, values=list(self.bg_colors.keys()), state="readonly", width=25)
        bg_color_menu.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        # Advanced Options (collapsible frame)
        self.options_frame = ttk.LabelFrame(root, text="高级选项", padding=(10, 5))
        self.options_frame.grid(row=4, column=0, columnspan=3, padx=5, pady=10, sticky="ew")
        
        tk.Label(self.options_frame, text="FPS:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.fps_var = tk.IntVar(value=30)
        tk.Entry(self.options_frame, textvariable=self.fps_var, width=5).grid(row=0, column=1, padx=5, pady=2, sticky="w")

        tk.Label(self.options_frame, text="窗口时长 (秒):").grid(row=0, column=2, padx=5, pady=2, sticky="w")
        self.window_sec_var = tk.DoubleVar(value=3.0)
        tk.Entry(self.options_frame, textvariable=self.window_sec_var, width=5).grid(row=0, column=3, padx=5, pady=2, sticky="w")
        
        tk.Label(self.options_frame, text="线条宽度:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.line_width_var = tk.DoubleVar(value=1.5)
        tk.Entry(self.options_frame, textvariable=self.line_width_var, width=5).grid(row=1, column=1, padx=5, pady=2, sticky="w")
        
        tk.Label(self.options_frame, text="视频宽度:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.video_width_var = tk.IntVar(value=1280) # e.g., 1280 for 1280x270 if height is 2.5 and figsize (12,2.5)
        tk.Entry(self.options_frame, textvariable=self.video_width_var, width=7).grid(row=2, column=1, padx=5, pady=2, sticky="w")

        tk.Label(self.options_frame, text="视频高度:").grid(row=2, column=2, padx=5, pady=2, sticky="w")
        self.video_height_var = tk.IntVar(value=270) # This is approx, figsize controls aspect ratio more
        tk.Entry(self.options_frame, textvariable=self.video_height_var, width=7).grid(row=2, column=3, padx=5, pady=2, sticky="w")
        tk.Label(self.options_frame, text="(提示: 视频尺寸由 Matplotlib 的 figsize 控制，这里仅作参考)").grid(row=3, column=0, columnspan=4, padx=5, pady=2, sticky="w",ipady=2)


        # Start Button
        self.start_button = tk.Button(root, text="开始生成", command=self.start_processing, bg="lightblue", width=15, height=2)
        self.start_button.grid(row=5, column=0, columnspan=3, padx=5, pady=10)

        # Log Area
        tk.Label(root, text="处理日志:").grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.log_area = scrolledtext.ScrolledText(root, width=80, height=15, wrap=tk.WORD, state=tk.DISABLED)
        self.log_area.grid(row=7, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        root.grid_columnconfigure(1, weight=1) # Allow entry and comboboxes to expand
        self.update_template_preview() # Initial preview

    def log_message(self, message):
        if self.root.winfo_exists(): # Check if window still exists
            self.log_area.configure(state=tk.NORMAL)
            self.log_area.insert(tk.END, message + "\n")
            self.log_area.configure(state=tk.DISABLED)
            self.log_area.see(tk.END)
            self.root.update_idletasks() # Ensure GUI updates

    def select_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.folder_path_var.set(folder_selected)
            self.log_message(f"已选择文件夹: {folder_selected}")

    def update_template_preview(self, event=None):
        template_name = self.template_var.get()
        # preview_path = self.templates[template_name]["preview"]
        # try:
        #     # Ensure preview images are in the same directory or provide full path
        #     img = Image.open(preview_path)
        #     img.thumbnail((150, 100)) # Resize for preview
        #     self.current_preview_image = ImageTk.PhotoImage(img)
        #     self.preview_label.config(image=self.current_preview_image, text="")
        # except FileNotFoundError:
        #     self.preview_label.config(image=None, text=f"预览图\n{preview_path}\n未找到")
        # except Exception as e:
        #     self.preview_label.config(image=None, text=f"无法加载预览:\n{e}")
        self.preview_label.config(image=None, text=f"模板:\n{template_name}\n(预览图功能已注释)")


    def start_processing(self):
        folder = self.folder_path_var.get()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("错误", "请选择一个有效的文件夹。")
            return

        if not check_ffmpeg_installed():
            messagebox.showerror("FFmpeg 错误", "未检测到 FFmpeg。请确保已安装并配置在系统 PATH。")
            return

        self.start_button.config(state=tk.DISABLED, text="正在处理...")
        self.log_message("开始处理任务...")

        # This will run in a separate thread to keep GUI responsive
        thread = threading.Thread(target=self._process_folder_thread, args=(folder,), daemon=True)
        thread.start()

    def _process_folder_thread(self, folder_path):
        try:
            audio_extensions = ('.wav', '.mp3', '.flac', '.aac', '.m4a', '.ogg')
            audio_files = [f for f in os.listdir(folder_path) if f.lower().endswith(audio_extensions)]

            if not audio_files:
                self.log_message("在选定文件夹中未找到支持的音频文件。")
                if self.root.winfo_exists(): self.start_button.config(state=tk.NORMAL, text="开始生成")
                return

            output_base_folder = os.path.join(folder_path, "waveform_videos_output")
            os.makedirs(output_base_folder, exist_ok=True)

            selected_template_name = self.template_var.get()
            waveform_color_name = self.wave_color_var.get()
            background_color_name = self.bg_color_var.get()

            # Calculate figsize based on user input for dimensions (aspect ratio)
            # Matplotlib's figsize is in inches. Default DPI is 100.
            # So, width_pixels = width_inches * dpi
            # width_inches = width_pixels / dpi
            dpi = 100 # Matplotlib default
            fig_width_inches = self.video_width_var.get() / dpi
            fig_height_inches = self.video_height_var.get() / dpi


            style_params = {
                "template_type": self.templates[selected_template_name]["type"],
                "waveform_color": self.colors[waveform_color_name],
                "background_color": self.bg_colors[background_color_name],
                "fps": self.fps_var.get(),
                "window_duration_sec": self.window_sec_var.get(),
                "line_width": self.line_width_var.get(),
                "figsize": (fig_width_inches, fig_height_inches)
            }

            for audio_file_name in audio_files:
                if not self.root.winfo_exists(): # Stop if GUI is closed
                    self.log_message("GUI已关闭，处理中止。")
                    break
                
                full_audio_path = os.path.join(folder_path, audio_file_name)
                base_name, _ = os.path.splitext(audio_file_name)
                output_video_name = f"{base_name}_waveform.mp4"
                output_video_path = os.path.join(output_base_folder, output_video_name)

                generate_waveform_video(full_audio_path, output_video_path, style_params, self.log_message)
            
            self.log_message("所有文件处理完毕！")

        except Exception as e:
            self.log_message(f"处理过程中发生严重错误: {e}")
        finally:
            if self.root.winfo_exists(): # Check if window still exists
                 self.start_button.config(state=tk.NORMAL, text="开始生成")


if __name__ == "__main__":
    # # Optional: Helper to generate dummy preview images if you don't have them
    # def create_dummy_preview(filename, text_label):
    #     try:
    #         from PIL import Image, ImageDraw, ImageFont
    #         img = Image.new('RGB', (200, 80), color = (73, 109, 137))
    #         d = ImageDraw.Draw(img)
    #         try:
    #             font = ImageFont.truetype("arial.ttf", 15)
    #         except IOError:
    #             font = ImageFont.load_default()
    #         text_bbox = d.textbbox((0,0), text_label, font=font)
    #         text_width = text_bbox[2] - text_bbox[0]
    #         text_height = text_bbox[3] - text_bbox[1]
    #         d.text(((200-text_width)/2, (80-text_height)/2), text_label, fill=(255,255,0), font=font)
    #         img.save(filename)
    #         print(f"Created dummy preview: {filename}")
    #     except Exception as e:
    #         print(f"Could not create dummy preview {filename}: {e}")
            
    # if not os.path.exists("scrolling_line_preview.png"):
    #      create_dummy_preview("scrolling_line_preview.png", "滚动线条预览")
    # if not os.path.exists("scrolling_fill_preview.png"):
    #      create_dummy_preview("scrolling_fill_preview.png", "滚动填充预览")


    main_root = tk.Tk()
    app = WaveformApp(main_root)
    main_root.mainloop()
