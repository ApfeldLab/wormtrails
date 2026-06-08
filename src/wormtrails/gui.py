import os
os.environ.setdefault('TK_SILENCE_DEPRECATION', '1')

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
import threading

import wormtrails as wts
from wormtrails.processing import create_time_encoded_frame, fit_pixel_linear_model, subtract_average


class ProgressLabel(tk.Label):
    """Label that can be updated from a background thread via self.after()."""
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.pack(pady=(2, 0))

    def set_text(self, text):
        self.after(0, lambda: self.config(text=text))


class CollapsibleFrame(ttk.Frame):
    """A frame that can be collapsed/expanded via a toggle button."""
    def __init__(self, parent, title, **kwargs):
        super().__init__(parent, **kwargs)
        self._visible = True
        self._content = None

        hdr = ttk.Frame(self)
        hdr.pack(fill='x')
        self._toggle_btn = ttk.Button(hdr, text=f"[-] {title}", command=self.toggle)
        self._toggle_btn.pack(side='left', padx=(2, 0))
        ttk.Separator(hdr, orient='horizontal').pack(fill='x', pady=(0, 2))

        self._content = ttk.Frame(self)
        self._content.pack(fill='both', expand=True)

    def toggle(self):
        self._visible = not self._visible
        state = 'normal' if self._visible else 'hidden'
        self._content.pack_forget() if not self._visible else self._content.pack(fill='both', expand=True)
        self._toggle_btn.config(text=f"{'[-]' if self._visible else '[+]'} " + self._toggle_btn.cget('text')[4:])

    def row_configure(self, idx, **kw):
        self._content.rowconfigure(idx, **kw)

    def column_configure(self, idx, **kw):
        self._content.columnconfigure(idx, **kw)

    def _grid(self, widget, row, column, **kw):
        widget.grid(in_=self._content, row=row, column=column, **kw)

    def add_label_entry(self, label_text, var, row, col=1, **kw):
        ttk.Label(self._content, text=label_text).grid(row=row, column=0, padx=(6, 0), pady=3, sticky='e')
        ttk.Entry(self._content, textvariable=var, width=10).grid(row=row, column=col, padx=4, pady=3, sticky='w')

    def add_checkbutton(self, label_text, var, row, **kw):
        ttk.Checkbutton(self._content, text=label_text, variable=var).grid(row=row, column=0, columnspan=2, padx=6, pady=2, sticky='w', **kw)

    def add_combobox(self, label_text, var, values, row, **kw):
        ttk.Label(self._content, text=label_text).grid(row=row, column=0, padx=(6, 0), pady=3, sticky='e')
        cb = ttk.Combobox(self._content, textvariable=var, values=values, width=8, state='readonly')
        cb.grid(row=row, column=1, padx=4, pady=3, sticky='w')
        if var.get() not in values and values:
            var.set(values[0])
        return cb


def _safe_int(val, default):
    """Parse int from string, returning default if empty/invalid."""
    try:
        return int(val) if val else default
    except (ValueError, TypeError):
        return default

def _safe_float(val, default):
    """Parse float from string, returning default if empty/invalid."""
    try:
        return float(val) if val else default
    except (ValueError, TypeError):
        return default


class WormtrailsGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wormtrails Analysis & Visualization")
        self.geometry("720x700")

        self.video_path = tk.StringVar()
        self._motion = None
        self._te_params = None
        self.create_widgets()

    def create_widgets(self):
        # Top Frame for file loading
        file_frame = tk.Frame(self)
        file_frame.pack(pady=6, fill='x', padx=10)

        tk.Label(file_frame, text="Video File:").pack(side='left')
        tk.Entry(file_frame, textvariable=self.video_path, width=45).pack(side='left', padx=5)
        tk.Button(file_frame, text="Browse", command=self.browse_file).pack(side='left')

        # Status bar
        self.status_bar = tk.Label(self, text="Ready", anchor='w', bd=1, relief='sunken')
        self.status_bar.pack(side='bottom', fill='x')

        # Notebook for modes
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=6)

        # Tab 1: Visualizations
        self.tab_vis = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_vis, text="Visualizations")
        self.setup_visualization_tab()

        # Tab 2: Count Video
        self.tab_count = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_count, text="Count Worms")
        self.setup_count_tab()

        # Tab 3: Measure Chemotaxis
        self.tab_chemotaxis = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_chemotaxis, text="Measure Chemotaxis")
        self.setup_chemotaxis_tab()

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.avi *.mp4 *.mkv"), ("All Files", "*.*")])
        if path:
            self.video_path.set(path)

    # ---------------------------
    #  Wrapper for logic execution
    # ---------------------------
    def measure_action_wrapper(self, action_func, status_msg="Processing..."):
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file first.")
            return

        def run_thread():
            try:
                self.after(0, lambda s=status_msg: self.status_bar.config(text=s))
                action_func()
                self.after(0, lambda: self.status_bar.config(text="Done"))
            except Exception as ex:
                self.after(0, lambda: self.status_bar.config(text="Error"))
                self.after(0, lambda ex=ex: messagebox.showerror("Execution Error", str(ex)))

        threading.Thread(target=run_thread, daemon=True).start()

    # --- Vis Tab Setup ---
    def setup_visualization_tab(self):
        desc = tk.Label(self.tab_vis, text="Create previews of the video using different processing pipelines.", wraplength=600, justify="left")
        desc.pack(pady=6, anchor='w', padx=10)

        # Vignetting correction params
        vig = CollapsibleFrame(self.tab_vis, "Vignetting Correction")
        self._vig_kernel = tk.StringVar(value="")
        self._vig_median = tk.BooleanVar(value=False)
        vig.add_label_entry("Kernel Size (blank=auto):", self._vig_kernel, row=0)
        vig.add_checkbutton("Use Median Blur", self._vig_median, row=1)
        vig.pack(fill='x', padx=6, pady=2)

        # Subtract average params
        sub = CollapsibleFrame(self.tab_vis, "Subtract Average / Motion Detection")
        self._vis_motion_method = tk.StringVar(value="subtract_average")
        self._sub_start = tk.StringVar(value="0")
        self._sub_end = tk.StringVar(value="")
        self._sub_abs = tk.BooleanVar(value=True)
        self._sub_proj = tk.BooleanVar(value=False)
        self._sub_light = tk.BooleanVar(value=True)
        sub.add_combobox("Motion Method:", self._vis_motion_method,
                         ["subtract_average", "linear_model_residuals"], row=0)
        sub.add_label_entry("Average Start:", self._sub_start, row=1)
        sub.add_label_entry("Average End (blank=last):", self._sub_end, row=2)
        sub.add_checkbutton("Absolute Difference", self._sub_abs, row=3)
        sub.add_checkbutton("Use Projection", self._sub_proj, row=4)
        sub.add_checkbutton("Light Background", self._sub_light, row=5)
        sub.pack(fill='x', padx=6, pady=2)

        # Time encoding params
        te = CollapsibleFrame(self.tab_vis, "Time Encoding")
        self._te_colormap = tk.StringVar(value="blue_to_red")
        self._te_window = tk.StringVar(value="20")
        self._te_scale = tk.StringVar(value="1")
        self._te_offset = tk.StringVar(value="0")
        self._te_start_frame = tk.StringVar(value="0")
        self._te_light = tk.BooleanVar(value=True)
        te.add_combobox("Colormap:", self._te_colormap,
                        ["blue_to_red", "white_to_black", "black_to_white", "banded_blue_to_red", "dark_separated_blue_to_red", "middle_grey_last_black", "hsv_rainbow"], row=0)
        te.add_label_entry("Window:", self._te_window, row=1)
        te.add_label_entry("Scale Factor:", self._te_scale, row=2)
        te.add_label_entry("Offset:", self._te_offset, row=3)
        te.add_label_entry("Start Frame:", self._te_start_frame, row=4)
        te.add_checkbutton("Light Background", self._te_light, row=5)
        te.pack(fill='x', padx=6, pady=2)

        # Track array params
        tr = CollapsibleFrame(self.tab_vis, "Track Array")
        self._tr_window = tk.StringVar(value="20")
        tr.add_label_entry("Window:", self._tr_window, row=0)
        tr.pack(fill='x', padx=6, pady=2)

        # Progress
        self.vis_progress = ProgressLabel(self.tab_vis)

        # Buttons
        btn_frame = tk.Frame(self.tab_vis)
        btn_frame.pack(pady=8)
        tk.Button(btn_frame, text="Play Original Video", width=28,
                   command=lambda: self.measure_action_wrapper(self._do_show_video, "Playing video...")).pack(side='left', padx=5)
        tk.Button(btn_frame, text="Preview Time Encoding", width=28,
                   command=lambda: self.measure_action_wrapper(self._do_preview_time_encoding, "Creating preview...")).pack(side='left', padx=5)
        tk.Button(btn_frame, text="Save Frame as Image", width=28,
                   command=self._do_save_frame).pack(side='left', padx=5)
        tk.Button(btn_frame, text="Save as Video", width=28,
                   command=self._do_save_video).pack(side='left', padx=5)
        tk.Button(btn_frame, text="Show Track Array", width=28,
                   command=lambda: self.measure_action_wrapper(self._do_track_array, "Creating track array...")).pack(side='left', padx=5)

    def _get_vig_params(self):
        kernel = _safe_int(self._vig_kernel.get(), None)
        return {
            'kernel_size': kernel,
            'use_median_blur': self._vig_median.get(),
        }

    def _get_vis_sub_params(self):
        start = _safe_int(self._sub_start.get(), 0)
        end = _safe_int(self._sub_end.get(), -1)
        return {
            'average_start': start,
            'average_end': end,
            'use_absolute_difference': self._sub_abs.get(),
            'use_projection': self._sub_proj.get(),
            'light_background': self._sub_light.get(),
        }

    def _compute_motion(self, video, method, sub_params):
        if method == "linear_model_residuals":
            residuals, _, _ = fit_pixel_linear_model(video)
            residuals[residuals > 0] = 0  # only negative residuals for dark worms on light background
            residuals = residuals ** 2
            residuals[residuals > 255] = 255
            return residuals.astype(np.uint8)
        else:
            return subtract_average(video, **sub_params)

    def _get_te_params(self):
        colormap_map = {
            'blue_to_red': wts.blue_to_red,
            'white_to_black': wts.white_to_black,
            'black_to_white': wts.black_to_white,
            'banded_blue_to_red': wts.banded_blue_to_red,
            'dark_separated_blue_to_red': wts.dark_separated_blue_to_red,
            'middle_grey_last_black': wts.middle_grey_last_black,
            'hsv_rainbow': wts.hsv_rainbow,
        }
        cm_name = self._te_colormap.get()
        return {
            'colormap': colormap_map.get(cm_name, wts.blue_to_red),
            'window': _safe_int(self._te_window.get(), 20),
            'scale_factor': _safe_float(self._te_scale.get(), 1),
            'offset': _safe_int(self._te_offset.get(), 0),
            'light_background': self._te_light.get(),
        }

    def _do_show_video(self):
        video = wts.read_video_file(self.video_path.get())
        self.after(0, lambda v=video: wts.show_video_array(v))

    def _do_preview_time_encoding(self):
        video = wts.read_video_file(self.video_path.get())
        vig = self._get_vig_params()
        corrected = wts.correct_vignetting(video, **vig)
        sub = self._get_vis_sub_params()
        method = self._vis_motion_method.get()
        self._motion = self._compute_motion(corrected, method, sub)
        self._te_params = self._get_te_params()
        motion = self._motion
        te_params = self._te_params
        self.after(0, lambda m=motion, p=te_params: wts.show_time_encoding(m, **p))

    def _do_save_frame(self):
        if self._motion is None:
            messagebox.showerror("Error", "Please preview time encoding first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg")],
            title="Save Time Encoded Frame",
        )
        if not path:
            return
        start_frame = _safe_int(self._te_start_frame.get(), 0)
        frame = create_time_encoded_frame(
            self._motion,
            colormap=self._te_params['colormap'],
            window=self._te_params['window'],
            start_time=start_frame,
            scale_factor=self._te_params['scale_factor'],
            offset=self._te_params['offset'],
            light_background=self._te_params['light_background'],
        )
        cv2.imwrite(path, frame)
        messagebox.showinfo("Success", f"Saved frame to {path}")

    def _do_save_video(self):
        if self._motion is None:
            messagebox.showerror("Error", "Please preview time encoding first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 Files", "*.mp4"), ("AVI Files", "*.avi")],
            title="Save Time Encoded Video",
        )
        if not path:
            return
        trails = wts.create_time_encoded_array(
            self._motion,
            colormap=self._te_params['colormap'],
            window=self._te_params['window'],
            scale_factor=self._te_params['scale_factor'],
            offset=self._te_params['offset'],
            light_background=self._te_params['light_background'],
        )
        if path.endswith('.avi'):
            wts.write_avi(trails, path)
        else:
            wts.write_mp4(trails, path)

    def _do_track_array(self):
        video = wts.read_video_file(self.video_path.get())
        vig = self._get_vig_params()
        corrected = wts.correct_vignetting(video, **vig)
        sub = self._get_vis_sub_params()
        method = self._vis_motion_method.get()
        motion = self._compute_motion(corrected, method, sub)
        window = _safe_int(self._tr_window.get(), 20)
        tracks = wts.create_track_array(motion, window=window)
        self.after(0, lambda t=tracks: wts.show_video_array(t))

    # --- Count Tab Setup ---
    def setup_count_tab(self):
        desc = tk.Label(self.tab_count, text="Count the number of living worms in the selected video using motion detection.", wraplength=600, justify="left")
        desc.pack(pady=6, anchor='w', padx=10)

        # Basic params
        basic = ttk.LabelFrame(self.tab_count, text="Basic Parameters")
        basic.pack(fill='x', padx=10, pady=4)
        ttk.Label(basic, text="Min Worm Area:").grid(row=0, column=0, padx=6, pady=4, sticky='e')
        self.count_min = tk.IntVar(value=20)
        ttk.Entry(basic, textvariable=self.count_min, width=10).grid(row=0, column=1, padx=4, pady=4, sticky='w')

        ttk.Label(basic, text="Max Worm Area:").grid(row=1, column=0, padx=6, pady=4, sticky='e')
        self.count_max = tk.IntVar(value=300)
        ttk.Entry(basic, textvariable=self.count_max, width=10).grid(row=1, column=1, padx=4, pady=4, sticky='w')

        # Advanced params (collapsible)
        adv = CollapsibleFrame(self.tab_count, "Advanced Parameters")
        self._count_mwl = tk.StringVar(value="30")
        self._count_wks = tk.StringVar(value="11")
        self._count_wt = tk.StringVar(value="5")
        self._count_mt = tk.StringVar(value="")  # empty = auto (None -> Otsu)
        self._count_smt = tk.StringVar(value="")  # empty = auto
        self._count_smd = tk.StringVar(value="1")
        self._count_sd = tk.StringVar(value="1")
        self._count_mr = tk.StringVar(value="375")
        self._count_vis = tk.BooleanVar(value=True)
        adv.add_label_entry("Max Worm Length:", self._count_mwl, row=0)
        adv.add_label_entry("Worm Kernel Size:", self._count_wks, row=1)
        adv.add_label_entry("Worm Thresh:", self._count_wt, row=2)
        adv.add_label_entry("Motion Thresh (blank=auto):", self._count_mt, row=3)
        adv.add_label_entry("Strict Motion Thresh (blank=auto):", self._count_smt, row=4)
        adv.add_label_entry("Strict Motion Dilation:", self._count_smd, row=5)
        adv.add_label_entry("Stationary Dilation:", self._count_sd, row=6)
        adv.add_label_entry("Mask Radius:", self._count_mr, row=7)
        adv.add_checkbutton("Return Visualization", self._count_vis, row=8)
        adv.pack(fill='x', padx=6, pady=2)

        # Progress
        self.count_progress = ProgressLabel(self.tab_count)

        # Result and button
        result_frame = tk.Frame(self.tab_count)
        result_frame.pack(pady=8)
        self.count_result = tk.StringVar(value="")
        ttk.Label(result_frame, textvariable=self.count_result, font=('Arial', 13, 'bold'), foreground="blue").pack()
        ttk.Button(result_frame, text="Count Worms",
                   command=lambda: self.measure_action_wrapper(self._do_count, "Counting worms...")).pack(pady=4)
        ttk.Button(result_frame, text="Count Assist (Manual)",
                   command=lambda: self.measure_action_wrapper(self._do_count_assist, "Preparing Count Assist...")).pack(pady=4)

    def _get_count_params(self):
        return {
            'min_worm_area': self.count_min.get(),
            'max_worm_area': self.count_max.get(),
            'max_worm_length': _safe_int(self._count_mwl.get(), 30),
            'worm_kernel_size': _safe_int(self._count_wks.get(), 11),
            'worm_thresh': _safe_int(self._count_wt.get(), 5),
            'motion_thresh': _safe_int(self._count_mt.get(), None),
            'strict_motion_thresh': _safe_int(self._count_smt.get(), None),
            'strict_motion_dilation': _safe_int(self._count_smd.get(), 1),
            'stationary_dilation': _safe_int(self._count_sd.get(), 1),
            'mask_radius': _safe_int(self._count_mr.get(), 375),
            'return_vis': self._count_vis.get(),
        }

    def _do_count(self):
        def update_result(val):
            self.count_result.set(val)

        video = wts.read_video_file(self.video_path.get())
        n_roaming, n_stationary, vis = wts.count_video(video, **self._get_count_params())

        self.after(0, lambda: update_result(f"Roaming: {n_roaming}   Stationary: {n_stationary}"))

        if self._count_vis.get():
            self.after(0, lambda: wts.show_video_array(vis))

    def _do_count_assist(self):
        video = wts.read_video_file(self.video_path.get())
        import os
        filename = os.path.basename(self.video_path.get())
        def run():
            markers = wts.count_assist(video, window_name=filename)
            if markers is not None:
                self.count_result.set(f"Manual Count: {len(markers)}")
        self.after(0, run)

    # --- Chemotaxis Tab Setup ---
    def setup_chemotaxis_tab(self):
        desc = tk.Label(self.tab_chemotaxis, text="Measure trajectory, speed, and relative angle towards a bait spot over time windows.", wraplength=600, justify="left")
        desc.pack(pady=6, anchor='w', padx=10)

        # Preprocessing params
        prep = CollapsibleFrame(self.tab_chemotaxis, "Preprocessing")
        self._chem_motion_method = tk.StringVar(value="subtract_average")
        self._chem_vig_k = tk.StringVar(value="")
        self._chem_vig_m = tk.BooleanVar(value=False)
        self._chem_sub_s = tk.StringVar(value="0")
        self._chem_sub_e = tk.StringVar(value="")
        self._chem_sub_a = tk.BooleanVar(value=True)
        self._chem_sub_p = tk.BooleanVar(value=False)
        self._chem_sub_l = tk.BooleanVar(value=True)
        self._chem_thresh = tk.StringVar(value="30")
        prep.add_combobox("Motion Method:", self._chem_motion_method,
                          ["subtract_average", "linear_model_residuals"], row=0)
        prep.add_label_entry("Vignetting Kernel (blank=auto):", self._chem_vig_k, row=1)
        prep.add_checkbutton("Vignetting: Median Blur", self._chem_vig_m, row=2)
        prep.add_label_entry("Subtract Avg Start:", self._chem_sub_s, row=3)
        prep.add_label_entry("Subtract Avg End (blank=last):", self._chem_sub_e, row=4)
        prep.add_checkbutton("Subtract: Absolute Diff", self._chem_sub_a, row=5)
        prep.add_checkbutton("Subtract: Use Projection", self._chem_sub_p, row=6)
        prep.add_checkbutton("Subtract: Light Background", self._chem_sub_l, row=7)
        prep.add_label_entry("Threshold Value:", self._chem_thresh, row=8)
        prep.pack(fill='x', padx=6, pady=2)

        # Analysis params
        anal = ttk.LabelFrame(self.tab_chemotaxis, text="Analysis Parameters")
        anal.pack(fill='x', padx=10, pady=4)
        ttk.Label(anal, text="Time Window (frames):").grid(row=0, column=0, padx=6, pady=4, sticky='e')
        self.chemo_window = tk.IntVar(value=10)
        ttk.Entry(anal, textvariable=self.chemo_window, width=10).grid(row=0, column=1, padx=4, pady=4, sticky='w')

        ttk.Label(anal, text="Interval (frames):").grid(row=1, column=0, padx=6, pady=4, sticky='e')
        self.chemo_int = tk.IntVar(value=60)
        ttk.Entry(anal, textvariable=self.chemo_int, width=10).grid(row=1, column=1, padx=4, pady=4, sticky='w')

        ttk.Label(anal, text="Min Size (px):").grid(row=2, column=0, padx=6, pady=4, sticky='e')
        self.chemo_min = tk.IntVar(value=10)
        ttk.Entry(anal, textvariable=self.chemo_min, width=10).grid(row=2, column=1, padx=4, pady=4, sticky='w')

        ttk.Label(anal, text="Max Size (px):").grid(row=3, column=0, padx=6, pady=4, sticky='e')
        self.chemo_max = tk.IntVar(value=1000)
        ttk.Entry(anal, textvariable=self.chemo_max, width=10).grid(row=3, column=1, padx=4, pady=4, sticky='w')

        # Bait spot
        bait_frame = ttk.LabelFrame(self.tab_chemotaxis, text="Bait Spot (optional)")
        bait_frame.pack(fill='x', padx=10, pady=4)
        ttk.Label(bait_frame, text="X:").grid(row=0, column=0, padx=6, pady=4, sticky='e')
        self.chemo_bait_x = tk.StringVar(value="")
        ttk.Entry(bait_frame, textvariable=self.chemo_bait_x, width=10).grid(row=0, column=1, padx=4, pady=4, sticky='w')
        ttk.Label(bait_frame, text="Y:").grid(row=0, column=2, padx=(12, 6), pady=4, sticky='e')
        self.chemo_bait_y = tk.StringVar(value="")
        ttk.Entry(bait_frame, textvariable=self.chemo_bait_y, width=10).grid(row=0, column=3, padx=4, pady=4, sticky='w')

        # Progress
        self.chemo_progress = ProgressLabel(self.tab_chemotaxis)

        # Button
        ttk.Button(self.tab_chemotaxis, text="Measure Chemotaxis & Save CSV",
                   command=lambda: self.measure_action_wrapper(self._do_chemo, "Measuring chemotaxis...")).pack(pady=8)

    def _get_chemo_prep_params(self):
        vig_kernel = _safe_int(self._chem_vig_k.get(), None)
        sub_end_raw = self._chem_sub_e.get()
        sub_end = _safe_int(sub_end_raw, -1)
        if sub_end_raw == "":
            sub_end = -1
        return {
            'motion_method': self._chem_motion_method.get(),
            'vig': {'kernel_size': vig_kernel, 'use_median_blur': self._chem_vig_m.get()},
            'sub': {
                'average_start': _safe_int(self._chem_sub_s.get(), 0),
                'average_end': sub_end,
                'use_absolute_difference': self._chem_sub_a.get(),
                'use_projection': self._chem_sub_p.get(),
                'light_background': self._chem_sub_l.get(),
            },
            'threshold': _safe_int(self._chem_thresh.get(), 30),
        }

    def _do_chemo(self):
        video = wts.read_video_file(self.video_path.get())
        prep = self._get_chemo_prep_params()

        corrected = wts.correct_vignetting(video, **prep['vig'])
        motion = self._compute_motion(corrected, prep['motion_method'], prep['sub'])
        thresh_val = prep['threshold']
        motion[motion > thresh_val] = 255
        motion[motion <= thresh_val] = 0

        bx = self.chemo_bait_x.get()
        by = self.chemo_bait_y.get()
        test_spot = None
        if bx.isdigit() and by.isdigit():
            test_spot = (int(by), int(bx))

        df = wts.measure_chemotaxis(
            motion,
            time_window=self.chemo_window.get(),
            interval=self.chemo_int.get(),
            minimum_size=self.chemo_min.get(),
            maximum_size=self.chemo_max.get(),
            test_spot=test_spot,
        )

        def request_save():
            save_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv")],
                title="Save Chemotaxis Data"
            )
            if save_path:
                df.to_csv(save_path, index=False)
                messagebox.showinfo("Success", f"Saved to {save_path}")

        self.after(0, request_save)


def main():
    app = WormtrailsGUI()
    app.mainloop()


if __name__ == '__main__':
    main()
