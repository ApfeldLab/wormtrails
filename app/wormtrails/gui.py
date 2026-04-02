import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
import threading

import wormtrails as wts

class WormtrailsGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wormtrails Analysis & Visualization")
        self.geometry("650x550")
        
        self.video_path = tk.StringVar()
        self.create_widgets()

    def create_widgets(self):
        # Top Frame for file loading
        file_frame = tk.Frame(self)
        file_frame.pack(pady=10, fill='x', padx=10)
        
        tk.Label(file_frame, text="Video File:").pack(side='left')
        tk.Entry(file_frame, textvariable=self.video_path, width=45).pack(side='left', padx=5)
        tk.Button(file_frame, text="Browse", command=self.browse_file).pack(side='left')

        # Notebook for modes
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

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
    def measure_action_wrapper(self, action_func):
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file first.")
            return
        
        def run_thread():
            try:
                action_func()
            except Exception as e:
                # Use after to dispatch error to the main thread
                self.after(0, lambda: messagebox.showerror("Execution Error", str(e)))
                
        # Running in a simple thread to avoid blocking the UI completely
        threading.Thread(target=run_thread, daemon=True).start()

    # --- Vis Tab Setup ---
    def setup_visualization_tab(self):
        desc = tk.Label(self.tab_vis, text="Create simple previews of the video and trails.", wraplength=500, justify="left")
        desc.pack(pady=10, anchor='w', padx=10)

        vis_frame = tk.Frame(self.tab_vis)
        vis_frame.pack(pady=10)

        tk.Button(vis_frame, text="Play Original Video", width=30, command=lambda: self.measure_action_wrapper(self._do_show_video)).pack(pady=5)
        tk.Button(vis_frame, text="Show Time Encoded Trails", width=30, command=lambda: self.measure_action_wrapper(self._do_time_encoded)).pack(pady=5)

    def _do_show_video(self):
        video = wts.read_video_file(self.video_path.get())
        wts.show_video_array(video)

    def _do_time_encoded(self):
        video = wts.read_video_file(self.video_path.get())
        corrected = wts.correct_vignetting(video)
        motion = wts.subtract_average(corrected)
        # Using a simple colormap for default
        from wormtrails.src.colormaps import blue_to_red
        trails = wts.create_time_encoded_array(motion, colormap=blue_to_red, window=20)
        wts.show_time_encoding(trails)

    # --- Count Tab Setup ---
    def setup_count_tab(self):
        desc = tk.Label(self.tab_count, text="Count the number of living worms in the selected video using motion detection.", wraplength=500, justify="left")
        desc.pack(pady=10, anchor='w', padx=10)

        frame = tk.Frame(self.tab_count)
        frame.pack(pady=10)
        
        tk.Label(frame, text="Min Size (px):").grid(row=0, column=0, pady=5, padx=5, sticky='e')
        self.count_min = tk.IntVar(value=10)
        tk.Entry(frame, textvariable=self.count_min, width=10).grid(row=0, column=1, sticky='w')

        tk.Label(frame, text="Max Size (px):").grid(row=1, column=0, pady=5, padx=5, sticky='e')
        self.count_max = tk.IntVar(value=300)
        tk.Entry(frame, textvariable=self.count_max, width=10).grid(row=1, column=1, sticky='w')

        self.count_result = tk.StringVar(value="")
        tk.Label(frame, textvariable=self.count_result, font=('Arial', 14, 'bold'), fg="blue").grid(row=2, column=0, columnspan=2, pady=15)

        tk.Button(frame, text="Count Worms", command=lambda: self.measure_action_wrapper(self._do_count)).grid(row=3, column=0, columnspan=2)

    def _do_count(self):
        def update_label(val):
            self.count_result.set(val)
        
        gui_update = lambda text: self.after(0, update_label, text)
        gui_update("Processing... Please wait.")
        
        try:
            video = wts.read_video_file(self.video_path.get())
            count = wts.count_video(video, min_size=self.count_min.get(), max_size=self.count_max.get())
            gui_update(f"Detected Objects: {count}")
        except Exception as e:
            gui_update(f"Error: {e}")
            raise e

    # --- Chemotaxis Tab Setup ---
    def setup_chemotaxis_tab(self):
        desc = tk.Label(self.tab_chemotaxis, text="Measure trajectory, speed, and (optional) relative angle towards a bait spot over time windows.", wraplength=500, justify="left")
        desc.pack(pady=10, anchor='w', padx=10)

        frame = tk.Frame(self.tab_chemotaxis)
        frame.pack(pady=10)

        tk.Label(frame, text="Time Window (frames):").grid(row=0, column=0, pady=5, padx=5, sticky='e')
        self.chemo_window = tk.IntVar(value=10)
        tk.Entry(frame, textvariable=self.chemo_window, width=10).grid(row=0, column=1, sticky='w')

        tk.Label(frame, text="Interval (frames):").grid(row=1, column=0, pady=5, padx=5, sticky='e')
        self.chemo_int = tk.IntVar(value=60)
        tk.Entry(frame, textvariable=self.chemo_int, width=10).grid(row=1, column=1, sticky='w')

        tk.Label(frame, text="Min Size (px):").grid(row=2, column=0, pady=5, padx=5, sticky='e')
        self.chemo_min = tk.IntVar(value=10)
        tk.Entry(frame, textvariable=self.chemo_min, width=10).grid(row=2, column=1, sticky='w')

        tk.Label(frame, text="Max Size (px):").grid(row=3, column=0, pady=5, padx=5, sticky='e')
        self.chemo_max = tk.IntVar(value=1000)
        tk.Entry(frame, textvariable=self.chemo_max, width=10).grid(row=3, column=1, sticky='w')
        
        tk.Label(frame, text="Bait X (optional):").grid(row=4, column=0, pady=5, padx=5, sticky='e')
        self.chemo_bait_x = tk.StringVar(value="")
        tk.Entry(frame, textvariable=self.chemo_bait_x, width=10).grid(row=4, column=1, sticky='w')
        
        tk.Label(frame, text="Bait Y (optional):").grid(row=5, column=0, pady=5, padx=5, sticky='e')
        self.chemo_bait_y = tk.StringVar(value="")
        tk.Entry(frame, textvariable=self.chemo_bait_y, width=10).grid(row=5, column=1, sticky='w')

        tk.Button(frame, text="Measure Chemotaxis & Save CSV", command=lambda: self.measure_action_wrapper(self._do_chemo)).grid(row=6, column=0, columnspan=2, pady=15)

    def _do_chemo(self):
        video = wts.read_video_file(self.video_path.get())
        corrected = wts.correct_vignetting(video)
        
        binary = wts.threshold_array(corrected, threshold=120, dark_objects=True)
        
        test_spot = None
        bx = self.chemo_bait_x.get()
        by = self.chemo_bait_y.get()
        if bx.isdigit() and by.isdigit():
            test_spot = (int(by), int(bx))

        df = wts.measure_chemotaxis(
            binary, 
            time_window=self.chemo_window.get(), 
            interval=self.chemo_int.get(), 
            minimum_size=self.chemo_min.get(), 
            maximum_size=self.chemo_max.get(),
            test_spot=test_spot
        )
        
        def request_save():
            save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")], title="Save Chemotaxis Data")
            if save_path:
                df.to_csv(save_path, index=False)
                messagebox.showinfo("Success", f"Saved to {save_path}")
        
        self.after(0, request_save)

def main():
    app = WormtrailsGUI()
    app.mainloop()

if __name__ == '__main__':
    main()
