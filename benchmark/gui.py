"""
Graphical user interface (GUI) for the Video Matting benchmark framework.

Allows visual configuration of:
  - Number of videos (with random selection option).
  - Multi-model selection.
  - Mask-saving option.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import sys
import os
from pathlib import Path

# Add the parent to the path to import the benchmark modules
sys.path.append(str(Path(__file__).parent.parent))

from benchmark.models import MODEL_REGISTRY
from benchmark.runner import run_benchmark
from benchmark.config import VIDEOS_DIR, GROUND_TRUTH_DIR, OUTPUT_DIR, TEMP_RESULTS_DIR

class BenchmarkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎬 Video Matting Benchmark")
        self.root.geometry("600x750")
        self.root.configure(bg="#f5f5f7")

        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Custom styles for a modern look
        self.style.configure("TFrame", background="#f5f5f7")
        self.style.configure("TLabel", background="#f5f5f7", font=("Helvetica", 10))
        self.style.configure("Header.TLabel", font=("Helvetica", 16, "bold"), foreground="#1d1d1f")
        self.style.configure("TCheckbutton", background="#f5f5f7")
        self.style.configure("Action.TButton", font=("Helvetica", 11, "bold"), padding=10)

        self._create_widgets()

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="30")
        main_frame.pack(fill="both", expand=True)

        # ── Header ──
        ttk.Label(main_frame, text="Video Matting Benchmark", style="Header.TLabel").pack(pady=(0, 20))

        # ── Video configuration ──
        video_frame = ttk.LabelFrame(main_frame, text=" Video Configuration ", padding="15")
        video_frame.pack(fill="x", pady=10)

        num_frame = ttk.Frame(video_frame)
        num_frame.pack(fill="x")
        ttk.Label(num_frame, text="Number of videos to process:").pack(side="left")

        self.num_videos_var = tk.StringVar(value="0") # 0 = all
        self.num_entry = ttk.Entry(num_frame, textvariable=self.num_videos_var, width=5)
        self.num_entry.pack(side="left", padx=10)
        ttk.Label(num_frame, text="(0 = all)", font=("Helvetica", 8, "italic")).pack(side="left")

        self.shuffle_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(video_frame, text="Random selection (Shuffle)", variable=self.shuffle_var).pack(anchor="w", pady=(10,0))

        # ── Model selection ──
        model_frame = ttk.LabelFrame(main_frame, text=" Model Selection ", padding="15")
        model_frame.pack(fill="both", expand=True, pady=10)

        # Quick-selection buttons
        btn_frame = ttk.Frame(model_frame)
        btn_frame.pack(fill="x", pady=(0, 10))
        ttk.Button(btn_frame, text="All", command=self._select_all_models, width=10).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="None", command=self._select_no_models, width=10).pack(side="left", padx=2)

        # Scrollable list for models
        list_container = ttk.Frame(model_frame)
        list_container.pack(fill="both", expand=True)

        canvas = tk.Canvas(list_container, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas, style="TFrame")
        self.scrollable_frame.configure(background="white")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        self.model_vars = {}
        for key, cls in MODEL_REGISTRY.items():
            var = tk.BooleanVar(value=True)
            self.model_vars[key] = var
            cb = tk.Checkbutton(
                self.scrollable_frame,
                text=f"{cls().name} ({key})",
                variable=var,
                bg="white",
                activebackground="white",
                anchor="w"
            )
            cb.pack(fill="x", padx=10, pady=2)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # ── Output options ──
        output_frame = ttk.LabelFrame(main_frame, text=" Output Options ", padding="15")
        output_frame.pack(fill="x", pady=10)

        self.save_masks_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            output_frame,
            text="Save masks (output/masks/)",
            variable=self.save_masks_var
        ).pack(anchor="w")

        # ── Launch button ──
        self.run_btn = ttk.Button(
            main_frame,
            text="🚀 LAUNCH BENCHMARK",
            style="Action.TButton",
            command=self._start_benchmark_thread
        )
        self.run_btn.pack(fill="x", pady=20)

        # ── Status bar ──
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w", padding=5)
        status_bar.pack(side="bottom", fill="x")

    def _select_all_models(self):
        for var in self.model_vars.values():
            var.set(True)

    def _select_no_models(self):
        for var in self.model_vars.values():
            var.set(False)

    def _start_benchmark_thread(self):
        # Get selected models
        selected_keys = [key for key, var in self.model_vars.items() if var.get()]
        if not selected_keys:
            messagebox.showwarning("Warning", "Please select at least one model.")
            return

        try:
            num = int(self.num_videos_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid number of videos.")
            return

        # Disable the button during the run
        self.run_btn.config(state="disabled")
        self.status_var.set("Benchmark in progress... (see the terminal for details)")

        # Launch in a thread to avoid blocking the UI
        thread = threading.Thread(target=self._run_benchmark_task, args=(selected_keys, num))
        thread.daemon = True
        thread.start()

    def _run_benchmark_task(self, selected_keys, num_videos):
        try:
            models = [MODEL_REGISTRY[key]() for key in selected_keys]

            # Parameters
            n_videos = num_videos if num_videos > 0 else None
            shuffle = self.shuffle_var.get()
            save = self.save_masks_var.get()

            print("\n" + "═"*50)
            print("🚀 LAUNCH FROM GUI INTERFACE")
            print(f"   Models : {len(models)}")
            print(f"   Videos  : {n_videos or 'All'}")
            print(f"   Random: {shuffle}")
            print(f"   Save masks: {save}")
            print("═"*50 + "\n")

            run_benchmark(
                models=models,
                videos_dir=VIDEOS_DIR,
                gt_dir=GROUND_TRUTH_DIR,
                output_dir=OUTPUT_DIR,
                temp_dir=TEMP_RESULTS_DIR,
                num_videos=n_videos,
                random_selection=shuffle,
                save_masks=save
            )

            self.root.after(0, lambda: self._on_complete("Success", "Benchmark completed!"))
        except Exception as e:
            self.root.after(0, lambda: self._on_complete("Error", f"An error occurred: {str(e)}", is_error=True))

    def _on_complete(self, title, message, is_error=False):
        self.run_btn.config(state="normal")
        self.status_var.set("Completed" if not is_error else "Error")
        if is_error:
            messagebox.showerror(title, message)
        else:
            messagebox.showinfo(title, message)

if __name__ == "__main__":
    root = tk.Tk()
    app = BenchmarkGUI(root)
    root.mainloop()
