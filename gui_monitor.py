#!/usr/bin/python3
"""Tkinter GUI monitor for run_gs_pipeline_core.sh."""

from __future__ import annotations

import argparse
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass, field
from tkinter import filedialog, messagebox, ttk
from typing import Optional


STAGE_RE = re.compile(r"\[(\d+)/(\d+)\]\s+(.*)")
BRUSH_RE = re.compile(r"^\[brush\]\s+(.*)")
VRAM_RE = re.compile(r"(\d+)\s*,\s*(\d+)")
UI_REFRESH_MS = 250
VRAM_REFRESH_S = 1.0
DEFAULT_STAGE_NAMES = {
    1: "Load input frames",
    2: "COLMAP feature extraction",
    3: "COLMAP matching",
    4: "Sparse mapping",
    5: "Sparse densification",
    6: "Convert sparse model to PLY",
    7: "Export COLMAP text model",
    8: "Dense reconstruction",
    9: "Finalize outputs",
    10: "Brush training/export",
}


@dataclass
class GuiState:
    current_stage: int = 0
    total_stages: int = 9
    stage_label: str = "Starting"
    started_at: float = field(default_factory=time.time)
    stage_started_at: float = field(default_factory=time.time)
    last_vram_at: float = 0.0
    vram_used_mb: Optional[int] = None
    vram_total_mb: Optional[int] = None
    last_line: str = ""
    log_tail: deque[str] = field(default_factory=lambda: deque(maxlen=200))
    stage_names: dict[int, str] = field(default_factory=lambda: dict(DEFAULT_STAGE_NAMES))
    stage_durations: dict[int, float] = field(default_factory=dict)
    return_code: Optional[int] = None
    process_done: bool = False
    cancel_requested: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run run_gs_pipeline_core.sh in a native tkinter GUI."
    )
    parser.add_argument("--title", default="3DGS Pipeline Monitor", help="Window title.")
    parser.add_argument(
        "--workspace",
        help="Workspace passed to the pipeline. Optional here if you pass it after --.",
    )
    parser.add_argument(
        "--no-vram",
        action="store_true",
        help="Disable VRAM sampling via nvidia-smi.",
    )
    parser.add_argument(
        "pipeline_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to run_gs_pipeline.sh. Prefix with -- to separate.",
    )
    return parser.parse_args()


def normalize_pipeline_args(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def sample_vram() -> tuple[Optional[int], Optional[int]]:
    if shutil.which("nvidia-smi") is None:
        return None, None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None

    lines = result.stdout.strip().splitlines()
    if not lines:
        return None, None
    match = VRAM_RE.search(lines[0])
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


class PipelineGui:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.state = GuiState()
        self.forwarded_args = normalize_pipeline_args(self.args.pipeline_args)
        self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.pipeline: Optional[subprocess.Popen[str]] = None
        self.reader_thread: Optional[threading.Thread] = None
        self.pipeline_started: bool = False
        self.workspace_path = self.resolve_workspace_path()

        self.root = tk.Tk()
        self.root.title(args.title)
        self.root.geometry("1140x790")
        self.root.minsize(980, 680)
        self.root.configure(bg="#efefef")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Main.TFrame", background="#efefef")
        style.configure("Hero.TFrame", background="#d9d9d9")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("CardAlt.TFrame", background="#f6f6f6")
        style.configure("Title.TLabel", background="#d9d9d9", foreground="#1e1e1e", font=("DejaVu Serif", 22, "bold"))
        style.configure("HeroSub.TLabel", background="#d9d9d9", foreground="#4d4d4d", font=("DejaVu Sans", 10))
        style.configure("CardTitle.TLabel", background="#ffffff", foreground="#16202c", font=("DejaVu Sans", 11, "bold"))
        style.configure("CardTitleAlt.TLabel", background="#f6f6f6", foreground="#16202c", font=("DejaVu Sans", 11, "bold"))
        style.configure("Value.TLabel", background="#ffffff", foreground="#0f1720", font=("DejaVu Sans", 10))
        style.configure("ValueAlt.TLabel", background="#f6f6f6", foreground="#0f1720", font=("DejaVu Sans", 10))
        style.configure("Muted.TLabel", background="#ffffff", foreground="#66768b", font=("DejaVu Sans", 9))
        style.configure("MutedAlt.TLabel", background="#f6f6f6", foreground="#66768b", font=("DejaVu Sans", 9))
        style.configure(
            "Accent.Horizontal.TProgressbar",
            troughcolor="#d9d9d9",
            background="#2f7cf6",
            bordercolor="#d9d9d9",
            lightcolor="#2f7cf6",
            darkcolor="#2f7cf6",
            thickness=20,
        )
        style.configure(
            "Primary.TButton",
            font=("DejaVu Sans", 10, "bold"),
            padding=(14, 8),
            background="#2f7cf6",
            foreground="#ffffff",
            borderwidth=0,
        )
        style.map("Primary.TButton", background=[("active", "#1d67dc"), ("disabled", "#98b7eb")], foreground=[("disabled", "#eef4ff")])
        style.configure(
            "Danger.TButton",
            font=("DejaVu Sans", 10, "bold"),
            padding=(14, 8),
            background="#d04a3f",
            foreground="#ffffff",
            borderwidth=0,
        )
        style.map("Danger.TButton", background=[("active", "#b53d34"), ("disabled", "#e7b6b0")], foreground=[("disabled", "#fff6f5")])
        style.configure("Segment.TLabelframe", background="#efefef", borderwidth=0, relief="flat")
        style.configure("Segment.TLabelframe.Label", background="#efefef", foreground="#5a5a5a", font=("DejaVu Sans", 9, "bold"))
        style.configure("Segment.TRadiobutton", background="#efefef", foreground="#16202c", font=("DejaVu Sans", 10))
        style.configure(
            "TEntry",
            fieldbackground="#f9fbfd",
            foreground="#0f1720",
            bordercolor="#c8d4e3",
            lightcolor="#c8d4e3",
            darkcolor="#c8d4e3",
            padding=(8, 6),
        )
        style.configure(
            "Treeview",
            background="#ffffff",
            foreground="#16202c",
            fieldbackground="#ffffff",
            rowheight=28,
            bordercolor="#d5deea",
            lightcolor="#d5deea",
            darkcolor="#d5deea",
            font=("DejaVu Sans", 10),
        )
        style.configure(
            "Treeview.Heading",
            background="#eef3f9",
            foreground="#36475b",
            bordercolor="#d5deea",
            lightcolor="#d5deea",
            darkcolor="#d5deea",
            font=("DejaVu Sans", 9, "bold"),
            padding=(6, 6),
        )
        style.map("Treeview", background=[("selected", "#d9e8ff")], foreground=[("selected", "#11335e")])

        self.progress_var = tk.StringVar(value="0 / 9")
        self.percent_var = tk.StringVar(value="0%")
        self.current_var = tk.StringVar(value="Starting")
        self.elapsed_var = tk.StringVar(value="00:00")
        self.stage_elapsed_var = tk.StringVar(value="00:00")
        self.vram_var = tk.StringVar(value="n/a")
        self.input_var = tk.StringVar(value=self.resolve_input_path() or "")
        self.workspace_var = tk.StringVar(value=self.workspace_path or "(from pipeline args)")
        self.latest_var = tk.StringVar(value="Configure options and start the pipeline")
        self.header_hint_var = tk.StringVar(value="Select options below, then click Start Pipeline.")
        self.dense_var = tk.StringVar(value="on" if self.has_flag("--dense") else "off")
        self.brush_auto_var = tk.StringVar(value="on")
        self.brush_viewer_var = tk.StringVar(value="on")

        self.build_ui()

    def resolve_workspace_path(self) -> Optional[str]:
        candidate = self.args.workspace
        if not candidate:
            for idx, arg in enumerate(self.forwarded_args[:-1]):
                if arg == "--workspace":
                    candidate = self.forwarded_args[idx + 1]
                    break
        if not candidate:
            return None
        if os.path.isabs(candidate):
            return candidate
        return os.path.abspath(candidate)

    def resolve_input_path(self) -> Optional[str]:
        candidate = None
        for idx, arg in enumerate(self.forwarded_args[:-1]):
            if arg == "--input":
                candidate = self.forwarded_args[idx + 1]
                break
        if not candidate:
            return None
        if os.path.isabs(candidate):
            return candidate
        return os.path.abspath(candidate)

    def has_flag(self, flag: str) -> bool:
        return flag in self.forwarded_args

    def build_command(self) -> list[str]:
        skip_flags = {"--dense", "--brush-auto", "--brush-with-viewer", "--input", "--workspace"}
        args: list[str] = []
        skip_next = False
        for arg in self.forwarded_args:
            if skip_next:
                skip_next = False
                continue
            if arg in {"--input", "--workspace"}:
                skip_next = True
                continue
            if arg in skip_flags:
                continue
            args.append(arg)
        input_path = self.input_var.get().strip()
        workspace_path = self.workspace_var.get().strip()
        if input_path:
            args.extend(["--input", input_path])
        if workspace_path and workspace_path != "(from pipeline args)":
            args.extend(["--workspace", workspace_path])
        if self.dense_var.get() == "on":
            args.append("--dense")
        if self.brush_auto_var.get() == "on":
            args.append("--brush-auto")
        if self.brush_viewer_var.get() == "on":
            args.append("--brush-with-viewer")
        return ["./run_gs_pipeline_core.sh", *args]

    def build_ui(self) -> None:
        outer = ttk.Frame(self.root, style="Main.TFrame", padding=18)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        hero = ttk.Frame(outer, style="Hero.TFrame", padding=24)
        hero.grid(row=0, column=0, sticky="ew")
        hero.columnconfigure(0, weight=1)
        hero.columnconfigure(1, weight=0)

        title_block = ttk.Frame(hero, style="Hero.TFrame")
        title_block.grid(row=0, column=0, sticky="w")
        ttk.Label(title_block, text="3DGS Pipeline Monitor", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            title_block,
            text="Live reconstruction status, GPU memory, stage timeline, and streaming logs.",
            style="HeroSub.TLabel",
        ).pack(anchor="w", pady=(4, 0))
        ttk.Label(
            title_block,
            textvariable=self.header_hint_var,
            style="HeroSub.TLabel",
            wraplength=560,
            justify="left",
        ).pack(anchor="w", pady=(12, 0))

        hero_right = ttk.Frame(hero, style="Hero.TFrame")
        hero_right.grid(row=0, column=1, sticky="e")
        self.badge = tk.Label(
            hero_right,
            text="READY",
            bg="#ededed",
            fg="#3a3a3a",
            padx=16,
            pady=7,
            font=("DejaVu Sans", 10, "bold"),
        )
        self.badge.pack(anchor="e")
        tk.Label(
            hero_right,
            textvariable=self.percent_var,
            bg="#d9d9d9",
            fg="#232323",
            font=("DejaVu Serif", 26, "bold"),
        ).pack(anchor="e", pady=(12, 0))
        tk.Label(
            hero_right,
            textvariable=self.progress_var,
            bg="#d9d9d9",
            fg="#595959",
            font=("DejaVu Sans", 10),
        ).pack(anchor="e")

        content = ttk.Frame(outer, style="Main.TFrame")
        content.grid(row=1, column=0, sticky="nsew", pady=(14, 0))
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(1, weight=1)

        summary = ttk.Frame(content, style="Card.TFrame", padding=18)
        summary.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        ttk.Label(summary, text="Overview", style="CardTitle.TLabel").pack(anchor="w")

        selectors = ttk.Frame(summary, style="Card.TFrame")
        selectors.pack(fill="x", pady=(12, 10))
        selectors.columnconfigure(1, weight=1)
        selectors.columnconfigure(2, weight=0)
        selectors.columnconfigure(3, weight=0)
        selectors.columnconfigure(4, weight=0)

        ttk.Label(selectors, text="Input", style="Muted.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 8))
        input_entry = ttk.Entry(selectors, textvariable=self.input_var)
        input_entry.grid(row=0, column=1, sticky="ew", pady=(0, 8))
        ttk.Button(selectors, text="File", command=self.pick_input_file).grid(row=0, column=2, sticky="e", padx=(8, 0), pady=(0, 8))
        ttk.Button(selectors, text="Folder", command=self.pick_input_dir).grid(row=0, column=3, sticky="e", padx=(8, 0), pady=(0, 8))

        ttk.Label(selectors, text="Workspace", style="Muted.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 8))
        workspace_entry = ttk.Entry(selectors, textvariable=self.workspace_var)
        workspace_entry.grid(row=1, column=1, sticky="ew")
        ttk.Button(selectors, text="Browse", command=self.pick_workspace_dir).grid(row=1, column=2, sticky="e", padx=(8, 0))

        self.progress = ttk.Progressbar(summary, maximum=100, style="Accent.Horizontal.TProgressbar")
        self.progress.pack(fill="x", pady=(4, 10))

        stats = ttk.Frame(summary, style="Card.TFrame")
        stats.pack(fill="x")
        self.stat_row(stats, "Progress", self.progress_var, 0, 0)
        self.stat_row(stats, "Current stage", self.current_var, 0, 1)
        self.stat_row(stats, "Elapsed", self.elapsed_var, 1, 0)
        self.stat_row(stats, "Stage time", self.stage_elapsed_var, 1, 1)
        self.stat_row(stats, "VRAM", self.vram_var, 2, 0)
        self.stat_row(stats, "Workspace", self.workspace_var, 2, 1)

        latest = ttk.Frame(content, style="CardAlt.TFrame", padding=18)
        latest.grid(row=0, column=1, sticky="nsew", pady=(0, 10))
        ttk.Label(latest, text="Latest Event", style="CardTitleAlt.TLabel").pack(anchor="w")
        ttk.Label(latest, textvariable=self.latest_var, style="ValueAlt.TLabel", wraplength=300, justify="left").pack(anchor="w", pady=(12, 0))
        ttk.Label(
            latest,
            text="The stage label above is the last significant pipeline event parsed from stdout.",
            style="MutedAlt.TLabel",
            wraplength=300,
            justify="left",
        ).pack(anchor="w", pady=(12, 0))

        logs = ttk.Frame(content, style="Card.TFrame", padding=18)
        logs.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        logs.rowconfigure(1, weight=1)
        logs.columnconfigure(0, weight=1)
        ttk.Label(logs, text="Live Logs", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")

        self.log_text = tk.Text(
            logs,
            wrap="word",
            bg="#f2f2f2",
            fg="#2d2d2d",
            insertbackground="#2d2d2d",
            relief="flat",
            font=("DejaVu Sans Mono", 10),
            padx=12,
            pady=12,
        )
        self.log_text.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        self.log_text.tag_configure("stage", foreground="#2f6fb5")
        self.log_text.tag_configure("error", foreground="#b03a2e")
        self.log_text.tag_configure("warning", foreground="#9a6a00")
        self.log_text.tag_configure("success", foreground="#2e7d32")
        self.log_text.tag_configure("info", foreground="#2d2d2d")
        log_scroll = ttk.Scrollbar(logs, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=1, column=1, sticky="ns", pady=(12, 0))
        self.log_text.configure(yscrollcommand=log_scroll.set)

        timeline = ttk.Frame(content, style="Card.TFrame", padding=18)
        timeline.grid(row=1, column=1, sticky="nsew")
        timeline.rowconfigure(1, weight=1)
        timeline.columnconfigure(0, weight=1)
        ttk.Label(timeline, text="Stage Timeline", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.stage_table = ttk.Treeview(
            timeline,
            columns=("idx", "label", "status", "duration"),
            show="headings",
            height=12,
        )
        self.stage_table.heading("idx", text="#")
        self.stage_table.heading("label", text="Stage")
        self.stage_table.heading("status", text="Status")
        self.stage_table.heading("duration", text="Time")
        self.stage_table.column("idx", width=40, anchor="center", stretch=False)
        self.stage_table.column("label", width=230, anchor="w", stretch=True)
        self.stage_table.column("status", width=70, anchor="center", stretch=False)
        self.stage_table.column("duration", width=70, anchor="center", stretch=False)
        self.stage_table.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        self.stage_table["displaycolumns"] = ("idx", "label", "status", "duration")
        stage_scroll = ttk.Scrollbar(timeline, orient="vertical", command=self.stage_table.yview)
        stage_scroll.grid(row=1, column=1, sticky="ns", pady=(12, 0))
        self.stage_table.configure(yscrollcommand=stage_scroll.set)
        timeline.columnconfigure(0, weight=1)
        self.stage_table.tag_configure("done", background="#e9f7ef", foreground="#1a6a3a")
        self.stage_table.tag_configure("now", background="#ecf4ff", foreground="#1e56b5")
        self.stage_table.tag_configure("wait", background="#ffffff", foreground="#526273")
        self.stage_table.tag_configure("fail", background="#fff1f1", foreground="#b5352c")

        controls = ttk.Frame(outer, style="Main.TFrame")
        controls.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        controls.columnconfigure(0, weight=1)
        options = ttk.Frame(controls, style="Main.TFrame")
        options.grid(row=0, column=0, sticky="w")
        self.build_toggle_group(options, "Dense", self.dense_var, 0)
        self.build_toggle_group(options, "Brush Auto", self.brush_auto_var, 1)
        self.build_toggle_group(options, "Brush Viewer", self.brush_viewer_var, 2)

        actions = ttk.Frame(controls, style="Main.TFrame")
        actions.grid(row=0, column=1, sticky="e")
        self.start_button = ttk.Button(
            actions,
            text="Start Pipeline",
            command=self.launch_pipeline,
            style="Primary.TButton",
        )
        self.start_button.grid(row=0, column=0, sticky="e", padx=(0, 8))
        self.cancel_button = ttk.Button(
            actions,
            text="Cancel Pipeline",
            command=self.cancel_pipeline,
            style="Danger.TButton",
        )
        self.cancel_button.grid(row=0, column=1, sticky="e")
        self.cancel_button.configure(state="disabled")

    def build_toggle_group(self, parent: ttk.Frame, label: str, variable: tk.StringVar, column: int) -> None:
        group = ttk.LabelFrame(parent, text=label, style="Segment.TLabelframe", padding=(10, 7))
        group.grid(row=0, column=column, sticky="w", padx=(0, 12))
        ttk.Radiobutton(group, text="On", value="on", variable=variable, style="Segment.TRadiobutton").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(group, text="Off", value="off", variable=variable, style="Segment.TRadiobutton").grid(row=0, column=1, sticky="w", padx=(8, 0))

    def stat_row(self, parent: ttk.Frame, label: str, variable: tk.StringVar, row: int, col: int) -> None:
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.grid(row=row, column=col, sticky="ew", padx=(0 if col == 0 else 12, 0), pady=(0, 12))
        parent.columnconfigure(col, weight=1)
        ttk.Label(frame, text=label, style="Muted.TLabel").pack(anchor="w")
        ttk.Label(frame, textvariable=variable, style="Value.TLabel", wraplength=260, justify="left").pack(anchor="w", pady=(4, 0))

    def start(self) -> int:
        self.root.after(UI_REFRESH_MS, self.poll)
        self.root.mainloop()
        return self.state.return_code or 0

    def pick_input_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select input video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.mkv *.avi *.webm"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.input_var.set(path)
            if not self.workspace_var.get().strip() or self.workspace_var.get().strip() == "(from pipeline args)":
                self.workspace_var.set(os.path.abspath(os.path.splitext(path)[0]))
                self.workspace_path = self.workspace_var.get().strip()

    def pick_input_dir(self) -> None:
        path = filedialog.askdirectory(title="Select input image directory")
        if path:
            self.input_var.set(path)
            if not self.workspace_var.get().strip() or self.workspace_var.get().strip() == "(from pipeline args)":
                self.workspace_var.set(os.path.abspath(path + "_workspace"))
                self.workspace_path = self.workspace_var.get().strip()

    def pick_workspace_dir(self) -> None:
        path = filedialog.askdirectory(title="Select workspace directory")
        if path:
            self.workspace_var.set(path)
            self.workspace_path = path

    def launch_pipeline(self) -> None:
        if self.pipeline_started:
            return
        input_path = self.input_var.get().strip()
        workspace_path = self.workspace_var.get().strip()
        if not input_path:
            messagebox.showerror("Missing Input", "Select an input video file or image directory first.")
            return
        if not os.path.exists(input_path):
            messagebox.showerror("Invalid Input", f"Input path does not exist:\n{input_path}")
            return
        if not workspace_path or workspace_path == "(from pipeline args)":
            messagebox.showerror("Missing Workspace", "Select or enter a workspace directory.")
            return
        self.workspace_path = os.path.abspath(workspace_path)
        self.workspace_var.set(self.workspace_path)
        cmd = self.build_command()
        self.pipeline = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.pipeline_started = True
        self.state.started_at = time.time()
        self.state.stage_started_at = self.state.started_at
        self.state.total_stages = 10 if self.brush_auto_var.get() == "on" else 9
        self.latest_var.set("Pipeline started.")
        self.header_hint_var.set("Pipeline is running.")
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.dense_var.set(self.dense_var.get())

        def read_output() -> None:
            assert self.pipeline is not None and self.pipeline.stdout is not None
            for raw_line in self.pipeline.stdout:
                self.queue.put(("line", raw_line.rstrip("\n")))
            self.queue.put(("done", self.pipeline.wait()))

        self.reader_thread = threading.Thread(target=read_output, daemon=True)
        self.reader_thread.start()

    def poll(self) -> None:
        if not self.pipeline_started:
            self.refresh_ui()
            self.root.after(UI_REFRESH_MS, self.poll)
            return
        while True:
            try:
                event, payload = self.queue.get_nowait()
            except queue.Empty:
                break

            if event == "line":
                self.handle_line(str(payload))
            elif event == "done":
                self.finish(int(payload))
                return

        now = time.time()
        if not self.args.no_vram and now - self.state.last_vram_at >= VRAM_REFRESH_S:
            self.state.vram_used_mb, self.state.vram_total_mb = sample_vram()
            self.state.last_vram_at = now

        self.refresh_ui()
        if not self.state.process_done:
            self.root.after(UI_REFRESH_MS, self.poll)

    def handle_line(self, line: str) -> None:
        self.state.last_line = line
        self.state.log_tail.append(line)
        self.insert_log_line(line)
        self.log_text.see("end")

        match = STAGE_RE.search(line)
        if match:
            new_stage = int(match.group(1))
            total = int(match.group(2))
            label = match.group(3).strip()
            if new_stage != self.state.current_stage:
                if self.state.current_stage > 0:
                    self.state.stage_durations[self.state.current_stage] = time.time() - self.state.stage_started_at
                self.state.stage_started_at = time.time()
            self.state.current_stage = new_stage
            self.state.total_stages = total
            self.state.stage_label = label
            self.state.stage_names[new_stage] = label
            return

        brush_match = BRUSH_RE.search(line)
        if brush_match:
            brush_label = brush_match.group(1).strip()
            brush_stage_num = 10
            if self.state.current_stage != brush_stage_num:
                if self.state.current_stage > 0:
                    self.state.stage_durations[self.state.current_stage] = time.time() - self.state.stage_started_at
                self.state.stage_started_at = time.time()
            self.state.current_stage = brush_stage_num
            self.state.total_stages = max(self.state.total_stages, brush_stage_num)
            self.state.stage_label = f"Brush: {brush_label}"
            self.state.stage_names[brush_stage_num] = "Brush training/export"

    def refresh_ui(self) -> None:
        elapsed = format_duration(time.time() - self.state.started_at)
        stage_elapsed = format_duration(time.time() - self.state.stage_started_at)
        pct = int((self.state.current_stage / max(1, self.state.total_stages)) * 100)

        self.progress["value"] = pct
        self.progress_var.set(f"{self.state.current_stage} / {self.state.total_stages}  ({pct}%)")
        self.percent_var.set(f"{pct}%")
        self.current_var.set(self.state.stage_label if self.pipeline_started else "Ready to start")
        self.elapsed_var.set(elapsed if self.pipeline_started else "00:00")
        self.stage_elapsed_var.set(stage_elapsed if self.pipeline_started else "00:00")
        if self.pipeline_started:
            self.latest_var.set(self.state.last_line or "Waiting for pipeline output")
        if self.pipeline_started:
            self.header_hint_var.set(self.state.stage_label if self.state.current_stage else "Waiting for pipeline output")

        if self.state.vram_used_mb is None or self.state.vram_total_mb is None:
            self.vram_var.set("n/a")
        else:
            self.vram_var.set(f"{self.state.vram_used_mb}/{self.state.vram_total_mb} MB")

        if self.state.process_done:
            if self.state.return_code == 0:
                self.badge.configure(text="DONE", bg="#dff7e8", fg="#117a37")
            elif self.state.cancel_requested:
                self.badge.configure(text="CANCELED", bg="#fff4db", fg="#b45309")
            else:
                self.badge.configure(text=f"FAILED ({self.state.return_code})", bg="#fde7e8", fg="#b42318")
            self.cancel_button.configure(state="disabled")
            self.start_button.configure(state="disabled")
        else:
            if self.state.cancel_requested:
                self.badge.configure(text="CANCELING", bg="#fff4db", fg="#b45309")
                self.header_hint_var.set("Cancel requested. Waiting for subprocesses to stop.")
            else:
                self.badge.configure(text="RUNNING" if self.pipeline_started else "READY", bg="#e6f0ff", fg="#175cd3")

        self.rebuild_stage_list()

    def rebuild_stage_list(self) -> None:
        for item in self.stage_table.get_children():
            self.stage_table.delete(item)
        for stage_num in range(1, self.state.total_stages + 1):
            label = self.state.stage_names.get(stage_num, f"Stage {stage_num}")
            if stage_num < self.state.current_stage:
                prefix = "done"
                tag = "done"
            elif stage_num == self.state.current_stage and not self.state.process_done:
                prefix = "now"
                tag = "now"
            elif stage_num == self.state.current_stage and self.state.process_done and self.state.return_code == 0:
                prefix = "done"
                tag = "done"
            elif self.state.process_done and self.state.return_code not in (0, None) and stage_num == self.state.current_stage:
                prefix = "fail"
                tag = "fail"
            else:
                prefix = "wait"
                tag = "wait"

            duration = ""
            if stage_num in self.state.stage_durations:
                duration = format_duration(self.state.stage_durations[stage_num])
            elif stage_num == self.state.current_stage and not self.state.process_done and self.state.current_stage > 0:
                duration = format_duration(time.time() - self.state.stage_started_at)

            self.stage_table.insert(
                "",
                "end",
                values=(stage_num, label, prefix.upper(), duration or "--"),
                tags=(tag,),
            )

    def insert_log_line(self, line: str) -> None:
        lowered = line.lower()
        if STAGE_RE.search(line):
            tag = "stage"
        elif "error" in lowered or "failed" in lowered:
            tag = "error"
        elif "warning" in lowered or "skipped" in lowered:
            tag = "warning"
        elif "done" in lowered or "completed" in lowered or "success" in lowered:
            tag = "success"
        else:
            tag = "info"
        self.log_text.insert("end", line + "\n", (tag,))

    def cancel_pipeline(self) -> None:
        if not self.pipeline_started or self.pipeline is None or self.pipeline.poll() is not None or self.state.cancel_requested:
            return
        if not messagebox.askyesno("Cancel Pipeline", "Stop the running pipeline?"):
            return
        self.state.cancel_requested = True
        self.cancel_button.configure(state="disabled")
        self.latest_var.set("Cancel requested. Stopping pipeline...")
        self.header_hint_var.set("Cancel requested. Waiting for subprocesses to stop.")
        self.root.after(10, self.stop_pipeline)

    def stop_pipeline(self) -> None:
        if self.pipeline is None or self.pipeline.poll() is not None:
            return

        try:
            self.pipeline.terminate()
            self.pipeline.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.latest_var.set("Pipeline did not stop gracefully. Forcing termination...")
            try:
                self.pipeline.kill()
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            pass

    def on_close(self) -> None:
        if not self.pipeline_started:
            self.root.destroy()
            return
        if self.state.process_done:
            self.root.destroy()
            return
        self.cancel_pipeline()

    def finish(self, return_code: int) -> None:
        self.state.process_done = True
        self.state.return_code = return_code
        if self.state.current_stage > 0 and self.state.current_stage not in self.state.stage_durations:
            self.state.stage_durations[self.state.current_stage] = time.time() - self.state.stage_started_at
        self.refresh_ui()
        if return_code == 0:
            self.latest_var.set("Pipeline completed successfully.")
            messagebox.showinfo("Pipeline Complete", "Pipeline completed successfully.")
        elif self.state.cancel_requested or return_code in (-15, 143, -9, 137):
            self.latest_var.set("Pipeline canceled by user.")
            messagebox.showinfo("Pipeline Canceled", "Pipeline canceled.")
        else:
            self.latest_var.set(f"Pipeline failed with exit code {return_code}.")
            messagebox.showerror("Pipeline Failed", f"Pipeline failed with exit code {return_code}.")


def main() -> int:
    args = parse_args()
    app = PipelineGui(args)

    def handle_signal(_signum, _frame) -> None:
        app.state.cancel_requested = True
        app.stop_pipeline()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    return app.start()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
