#!/usr/bin/env python3
"""Terminal progress monitor for run_gs_pipeline_core.sh.

This wrapper keeps the existing shell pipeline intact and turns its stage logs
into a cleaner terminal dashboard using only standard Python libraries.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional


STAGE_RE = re.compile(r"\[(\d+)/(\d+)\]\s+(.*)")
VRAM_RE = re.compile(r"(\d+)\s*,\s*(\d+)")
DEFAULT_LOG_LINES = 8
REFRESH_SECONDS = 0.25
SPINNER_FRAMES = ["|", "/", "-", "\\"]

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

FG_WHITE = "\033[97m"
FG_CYAN = "\033[96m"
FG_GREEN = "\033[92m"
FG_YELLOW = "\033[93m"
FG_RED = "\033[91m"
FG_BLUE = "\033[94m"

BG_BLUE = "\033[44m"
BG_GREEN = "\033[42m"
BG_RED = "\033[41m"
BG_YELLOW = "\033[43m"


@dataclass
class MonitorState:
    current_stage: int = 0
    total_stages: int = 9
    stage_label: str = "Starting"
    started_at: float = 0.0
    stage_started_at: float = 0.0
    vram_used_mb: Optional[int] = None
    vram_total_mb: Optional[int] = None
    process_done: bool = False
    return_code: Optional[int] = None
    last_line: str = ""
    stage_names: dict[int, str] = field(default_factory=dict)
    stage_durations: dict[int, float] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run run_gs_pipeline_core.sh with a cleaner terminal progress view."
    )
    parser.add_argument(
        "--workspace",
        help="Workspace passed to the pipeline. Optional here if you pass it after --.",
    )
    parser.add_argument(
        "--log-lines",
        type=int,
        default=DEFAULT_LOG_LINES,
        help=f"Number of recent log lines to display (default: {DEFAULT_LOG_LINES}).",
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


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def colorize(text: str, *styles: str) -> str:
    return "".join(styles) + text + RESET


def status_badge(state: MonitorState) -> str:
    if state.process_done and state.return_code == 0:
        return colorize(" DONE ", BG_GREEN, FG_WHITE, BOLD)
    if state.process_done and state.return_code not in (0, None):
        return colorize(f" FAILED {state.return_code} ", BG_RED, FG_WHITE, BOLD)
    return colorize(" RUNNING ", BG_BLUE, FG_WHITE, BOLD)


def section_title(title: str, width: int) -> str:
    rule = "-" * max(8, width - len(title) - 1)
    return f"{colorize(title, BOLD, FG_CYAN)} {colorize(rule, DIM)}"


def pretty_label(text: str) -> str:
    return colorize(f"{text:<10}", FG_CYAN, BOLD)


def color_bar(done: int, total: int, width: int = 32) -> str:
    if total <= 0:
        total = 1
    ratio = max(0.0, min(1.0, done / total))
    filled = int(width * ratio)
    filled_part = colorize("#" * filled, FG_GREEN, BOLD) if filled else ""
    empty_part = colorize("-" * (width - filled), DIM) if width - filled else ""
    return "[" + filled_part + empty_part + RESET + "]"


def stage_marker(state: MonitorState, stage_num: int) -> str:
    if state.process_done and state.return_code == 0 and stage_num <= state.total_stages:
        if stage_num <= state.current_stage:
            return colorize("OK", FG_GREEN, BOLD)
    if stage_num < state.current_stage:
        return colorize("OK", FG_GREEN, BOLD)
    if stage_num == state.current_stage and not state.process_done:
        return colorize(">>", FG_YELLOW, BOLD)
    if state.process_done and state.return_code not in (0, None) and stage_num == state.current_stage:
        return colorize("!!", FG_RED, BOLD)
    return colorize("..", DIM)


def render_timeline(state: MonitorState, width: int) -> list[str]:
    lines = []
    for stage_num in range(1, state.total_stages + 1):
        label = state.stage_names.get(stage_num, f"Stage {stage_num}")
        marker = stage_marker(state, stage_num)
        duration_text = ""
        if stage_num in state.stage_durations:
            duration_text = format_duration(state.stage_durations[stage_num])
        elif stage_num == state.current_stage and not state.process_done and state.current_stage > 0:
            duration_text = format_duration(time.time() - state.stage_started_at)

        label_width = max(12, width - 18)
        trimmed_label = trim_line(label, label_width)
        line = f"{marker} {stage_num:>2}. {trimmed_label:<{label_width}}"
        if duration_text:
            line += " " + colorize(duration_text.rjust(8), FG_BLUE, BOLD)

        if stage_num == state.current_stage and not state.process_done:
            lines.append(colorize(line, FG_WHITE, BOLD))
        elif stage_num < state.current_stage:
            lines.append(colorize(line, FG_GREEN))
        else:
            lines.append(colorize(line, DIM))
    return lines


def trim_line(text: str, width: int) -> str:
    clean = text.replace("\t", "    ").strip()
    if len(clean) <= width:
        return clean
    return clean[: max(0, width - 3)] + "..."


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def visual_len(text: str) -> int:
    return len(strip_ansi(text))


def pad_visual(text: str, width: int) -> str:
    return text + " " * max(0, width - visual_len(text))


def box(title: str, lines: list[str], width: int) -> list[str]:
    inner = max(10, width - 4)
    header = f"+- {title} " + "-" * max(0, width - len(title) - 6) + "+"
    body = [f"| {pad_visual(line, inner)} |" for line in lines]
    footer = "+" + "-" * (width - 2) + "+"
    return [colorize(header, FG_CYAN, BOLD), *body, colorize(footer, DIM)]


def merge_columns(left: list[str], right: list[str], left_width: int, gap: int = 2) -> list[str]:
    rows = max(len(left), len(right))
    output = []
    for i in range(rows):
        left_line = left[i] if i < len(left) else ""
        right_line = right[i] if i < len(right) else ""
        output.append(pad_visual(left_line, left_width) + (" " * gap) + right_line)
    return output


def spinner_frame(state: MonitorState) -> str:
    if state.process_done:
        return colorize("*", FG_GREEN, BOLD) if state.return_code == 0 else colorize("x", FG_RED, BOLD)
    idx = int(time.time() / REFRESH_SECONDS) % len(SPINNER_FRAMES)
    return colorize(SPINNER_FRAMES[idx], FG_YELLOW, BOLD)


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

    first_line = result.stdout.strip().splitlines()
    if not first_line:
        return None, None
    match = VRAM_RE.search(first_line[0])
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def render_screen(state: MonitorState, logs: Deque[str], workspace: Optional[str]) -> str:
    term_width = shutil.get_terminal_size((100, 30)).columns
    body_width = max(60, term_width - 6)
    elapsed = format_duration(time.time() - state.started_at)
    stage_elapsed = format_duration(time.time() - state.stage_started_at)
    progress_bar = color_bar(state.current_stage, state.total_stages)
    percent = int((state.current_stage / max(1, state.total_stages)) * 100)

    if state.vram_used_mb is None or state.vram_total_mb is None:
        vram_text = colorize("n/a", DIM)
    else:
        vram_ratio = state.vram_used_mb / max(1, state.vram_total_mb)
        vram_color = FG_GREEN if vram_ratio < 0.6 else FG_YELLOW if vram_ratio < 0.85 else FG_RED
        vram_text = colorize(f"{state.vram_used_mb}/{state.vram_total_mb} MB", vram_color, BOLD)

    title = f"{spinner_frame(state)} {colorize('3DGS PIPELINE MONITOR', BOLD, FG_WHITE)}"
    lines = [title, colorize("=" * min(term_width, 96), DIM)]

    summary_lines = [
        f"{pretty_label('Status')} {status_badge(state)}",
        f"{pretty_label('Progress')} {colorize(f'{state.current_stage}/{state.total_stages}', FG_WHITE, BOLD)}  {progress_bar}  {colorize(f'{percent:>3}%', FG_GREEN, BOLD)}",
        f"{pretty_label('Stage')} {colorize(trim_line(state.stage_label, 44), FG_WHITE, BOLD)}",
        f"{pretty_label('Elapsed')} {colorize(elapsed, FG_WHITE, BOLD)} total",
        f"{pretty_label('Current')} {colorize(stage_elapsed, FG_BLUE, BOLD)} stage",
        f"{pretty_label('VRAM')} {vram_text}",
    ]
    if workspace:
        summary_lines.append(f"{pretty_label('Workspace')} {colorize(trim_line(workspace, 44), FG_WHITE)}")
    if state.last_line:
        summary_lines.append(f"{pretty_label('Latest')} {colorize(trim_line(state.last_line, 44), FG_YELLOW)}")

    use_columns = term_width >= 110
    if use_columns:
        left_width = min(54, max(42, term_width // 2 - 2))
        right_width = term_width - left_width - 2
        left_box = box("Summary", summary_lines, left_width)
        right_box = box("Stage Timeline", render_timeline(state, right_width - 4), right_width)
        lines.append("")
        lines.extend(merge_columns(left_box, right_box, left_width))
    else:
        lines.append("")
        lines.extend(box("Summary", summary_lines, min(term_width, 96)))
        lines.append("")
        lines.extend(box("Stage Timeline", render_timeline(state, min(term_width, 96) - 4), min(term_width, 96)))

    lines.append("")

    log_lines = []
    if logs:
        for line in logs:
            display = trim_line(line, body_width)
            if STAGE_RE.search(line):
                display = colorize(display, FG_CYAN, BOLD)
            elif "error" in line.lower() or "failed" in line.lower():
                display = colorize(display, FG_RED, BOLD)
            elif "warning" in line.lower() or "skipped" in line.lower():
                display = colorize(display, FG_YELLOW)
            wrapped_lines = textwrap.wrap(trim_line(line, body_width), width=body_width) or [""]
            for idx, wrapped in enumerate(wrapped_lines):
                if idx == 0:
                    log_lines.append("  " + display)
                else:
                    log_lines.append("  " + wrapped)
    else:
        log_lines.append(colorize("  (waiting for pipeline output)", DIM))

    lines.append("")
    lines.extend(box("Recent Logs", log_lines, min(term_width, 96)))
    lines.append("")
    lines.extend(
        box(
            "Controls",
            [colorize("Ctrl+C", FG_WHITE, BOLD) + " stops both the monitor and the pipeline."],
            min(term_width, 96),
        )
    )
    return "\033[2J\033[H" + "\n".join(lines)


def normalize_pipeline_args(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def main() -> int:
    args = parse_args()
    pipeline_args = normalize_pipeline_args(args.pipeline_args)
    cmd = ["./run_gs_pipeline_core.sh", *pipeline_args]

    state = MonitorState(started_at=time.time(), stage_started_at=time.time())
    logs: Deque[str] = deque(maxlen=max(1, args.log_lines))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def handle_signal(signum, _frame):
        if proc.poll() is None:
            proc.send_signal(signum)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    def read_output() -> None:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            state.last_line = line
            logs.append(line)
            match = STAGE_RE.search(line)
            if match:
                new_stage = int(match.group(1))
                total = int(match.group(2))
                label = match.group(3).strip()
                if new_stage != state.current_stage:
                    if state.current_stage > 0:
                        state.stage_durations[state.current_stage] = time.time() - state.stage_started_at
                    state.stage_started_at = time.time()
                state.current_stage = new_stage
                state.total_stages = total
                state.stage_label = label
                state.stage_names[new_stage] = label

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    try:
        while proc.poll() is None:
            if not args.no_vram:
                state.vram_used_mb, state.vram_total_mb = sample_vram()
            sys.stdout.write(render_screen(state, logs, args.workspace))
            sys.stdout.flush()
            time.sleep(REFRESH_SECONDS)
    except KeyboardInterrupt:
        sys.stdout.write("\nStopping pipeline...\n")
        sys.stdout.flush()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        state.process_done = True
        state.return_code = proc.returncode
        sys.stdout.write(render_screen(state, logs, args.workspace))
        sys.stdout.write("\n")
        sys.stdout.flush()
        return 130

    reader.join(timeout=1)
    state.process_done = True
    state.return_code = proc.returncode
    if not args.no_vram:
        state.vram_used_mb, state.vram_total_mb = sample_vram()
    if state.current_stage > 0 and state.current_stage not in state.stage_durations:
        state.stage_durations[state.current_stage] = time.time() - state.stage_started_at
    if proc.returncode == 0 and state.current_stage < state.total_stages:
        state.current_stage = state.total_stages
        if state.stage_label == "Starting":
            state.stage_label = "Done"

    sys.stdout.write(render_screen(state, logs, args.workspace))
    sys.stdout.write("\n")
    sys.stdout.flush()
    return proc.returncode


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
