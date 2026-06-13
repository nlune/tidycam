#!/usr/bin/env python3
"""Live Raspberry Pi room camera monitor with periodic Ollama vision checks."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

try:
    import requests
except ImportError:
    print("Missing dependency: requests. Install project dependencies with: uv sync")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Missing dependency: Pillow. Install project dependencies with: uv sync")
    sys.exit(1)


PROMPT = """
You are TidyCam, a practical room-tidying assistant.

Inspect this single camera snapshot. Describe the visible scene in clear detail:
the room type if apparent, major furniture, surfaces, floor areas, visible clutter,
storage, laundry, trash, dishes, cables, and anything that affects tidiness.

Then make a binary messy/not-messy classification. Messy means visible clutter,
items on the floor, crowded surfaces, unmade bedding, trash, dishes, laundry,
cables, or objects that appear out of place. Ignore normal furniture and permanent
decor. If the image is too dark, blocked, or does not show a room, set
is_messy to false, use low confidence, and explain the uncertainty.

Respond with JSON only:
{
  "description": "detailed description of the visible scene",
  "is_messy": true or false,
  "confidence": number from 0 to 1,
  "messiness_evidence": ["specific visible reason", "specific visible reason"],
  "cleanup_suggestions": ["specific action", "specific action"],
  "summary": "one short sentence about the room state"
}

If the room is not messy, keep cleanup_suggestions empty or include at most one
maintenance suggestion. Do not identify people or comment on personal traits.
""".strip()


OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "is_messy": {"type": "boolean"},
        "confidence": {"type": "number"},
        "messiness_evidence": {
            "type": "array",
            "items": {"type": "string"},
        },
        "cleanup_suggestions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "summary": {"type": "string"},
    },
    "required": [
        "description",
        "is_messy",
        "confidence",
        "messiness_evidence",
        "cleanup_suggestions",
        "summary",
    ],
}


def load_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def normalize_ollama_url(url: str) -> str:
    value = url.strip().rstrip("/")
    if "://" not in value:
        value = f"http://{value}"

    parts = urlsplit(value)
    path = parts.path.rstrip("/")
    for suffix in ("/api/generate", "/api/chat"):
        if path.endswith(suffix):
            path = path[: -len(suffix)].rstrip("/")
            break
    return urlunsplit((parts.scheme, parts.netloc, path, "", "")).rstrip("/")


@dataclass
class AnalysisResult:
    started_at: datetime
    ended_at: datetime
    payload: dict[str, Any] | None = None
    raw_response: str | None = None
    error: str | None = None
    paths: dict[str, Path] | None = None


class Picamera2Source:
    def __init__(self, width: int, height: int) -> None:
        from picamera2 import Picamera2

        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(1.0)

    def read_rgb(self) -> Any:
        return self.picam2.capture_array("main")

    def close(self) -> None:
        self.picam2.stop()


class OpenCVSource:
    def __init__(self, camera_index: int, width: int, height: int) -> None:
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV could not open camera index {camera_index}")

    def read_rgb(self) -> Any:
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("OpenCV camera read failed")
        return self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.cap.release()


class RpicamStillSource:
    def __init__(self, args: argparse.Namespace) -> None:
        command = args.rpicam_command or shutil.which("rpicam-still") or shutil.which("libcamera-still")
        if not command:
            raise RuntimeError("Neither rpicam-still nor libcamera-still was found on PATH")

        self.command = command
        self.width = args.width
        self.height = args.height
        self.timeout_ms = args.rpicam_timeout_ms

    def read_rgb(self) -> Image.Image:
        handle = tempfile.NamedTemporaryFile(prefix="tidycam-", suffix=".jpg", delete=False)
        image_path = Path(handle.name)
        handle.close()

        command = [
            self.command,
            "-n",
            "--output",
            str(image_path),
            "--timeout",
            str(self.timeout_ms),
            "--width",
            str(self.width),
            "--height",
            str(self.height),
        ]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                text=True,
                timeout=(self.timeout_ms / 1000) + 20,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()
                raise RuntimeError(f"{self.command} failed with exit code {completed.returncode}: {detail}")
            if not image_path.exists() or image_path.stat().st_size == 0:
                raise RuntimeError(f"{self.command} did not create a JPEG image")

            image = Image.open(image_path).convert("RGB")
            image.load()
            return image
        finally:
            image_path.unlink(missing_ok=True)

    def close(self) -> None:
        return None


CameraSource = Picamera2Source | RpicamStillSource | OpenCVSource


def open_camera(args: argparse.Namespace) -> CameraSource:
    errors: list[str] = []

    if args.camera in {"auto", "picamera2"}:
        try:
            return Picamera2Source(args.width, args.height)
        except Exception as exc:  # noqa: BLE001 - show concrete camera backend failures.
            errors.append(f"picamera2: {exc}")
            if args.camera == "picamera2":
                raise RuntimeError("\n".join(errors)) from exc

    if args.camera in {"auto", "rpicam"}:
        try:
            return RpicamStillSource(args)
        except Exception as exc:  # noqa: BLE001 - show concrete camera backend failures.
            errors.append(f"rpicam: {exc}")
            if args.camera == "rpicam":
                raise RuntimeError("\n".join(errors)) from exc

    if args.camera in {"auto", "opencv"}:
        try:
            return OpenCVSource(args.camera_index, args.width, args.height)
        except Exception as exc:  # noqa: BLE001 - show concrete camera backend failures.
            errors.append(f"opencv: {exc}")
            if args.camera == "opencv":
                raise RuntimeError("\n".join(errors)) from exc

    raise RuntimeError("Could not open a camera:\n" + "\n".join(errors))


def frame_to_image(frame_rgb: Any) -> Image.Image:
    if isinstance(frame_rgb, Image.Image):
        return frame_rgb.convert("RGB")
    return Image.fromarray(frame_rgb)


def frame_to_jpeg_bytes(frame_rgb: Any, quality: int) -> bytes:
    image = frame_to_image(frame_rgb)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def frame_to_jpeg_base64(frame_rgb: Any, quality: int) -> str:
    return base64.b64encode(frame_to_jpeg_bytes(frame_rgb, quality)).decode("ascii")


def parse_model_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def analyze_frame(frame_rgb: Any, args: argparse.Namespace) -> tuple[dict[str, Any] | None, str]:
    image_b64 = frame_to_jpeg_base64(frame_rgb, args.jpeg_quality)
    endpoint = f"{normalize_ollama_url(args.ollama_url)}/api/generate"
    response = requests.post(
        endpoint,
        json={
            "model": args.model,
            "prompt": PROMPT,
            "images": [image_b64],
            "stream": False,
            "format": OUTPUT_SCHEMA,
            "options": {
                "temperature": 0.1,
                "num_predict": args.num_predict,
            },
            "keep_alive": "10m",
        },
        timeout=args.timeout,
    )
    response.raise_for_status()

    data = response.json()
    if data.get("error"):
        raise RuntimeError(str(data["error"]))

    raw = str(data.get("response", "")).strip()
    try:
        payload = parse_model_json(raw)
    except json.JSONDecodeError:
        payload = None
    return payload, raw


def payload_is_messy(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    value = payload.get("is_messy", payload.get("messy"))
    return value is True


def format_seconds(seconds: float) -> str:
    if seconds % 60 == 0 and seconds >= 60:
        minutes = seconds / 60
        return f"{minutes:g} minute{'s' if minutes != 1 else ''}"
    return f"{seconds:g} second{'s' if seconds != 1 else ''}"


class SoundManager:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._player_command: list[str] | None | bool = False
        self._timers: list[threading.Timer] = []

    def trigger_for_result(self, result: AnalysisResult) -> threading.Timer | None:
        if self.args.no_sound or result.error or not payload_is_messy(result.payload):
            return None

        self.play(self.resolve_sound_path(self.args.wakey_sound), "messy alert")
        timer = threading.Timer(
            self.args.well_done_delay_seconds,
            self.play,
            args=(self.resolve_sound_path(self.args.well_done_sound), "well done"),
        )
        timer.daemon = True
        timer.start()
        self._timers.append(timer)
        print(f"Scheduled well done sound in {format_seconds(self.args.well_done_delay_seconds)}.")
        return timer

    def wait_for_delayed_sounds(self) -> None:
        for timer in list(self._timers):
            if timer.is_alive():
                print(f"Waiting for delayed sound: {format_seconds(self.args.well_done_delay_seconds)}.")
                timer.join()

    def resolve_sound_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        if path.exists():
            return path
        return Path(self.args.sounds_dir).expanduser() / path

    def audio_player_command(self) -> list[str] | None:
        if self._player_command is not False:
            return self._player_command

        if self.args.audio_player:
            command = shlex.split(self.args.audio_player)
            executable = shutil.which(command[0])
            self._player_command = command if executable else None
            return self._player_command

        candidates = [
            ["mpg123", "-q"],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
            ["mpv", "--no-video", "--really-quiet"],
            ["cvlc", "--play-and-exit", "--intf", "dummy"],
            ["afplay"],
        ]
        for command in candidates:
            if shutil.which(command[0]):
                self._player_command = command
                return command

        self._player_command = None
        return None

    def play(self, path: Path, label: str) -> None:
        if not path.exists():
            print(f"Sound file not found for {label}: {path}")
            return

        command = self.audio_player_command()
        if not command:
            print("No MP3 player found. Install mpg123 on the Pi: sudo apt install -y mpg123")
            return

        try:
            subprocess.Popen(
                [*command, str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"Playing {label} sound: {path}")
        except Exception as exc:  # noqa: BLE001 - audio failure should not stop monitoring.
            print(f"Could not play {label} sound {path}: {exc}")


def format_result(result: AnalysisResult) -> str:
    if result.error:
        return f"Analysis failed at {result.ended_at:%H:%M:%S}\n\n{result.error}"

    payload = result.payload or {}
    messy = payload_is_messy(payload)
    confidence = payload.get("confidence")
    summary = payload.get("summary") or "No summary returned."
    description = (
        payload.get("description")
        or payload.get("scene_description")
        or "No description returned."
    )
    evidence = payload.get("messiness_evidence") or []
    suggestions = payload.get("cleanup_suggestions") or []

    if isinstance(confidence, int | float):
        confidence_text = f"{confidence:.0%}"
    else:
        confidence_text = "unknown"

    messy_text = "Yes" if messy else "No"
    lines = [
        f"Last checked: {result.ended_at:%H:%M:%S}",
        f"Messy: {messy_text}",
        f"Confidence: {confidence_text}",
        "",
        "Summary:",
        str(summary),
        "",
        "Description:",
        str(description),
    ]

    if evidence:
        lines.append("")
        lines.append("Messiness evidence:")
        for item in evidence:
            lines.append(f"- {item}")

    if suggestions:
        lines.append("")
        lines.append("Cleanup suggestions:")
        for index, step in enumerate(suggestions, start=1):
            lines.append(f"{index}. {step}")

    if result.raw_response and not payload:
        lines.extend(["", "Raw response:", result.raw_response])

    return "\n".join(lines)


def result_record(result: AnalysisResult, image_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "started_at": result.started_at.isoformat(timespec="seconds"),
        "ended_at": result.ended_at.isoformat(timespec="seconds"),
        "model": args.model,
        "ollama_url": args.ollama_url,
        "image_path": str(image_path),
        "payload": result.payload,
        "raw_response": result.raw_response,
        "error": result.error,
    }


def write_result_files(frame_rgb: Any, result: AnalysisResult, args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = result.started_at.strftime("%Y%m%d-%H%M%S")
    image_path = output_dir / f"{timestamp}.jpg"
    latest_image_path = output_dir / "latest.jpg"
    json_path = output_dir / f"{timestamp}.json"
    latest_json_path = output_dir / "latest.json"
    markdown_path = output_dir / f"{timestamp}.md"
    latest_markdown_path = output_dir / "latest.md"
    log_path = output_dir / "analyses.jsonl"

    image_path.write_bytes(frame_to_jpeg_bytes(frame_rgb, args.jpeg_quality))
    shutil.copyfile(image_path, latest_image_path)

    record = result_record(result, image_path, args)
    json_text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    json_path.write_text(json_text, encoding="utf-8")
    latest_json_path.write_text(json_text, encoding="utf-8")

    markdown = (
        f"# TidyCam analysis\n\n"
        f"- Captured: {result.started_at.isoformat(timespec='seconds')}\n"
        f"- Completed: {result.ended_at.isoformat(timespec='seconds')}\n"
        f"- Model: `{args.model}`\n"
        f"- Image: `{image_path.name}`\n\n"
        f"{format_result(result)}\n"
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    latest_markdown_path.write_text(markdown, encoding="utf-8")

    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")

    return {
        "image": image_path,
        "latest_image": latest_image_path,
        "json": json_path,
        "latest_json": latest_json_path,
        "markdown": markdown_path,
        "latest_markdown": latest_markdown_path,
        "log": log_path,
    }


class TidyCamApp:
    def __init__(self, root: Any, camera: CameraSource, args: argparse.Namespace) -> None:
        self.root = root
        self.camera = camera
        self.args = args
        self.result_queue: queue.Queue[AnalysisResult] = queue.Queue()
        self.preview_queue: queue.Queue[tuple[Any | None, Exception | None]] = queue.Queue()
        self.latest_frame: Any | None = None
        self.photo: Any | None = None
        self.analysis_running = False
        self.next_analysis_at = 0.0 if args.initial_check else time.monotonic() + args.interval
        self.next_preview_at = 0.0
        self.preview_running = False
        self.closed = False
        self.sound_manager = SoundManager(args)

        self.root.title("TidyCam")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self._build_ui()

    def _build_ui(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(0, weight=1)

        self.preview = ttk.Label(self.root)
        self.preview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        panel = ttk.Frame(self.root, padding=10)
        panel.grid(row=0, column=1, sticky="ns")
        panel.columnconfigure(0, weight=1)

        ttk.Label(panel, text="Room evaluation", font=("TkDefaultFont", 14, "bold")).grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(panel, text=f"Model: {self.args.model}").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(panel, text=f"Interval: {self.args.interval:g}s").grid(row=2, column=0, sticky="w")

        self.status_var = tk.StringVar(value="Starting camera...")
        ttk.Label(panel, textvariable=self.status_var).grid(row=3, column=0, sticky="w", pady=(8, 8))

        self.response_text = tk.Text(panel, width=42, height=22, wrap="word")
        self.response_text.grid(row=4, column=0, sticky="nsew")
        self.response_text.insert(
            "1.0",
            "Waiting for the first frame. The first analysis runs automatically.",
        )
        self.response_text.configure(state="disabled")

        controls = ttk.Frame(panel)
        controls.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(controls, text="Analyze now", command=self.analyze_now).grid(row=0, column=0, sticky="ew")
        ttk.Button(controls, text="Quit", command=self.close).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

    def run(self) -> None:
        self._tick()
        self.root.mainloop()

    def _tick(self) -> None:
        if self.closed:
            return

        now = time.monotonic()
        if not self.preview_running and now >= self.next_preview_at:
            self._start_preview_capture()

        self._drain_preview_results()
        self._drain_results()

        if self.latest_frame is not None and not self.analysis_running and now >= self.next_analysis_at:
            self._start_analysis(self.latest_frame)

        self.root.after(66, self._tick)

    def _show_frame(self, frame_rgb: Any) -> None:
        import tkinter as tk

        image = frame_to_image(frame_rgb)
        image.thumbnail((self.args.preview_width, self.args.preview_height), Image.Resampling.LANCZOS)
        try:
            self.photo = self._photo_image_from_temp_file(tk, image, "PPM", ".ppm")
        except tk.TclError:
            self.photo = self._photo_image_from_temp_file(tk, image, "GIF", ".gif")
        self.preview.configure(image=self.photo)

    def _photo_image_from_temp_file(self, tk_module: Any, image: Image.Image, image_format: str, suffix: str) -> Any:
        handle = tempfile.NamedTemporaryFile(prefix="tidycam-preview-", suffix=suffix, delete=False)
        preview_path = Path(handle.name)
        handle.close()
        try:
            image.save(preview_path, format=image_format)
            return tk_module.PhotoImage(file=str(preview_path))
        finally:
            preview_path.unlink(missing_ok=True)

    def _start_preview_capture(self) -> None:
        self.preview_running = True
        thread = threading.Thread(target=self._preview_worker, daemon=True)
        thread.start()

    def _preview_worker(self) -> None:
        try:
            frame = self.camera.read_rgb()
            self.preview_queue.put((frame, None))
        except Exception as exc:  # noqa: BLE001 - show camera backend failures in the UI.
            self.preview_queue.put((None, exc))

    def _drain_preview_results(self) -> None:
        while True:
            try:
                frame, error = self.preview_queue.get_nowait()
            except queue.Empty:
                return
            self._finish_preview_capture(frame, error)

    def _finish_preview_capture(self, frame: Any | None, error: Exception | None) -> None:
        self.preview_running = False
        self.next_preview_at = time.monotonic() + self.args.preview_refresh_seconds
        if self.closed:
            return

        if error:
            self.status_var.set(f"Camera error: {error}")
            return

        self.latest_frame = frame
        self._show_frame(frame)

    def analyze_now(self) -> None:
        if self.analysis_running:
            self.status_var.set("Analysis is already running.")
            return
        if self.preview_running:
            self.status_var.set("Camera capture is already running.")
            return
        self._start_manual_analysis()

    def _start_manual_analysis(self) -> None:
        self.analysis_running = True
        self.preview_running = True
        self.status_var.set("Capturing fresh image for analysis...")

        thread = threading.Thread(target=self._manual_analysis_worker, daemon=True)
        thread.start()

    def _manual_analysis_worker(self) -> None:
        started_at = datetime.now()
        try:
            frame = self.camera.read_rgb()
            self.preview_queue.put((frame, None))
            self._analysis_worker(frame)
        except Exception as exc:  # noqa: BLE001 - surface manual capture failures in the UI.
            result = AnalysisResult(
                started_at=started_at,
                ended_at=datetime.now(),
                error=f"Manual camera capture failed: {exc}",
            )
            self.result_queue.put(result)
            self.preview_queue.put((None, exc))

    def _start_analysis(self, frame_rgb: Any) -> None:
        frame_copy = frame_rgb.copy()
        self.analysis_running = True
        self.status_var.set("Sending snapshot to Ollama...")

        thread = threading.Thread(target=self._analysis_worker, args=(frame_copy,), daemon=True)
        thread.start()

    def _analysis_worker(self, frame_rgb: Any) -> None:
        started_at = datetime.now()
        try:
            payload, raw = analyze_frame(frame_rgb, self.args)
            result = AnalysisResult(
                started_at=started_at,
                ended_at=datetime.now(),
                payload=payload,
                raw_response=raw,
            )
        except Exception as exc:  # noqa: BLE001 - surface HTTP/model/camera errors in the UI.
            result = AnalysisResult(
                started_at=started_at,
                ended_at=datetime.now(),
                error=str(exc),
            )
        try:
            result.paths = write_result_files(frame_rgb, result, self.args)
        except Exception as exc:  # noqa: BLE001 - show filesystem errors alongside model errors.
            write_error = f"Failed to write output files: {exc}"
            result.error = f"{result.error}\n\n{write_error}" if result.error else write_error
        self.result_queue.put(result)

    def _drain_results(self) -> None:
        while True:
            try:
                result = self.result_queue.get_nowait()
            except queue.Empty:
                return

            self.analysis_running = False
            self.next_analysis_at = time.monotonic() + self.args.interval
            if result.paths:
                self.status_var.set(f"Saved {result.paths['latest_markdown']}. Next check in {self.args.interval:g}s.")
            else:
                self.status_var.set(f"Next check in {self.args.interval:g}s.")
            self.response_text.configure(state="normal")
            self.response_text.delete("1.0", "end")
            self.response_text.insert("1.0", format_result(result))
            self.response_text.configure(state="disabled")
            self.sound_manager.trigger_for_result(result)

    def close(self) -> None:
        self.closed = True
        try:
            self.camera.close()
        finally:
            self.root.destroy()


def has_display() -> bool:
    if sys.platform == "darwin":
        return True
    return bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))


def use_headless_mode(args: argparse.Namespace) -> bool:
    if args.mode == "headless":
        return True
    if args.mode == "gui":
        return False
    return not has_display()


def run_gui(camera: CameraSource, args: argparse.Namespace) -> int:
    try:
        import tkinter as tk
    except ImportError:
        print("Missing dependency: tkinter. Install it with: sudo apt install python3-tk")
        camera.close()
        return 1

    root = tk.Tk()
    app = TidyCamApp(root, camera, args)
    app.run()
    return 0


def run_headless(camera: CameraSource, args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    sound_manager = SoundManager(args)
    print(f"Running headless. Writing results to: {output_dir}")
    print("Press Ctrl+C to stop." if not args.once else "Running one capture.")

    if not args.initial_check:
        time.sleep(args.interval)

    try:
        while True:
            frame = camera.read_rgb()
            started_at = datetime.now()
            print(f"[{started_at:%Y-%m-%d %H:%M:%S}] Sending snapshot to Ollama...")

            try:
                payload, raw = analyze_frame(frame, args)
                result = AnalysisResult(
                    started_at=started_at,
                    ended_at=datetime.now(),
                    payload=payload,
                    raw_response=raw,
                )
            except Exception as exc:  # noqa: BLE001 - keep headless loop useful after transient errors.
                result = AnalysisResult(
                    started_at=started_at,
                    ended_at=datetime.now(),
                    error=str(exc),
                )

            result.paths = write_result_files(frame, result, args)
            print(format_result(result))
            print(f"\nSaved latest report: {result.paths['latest_markdown']}")
            print(f"Saved latest image: {result.paths['latest_image']}")
            sound_manager.trigger_for_result(result)

            if args.once:
                sound_manager.wait_for_delayed_sounds()
                return 0

            print(f"Next check in {args.interval:g}s.\n")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    finally:
        camera.close()


def parse_args() -> argparse.Namespace:
    load_env_file()

    parser = argparse.ArgumentParser(
        description="Capture Raspberry Pi camera snapshots and ask Ollama to describe messiness and cleanup steps.",
    )
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "gemma3:4b"))
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--interval", type=float, default=float(os.getenv("TIDYCAM_INTERVAL_SECONDS", "900")))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("TIDYCAM_OLLAMA_TIMEOUT", "180")))
    parser.add_argument("--num-predict", type=int, default=int(os.getenv("TIDYCAM_NUM_PREDICT", "600")))
    parser.add_argument("--output-dir", default=os.getenv("TIDYCAM_OUTPUT_DIR", "tidycam_results"))
    parser.add_argument("--no-sound", action="store_true", help="Disable messy-room audio alerts.")
    parser.add_argument("--sounds-dir", default=os.getenv("TIDYCAM_SOUNDS_DIR", "sound_files"))
    parser.add_argument(
        "--wakey-sound",
        default=os.getenv("TIDYCAM_WAKEY_SOUND", "Wakey Wakey v1.mp3"),
        help="MP3 played immediately when the model classifies the room as messy.",
    )
    parser.add_argument(
        "--well-done-sound",
        default=os.getenv("TIDYCAM_WELL_DONE_SOUND", "Well Done!.mp3"),
        help="MP3 played after the delayed cleanup window when the room was classified as messy.",
    )
    parser.add_argument(
        "--well-done-delay-seconds",
        type=float,
        default=float(os.getenv("TIDYCAM_WELL_DONE_DELAY_SECONDS", "300")),
        help="Seconds to wait after a messy classification before playing the well done sound.",
    )
    parser.add_argument(
        "--audio-player",
        default=os.getenv("TIDYCAM_AUDIO_PLAYER"),
        help='Optional audio player command, for example "mpg123 -q". Auto-detected by default.',
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "gui", "headless"],
        default=os.getenv("TIDYCAM_MODE", "auto"),
        help="Use GUI preview, headless file output, or auto-detect based on display availability.",
    )
    parser.add_argument("--once", action="store_true", help="Capture, evaluate, write output files, and exit.")
    parser.add_argument(
        "--camera",
        choices=["auto", "picamera2", "rpicam", "opencv"],
        default=os.getenv("TIDYCAM_CAMERA", "auto"),
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--rpicam-command",
        default=os.getenv("TIDYCAM_RPICAM_COMMAND"),
        help="Path to rpicam-still or libcamera-still. Auto-detected by default.",
    )
    parser.add_argument(
        "--rpicam-timeout-ms",
        type=int,
        default=int(os.getenv("TIDYCAM_RPICAM_TIMEOUT_MS", "1500")),
        help="Warm-up/capture timeout passed to rpicam-still in milliseconds.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--preview-width", type=int, default=960)
    parser.add_argument("--preview-height", type=int, default=540)
    parser.add_argument(
        "--preview-refresh-seconds",
        type=float,
        default=float(os.getenv("TIDYCAM_PREVIEW_REFRESH_SECONDS", "5")),
        help="Seconds between GUI preview captures. Use a higher value with rpicam-still.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument(
        "--no-initial-check",
        dest="initial_check",
        action="store_false",
        help="Wait one interval before the first Ollama analysis.",
    )
    parser.set_defaults(initial_check=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        camera = open_camera(args)
    except Exception as exc:  # noqa: BLE001 - command-line startup should print concrete backend failures.
        print(exc)
        return 1

    if use_headless_mode(args):
        return run_headless(camera, args)
    return run_gui(camera, args)


if __name__ == "__main__":
    raise SystemExit(main())
