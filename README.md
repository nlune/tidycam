# TidyCam

TidyCam is a Raspberry Pi room monitor that uses a camera snapshot and an
Ollama vision model to decide whether a room is messy.

It captures a fresh image every 15 minutes, asks the model for a JSON response
with a room description and `is_messy` classification, writes the latest result
to disk, and plays audio prompts when tidying is needed.

## What It Does

- Captures still images from a Raspberry Pi camera module with `rpicam-still`.
- Sends snapshots to an Ollama vision model such as `gemma3:4b`.
- Produces structured JSON with `description`, `is_messy`, confidence, evidence,
  and cleanup suggestions.
- Runs with a GUI preview or in SSH/headless mode.
- Lets the GUI user trigger a fresh model check with `Analyze now`.
- Plays `Wakey Wakey v1.mp3` when the room is messy.
- Plays `Well Done!.mp3` 5 minutes after a messy-room alert.
- Saves the latest image and model output for later inspection.

## Quick Start

On the Raspberry Pi:

```sh
cd ~/Documents/tidycam
sudo apt update
sudo apt install -y python3-tk mpg123
uv sync
```

Create or update `.env`:

```sh
OLLAMA_URL="http://192.168.1.205:11434"
TIDYCAM_CAMERA="rpicam"
TIDYCAM_INTERVAL_SECONDS="900"
```

Run with the GUI:

```sh
uv run python tidycam_pi.py --mode gui
```

Run one headless check:

```sh
uv run python tidycam_pi.py --mode headless --once
cat tidycam_results/latest.md
```

## Ollama Setup

TidyCam needs a vision-capable model. Gemma 2 is text-only for this use case;
use `gemma3:4b` or larger.

On the machine running Ollama:

```sh
ollama pull gemma3:4b
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

If Ollama is running as the macOS app, quit it, set the host, then reopen it:

```sh
launchctl setenv OLLAMA_HOST "0.0.0.0:11434"
```

From the Pi, verify the laptop is reachable:

```sh
curl http://192.168.1.205:11434/api/tags
```

`OLLAMA_URL` must be only the base server URL. Do not include `/api/chat` or
`/api/generate`; the script adds `/api/generate` itself.

## Camera Setup

For the Raspberry Pi camera module, TidyCam defaults to the `rpicam` backend:

```sh
uv run python tidycam_pi.py --mode gui --camera rpicam
```

Test the camera stack directly if capture fails:

```sh
CAMERA_CMD="$(command -v rpicam-still || command -v libcamera-still)"
"$CAMERA_CMD" -n --timeout 1500 --width 1280 --height 720 --output test.jpg
ls -lh test.jpg
```

For a USB webcam, install OpenCV and use the OpenCV backend:

```sh
sudo apt install -y python3-opencv
uv run python tidycam_pi.py --mode gui --camera opencv --camera-index 0
```

## Running TidyCam

GUI mode:

```sh
uv run python tidycam_pi.py --mode gui
```

The preview refreshes from still captures. `Analyze now` captures a fresh image
and sends it to the model immediately.

Headless continuous mode:

```sh
uv run python tidycam_pi.py --mode headless
```

Headless one-shot mode:

```sh
uv run python tidycam_pi.py --mode headless --once
```

Faster test interval:

```sh
uv run python tidycam_pi.py --mode gui --interval 30
```

## Output Files

Each model check writes to `tidycam_results/`:

- `latest.md` - human-readable result.
- `latest.json` - structured model output and run metadata.
- `latest.jpg` - latest captured frame.
- `analyses.jsonl` - append-only history.
- Timestamped `.md`, `.json`, and `.jpg` files for each run.

The model JSON includes:

```json
{
  "description": "Detailed description of the visible scene",
  "is_messy": true,
  "confidence": 0.92,
  "messiness_evidence": ["Visible clutter on the floor"],
  "cleanup_suggestions": ["Pick up items from the floor"],
  "summary": "The room has visible clutter and should be tidied."
}
```

## Sound Alerts

When the model returns `"is_messy": true`, TidyCam plays:

1. `sound_files/Wakey Wakey v1.mp3` immediately.
2. `sound_files/Well Done!.mp3` after 5 minutes.

For a quick timing test:

```sh
uv run python tidycam_pi.py --mode gui --well-done-delay-seconds 10
```

Disable sound:

```sh
uv run python tidycam_pi.py --mode gui --no-sound
```

## Configuration

The script automatically loads `.env`. Useful settings:

```sh
OLLAMA_URL="http://192.168.1.205:11434"
OLLAMA_MODEL="gemma3:4b"
TIDYCAM_CAMERA="rpicam"
TIDYCAM_INTERVAL_SECONDS="900"
TIDYCAM_PREVIEW_REFRESH_SECONDS="5"
TIDYCAM_WELL_DONE_DELAY_SECONDS="300"
```

Common CLI options:

```sh
--mode gui|headless
--camera rpicam|opencv|picamera2
--interval 900
--preview-refresh-seconds 5
--well-done-delay-seconds 300
--no-sound
```

## Troubleshooting

Camera works in `rpicam-still` but not OpenCV:

Use `--camera rpicam`. Many Raspberry Pi camera module setups are exposed
through `rpicam`/`libcamera`, not as a normal USB webcam.

Ollama connection failed:

Check that the Pi can reach the Ollama server:

```sh
curl http://192.168.1.205:11434/api/tags
```

If that fails, make sure Ollama is listening on `0.0.0.0:11434`, the laptop and
Pi are on the same network, and the firewall allows port `11434`.

No MP3 sound:

```sh
sudo apt install -y mpg123
```

Then confirm the files exist:

```sh
ls -lh sound_files
```

GUI preview is slow:

Increase or decrease the still-capture refresh rate:

```sh
uv run python tidycam_pi.py --mode gui --preview-refresh-seconds 3
```
