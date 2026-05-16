# Archived Python/OpenCV Version

This is the older local Python/OpenCV version of the Differential Growth Camera
experiment. The current GitHub Pages website lives at the repository root.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main_camera.py
```

## Camera Controls

- Q / Esc: quit
- R: reseed from current outline
- D: toggle camera background
- F: toggle fullscreen

The browser version also has a side controls menu with live sliders for growth
strength, spread, speed, smoothing, and line width.
