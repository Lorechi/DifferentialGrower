# Differential Growth Camera

This project uses a camera + MediaPipe segmentation and applies differential
growth to the detected human outline. It includes the original Python/OpenCV
app and a static browser version in `web/`.

For GitHub Pages, serve the repository from `/ (root)`. The root `index.html`
redirects to this static browser app.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Python app:

```bash
python main_camera.py
```

Static browser app:

```bash
cd web
python -m http.server 8000
```

Then open `http://localhost:8000`.

The browser version loads MediaPipe from a CDN, so it needs internet access the
first time it starts.

## Camera Controls

- Q / Esc: quit
- R: reseed from current outline
- D: toggle camera background
- F: toggle fullscreen

The browser version also has a side controls menu with live sliders for growth
strength, spread, speed, smoothing, and line width.
