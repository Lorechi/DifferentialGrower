# Differential Growth Camera

Static webcam-based differential growth outline experiment. The GitHub Pages
website lives at the repository root:

- `index.html`
- `styles.css`
- `app.js`

The older Python/OpenCV prototype is archived in `archive/`.

## GitHub Pages

Try the browser app at [lorechi.github.io/DifferentialGrower](https://lorechi.github.io/DifferentialGrower/).

The browser app uses the webcam, so it must be served from `https://` or
`localhost`. GitHub Pages works because it serves over HTTPS.

## Local Run

```bash
python -m http.server 8000
```

Then open `http://127.0.0.1:8000`.

## Controls

- Q / Esc: stop camera
- R: reseed from current outline
- D: toggle camera background
- F: toggle fullscreen

Use the side controls menu for live growth, smoothing, speed, and line-width
sliders.

## Python App

The old Python/OpenCV version is archived in `archive/`. Recreate a virtual
environment from its `requirements.txt` if you want to run it locally.
