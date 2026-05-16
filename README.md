# Differential Growth Camera

Webcam-based differential growth outline experiment.

## GitHub Pages

This repository can be served directly with GitHub Pages:

1. Open the repository settings on GitHub.
2. Go to **Pages**.
3. Set **Source** to `Deploy from a branch`.
4. Set the branch to `main` and the folder to `/ (root)`.
5. Save.

The root `index.html` redirects to the static browser app in
`pygame_app/web/`.

The browser app uses the webcam, so it must be served from `https://` or
`localhost`. GitHub Pages works because it serves over HTTPS.

## Local Run

```bash
cd pygame_app/web
python -m http.server 8000
```

Then open `http://127.0.0.1:8000`.

## Python App

The older Python/OpenCV version is still in `pygame_app/main_camera.py`.
Recreate a virtual environment from `pygame_app/requirements.txt` if you want
to run it locally.
