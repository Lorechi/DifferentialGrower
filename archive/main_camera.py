import math
import tkinter as tk
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

@dataclass
class Params:
    min_edge: float = 10.0  # Target spacing when resampling contours.
    min_contour_area: float = 2500.0  # Ignore tiny/noisy segmentation blobs.
    max_edge: float = 24.0  # Insert a point if a segment exceeds this length.
    curvature_threshold: float = 0.55  # Lower values insert more at sharp bends.
    max_new_points: int = 20  # Max points inserted per step.
    repel_radius: float = 26.0  # Neighbor repulsion range.
    repel_strength: float = 0.25  # Strength of repulsion force.
    relax_strength: float = 0.35  # Strength of neighbor attraction (tension).
    iterations: int = 1  # Growth iterations per frame.
    time_step: float = 0.5  # Integration step size (overall speed).
    max_points: int = 1000  # Hard cap on points.
    smooth_strength: float = 0.3  # Laplacian smoothing strength.
    smooth_every: int = 3  # Apply smoothing every N steps.
    seed_interval: float = 3.0  # Seconds between outline reseeds.
    growth_delay: float = 0.8  # Seconds to show a fresh outline before growth starts.
    max_contours: int = 3  # Max number of human outlines to grow.
    outline_color: tuple = (255, 255, 255)  # BGR color for outline.
    
    dissolve_time: float = 1.8  # Seconds before burst fragments fully disappear.
    fragment_points: int = 12  # Approximate points per dissolving segment.
    burst_speed: float = 95.0  # Base outward speed of dissolving fragments.
    burst_jitter: float = 45.0  # Random velocity added to each fragment.
    burst_drag: float = 0.92  # Per-frame velocity damping for fragments.
    fragment_thickness: int = 2  # Stroke width of dissolving fragments.
    debug_camera_overlay: bool = True  # Show a faint grayscale camera image on the canvas.
    debug_overlay_opacity: float = 0.5  # Blend amount for the grayscale debug overlay.


@dataclass
class BurstFragment:
    points: np.ndarray
    velocity: np.ndarray
    ttl: float
    age: float = 0.0


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def resample_contour(points, spacing):
    if len(points) < 2:
        return points
    resampled = [points[0][:]]
    carry = 0.0
    for i in range(1, len(points)):
        a = points[i - 1]
        b = points[i]
        seg_len = distance(a, b)
        if seg_len == 0:
            continue
        dir_x = (b[0] - a[0]) / seg_len
        dir_y = (b[1] - a[1]) / seg_len
        d = spacing - carry
        while d <= seg_len:
            resampled.append([a[0] + dir_x * d, a[1] + dir_y * d])
            d += spacing
        carry = d - seg_len
    return resampled


def apply_differential_growth(points, params):
    if len(points) < 2:
        return points

    count = len(points)
    forces = [[0.0, 0.0] for _ in range(count)]

    # Neighbor attraction
    for i in range(count):
        curr = points[i]
        prev = points[i - 1] if i > 0 else points[-1]
        nxt = points[i + 1] if i < count - 1 else points[0]
        forces[i][0] += (prev[0] - curr[0]) * params.relax_strength
        forces[i][1] += (prev[1] - curr[1]) * params.relax_strength
        forces[i][0] += (nxt[0] - curr[0]) * params.relax_strength
        forces[i][1] += (nxt[1] - curr[1]) * params.relax_strength

    # Repulsion (spatial grid)
    r = params.repel_radius
    r2 = r * r
    cell_size = max(4.0, r)
    grid = {}
    for idx, p in enumerate(points):
        cx = int(p[0] // cell_size)
        cy = int(p[1] // cell_size)
        grid.setdefault((cx, cy), []).append(idx)

    for i, a in enumerate(points):
        cx = int(a[0] // cell_size)
        cy = int(a[1] // cell_size)
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for j in grid.get((gx, gy), []):
                    if j <= i:
                        continue
                    b = points[j]
                    dx = b[0] - a[0]
                    dy = b[1] - a[1]
                    dist2 = dx * dx + dy * dy
                    if dist2 == 0 or dist2 > r2:
                        continue
                    dist = math.sqrt(dist2)
                    force = (1 - dist / r) * params.repel_strength
                    fx = (dx / dist) * force
                    fy = (dy / dist) * force
                    forces[i][0] -= fx
                    forces[i][1] -= fy
                    forces[j][0] += fx
                    forces[j][1] += fy

    # Apply forces
    step = params.time_step
    for i in range(count):
        points[i][0] += forces[i][0] * step
        points[i][1] += forces[i][1] * step

    # Insert points (random candidates)
    candidates = []
    for i in range(count):
        a = points[i]
        b = points[(i + 1) % count]
        insert = distance(a, b) > params.max_edge
        if count > 2:
            prev = points[i - 1] if i > 0 else points[-1]
            next_pt = points[(i + 1) % count]
            v1x = prev[0] - a[0]
            v1y = prev[1] - a[1]
            v2x = next_pt[0] - a[0]
            v2y = next_pt[1] - a[1]
            len1 = math.hypot(v1x, v1y)
            len2 = math.hypot(v2x, v2y)
            if len1 > 0 and len2 > 0:
                cos_theta = (v1x * v2x + v1y * v2y) / (len1 * len2)
                if cos_theta < params.curvature_threshold:
                    insert = True
        if insert:
            candidates.append(i)

    max_inserts = min(
        params.max_new_points,
        params.max_points - len(points),
        len(candidates),
    )
    selected = set()
    if max_inserts > 0:
        selected = set(np.random.choice(candidates, size=max_inserts, replace=False))

    new_points = []
    for i in range(count):
        a = points[i]
        b = points[(i + 1) % count]
        new_points.append(a)
        if i in selected:
            new_points.append([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2])

    return new_points


def smooth(points, strength):
    if len(points) < 3:
        return points
    smoothed = []
    count = len(points)
    for i in range(count):
        prev = points[i - 1]
        curr = points[i]
        nxt = points[(i + 1) % count]
        target_x = (prev[0] + curr[0] + nxt[0]) / 3
        target_y = (prev[1] + curr[1] + nxt[1]) / 3
        smoothed.append([
            curr[0] + (target_x - curr[0]) * strength,
            curr[1] + (target_y - curr[1]) * strength,
        ])
    return smoothed


def loop_centroid(points):
    if not points:
        return np.array([0.0, 0.0], dtype=np.float32)
    pts = np.asarray(points, dtype=np.float32)
    return pts.mean(axis=0)


def create_burst_fragments(loops, params):
    fragments = []
    for loop in loops:
        if len(loop) < 4:
            continue
        loop_center = loop_centroid(loop)
        stride = max(3, params.fragment_points - 3)
        frag_len = max(3, params.fragment_points)
        for start in range(0, len(loop), stride):
            chunk = loop[start : start + frag_len]
            if len(chunk) < 2:
                continue
            pts = np.asarray(chunk, dtype=np.float32)
            frag_center = pts.mean(axis=0)
            direction = frag_center - loop_center
            norm = np.linalg.norm(direction)
            if norm < 1e-5:
                angle = np.random.uniform(0.0, 2.0 * math.pi)
                direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
            else:
                direction = direction / norm
            tangent = np.array([-direction[1], direction[0]], dtype=np.float32)
            speed = params.burst_speed * np.random.uniform(0.7, 1.3)
            jitter = np.random.uniform(-params.burst_jitter, params.burst_jitter)
            velocity = direction * speed + tangent * jitter
            ttl = params.dissolve_time * np.random.uniform(0.8, 1.25)
            fragments.append(BurstFragment(points=pts.copy(), velocity=velocity, ttl=ttl))
    return fragments


def update_fragments(fragments, dt, params):
    active = []
    drag = params.burst_drag ** max(dt * 60.0, 1.0)
    for fragment in fragments:
        fragment.age += dt
        if fragment.age >= fragment.ttl:
            continue
        fragment.points += fragment.velocity * dt
        fragment.velocity *= drag
        active.append(fragment)
    return active


def draw_fragments(canvas, fragments, params):
    for fragment in fragments:
        if len(fragment.points) < 2:
            continue
        life = max(0.0, 1.0 - fragment.age / fragment.ttl)
        color_value = int(255 * (life ** 1.7))
        if color_value <= 0:
            continue
        pts = np.round(fragment.points).astype(np.int32).reshape((-1, 1, 2))
        color = (color_value, color_value, color_value)
        cv2.polylines(
            canvas,
            [pts],
            isClosed=False,
            color=color,
            thickness=params.fragment_thickness,
        )


def fit_to_screen(image, target_w, target_h):
    h, w = image.shape[:2]
    if w == 0 or h == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    # Cover the entire screen without stretching (crop overflow).
    scale = max(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x0 = max(0, (new_w - target_w) // 2)
    y0 = max(0, (new_h - target_h) // 2)
    cropped = resized[y0 : y0 + target_h, x0 : x0 + target_w]
    if cropped.shape[0] != target_h or cropped.shape[1] != target_w:
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        h2, w2 = cropped.shape[:2]
        canvas[:h2, :w2] = cropped
        return canvas
    return cropped


def apply_debug_camera_overlay(canvas, frame, opacity):
    if opacity <= 0.0:
        return canvas
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    alpha = float(np.clip(opacity, 0.0, 1.0))
    return cv2.addWeighted(gray_bgr, alpha, canvas, 1.0 - alpha, 0.0)


def extract_valid_loops(contours, params):
    valid_loops = []
    if not contours:
        return valid_loops

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours[: params.max_contours]:
        if cv2.contourArea(c) < params.min_contour_area:
            continue
        contour_pts = c.squeeze()
        if contour_pts.ndim != 2 or len(contour_pts) <= 10:
            continue
        valid_loops.append(resample_contour(contour_pts.tolist(), params.min_edge))
    return valid_loops


def main() -> None:
    root = tk.Tk()
    root.withdraw()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try a different index or check permissions.")

    mp_selfie = mp.solutions.selfie_segmentation
    segmenter = mp_selfie.SelfieSegmentation(model_selection=1)
    params = Params()
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    loops = []
    burst_fragments = []
    time_since_seed = 0.0
    time_since_growth_start = 0.0
    step = 0
    person_boxes = []
    fullscreen = False

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Human Outline Growth", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera", screen_w, screen_h)
    cv2.resizeWindow("Human Outline Growth", screen_w, screen_h)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if time_since_seed >= params.seed_interval:
                rects, _ = hog.detectMultiScale(
                    frame,
                    winStride=(8, 8),
                    padding=(8, 8),
                    scale=1.05,
                )
                person_boxes = rects

            contours = []
            kernel = np.ones((5, 5), np.uint8)
            if len(person_boxes) == 0:
                result = segmenter.process(rgb)
                mask = result.segmentation_mask
                if mask is not None:
                    binary = (mask > 0.1).astype(np.uint8) * 255
                    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
                    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=1)
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            else:
                for (x, y, w, h) in person_boxes:
                    x0 = max(0, x)
                    y0 = max(0, y)
                    x1 = min(frame.shape[1], x + w)
                    y1 = min(frame.shape[0], y + h)
                    if x1 - x0 < 10 or y1 - y0 < 10:
                        continue
                    roi = rgb[y0:y1, x0:x1]
                    result = segmenter.process(roi)
                    mask = result.segmentation_mask
                    if mask is None:
                        continue
                    binary = (mask > 0.1).astype(np.uint8) * 255
                    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
                    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=1)
                    roi_contours, _ = cv2.findContours(
                        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    for c in roi_contours:
                        c = c + np.array([[x0, y0]])
                        contours.append(c)

            dt = 1.0 / max(1.0, cap.get(cv2.CAP_PROP_FPS) or 30.0)
            time_since_seed += dt
            time_since_growth_start += dt
            burst_fragments = update_fragments(burst_fragments, dt, params)

            if time_since_seed >= params.seed_interval:
                new_loops = extract_valid_loops(contours, params)
                if new_loops:
                    burst_fragments.extend(create_burst_fragments(loops, params))
                    loops = new_loops
                    time_since_growth_start = 0.0
                elif loops:
                    burst_fragments.extend(create_burst_fragments(loops, params))
                    loops = []
                    time_since_growth_start = 0.0
                time_since_seed = 0.0

            if loops and time_since_growth_start >= params.growth_delay:
                for _ in range(params.iterations):
                    loops = [apply_differential_growth(loop, params) for loop in loops]
                    if step % params.smooth_every == 0:
                        loops = [smooth(loop, params.smooth_strength) for loop in loops]
                    step += 1

            canvas = np.zeros_like(frame)
            if params.debug_camera_overlay:
                canvas = apply_debug_camera_overlay(
                    canvas,
                    frame,
                    params.debug_overlay_opacity,
                )
            draw_fragments(canvas, burst_fragments, params)
            for loop in loops:
                if not loop:
                    continue
                pts = np.array(loop, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    canvas,
                    [pts],
                    isClosed=True,
                    color=params.outline_color,
                    thickness=2,
                )

            display_frame = fit_to_screen(frame, screen_w, screen_h)
            display_canvas = fit_to_screen(canvas, screen_w, screen_h)
            cv2.imshow("Camera", display_frame)
            cv2.imshow("Human Outline Growth", display_canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                burst_fragments.extend(create_burst_fragments(loops, params))
                loops = []
                time_since_growth_start = 0.0
                time_since_seed = params.seed_interval
            if key == ord("d"):
                params.debug_camera_overlay = not params.debug_camera_overlay
            if key == ord("f"):
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty(
                        "Human Outline Growth",
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN,
                    )
                else:
                    cv2.setWindowProperty(
                        "Human Outline Growth",
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_NORMAL,
                    )
    finally:
        segmenter.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
