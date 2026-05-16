const params = {
  minEdge: 10,
  minContourArea: 2500,
  maxEdge: 24,
  curvatureThreshold: 0.55,
  maxNewPoints: 20,
  repelRadius: 26,
  repelStrength: 0.25,
  relaxStrength: 0.35,
  iterations: 1,
  timeStep: 0.5,
  maxPoints: 1000,
  smoothStrength: 0.3,
  smoothEvery: 3,
  seedInterval: 3,
  growthDelay: 0.8,
  maxContours: 3,
  dissolveTime: 1.8,
  fragmentPoints: 12,
  burstSpeed: 95,
  burstJitter: 45,
  burstDrag: 0.92,
  fragmentThickness: 2,
  outlineThickness: 2,
  cameraBackground: false,
};

const video = document.getElementById("camera");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d", { alpha: false });
const statusEl = document.getElementById("status");
const controls = document.getElementById("controls");
const menuToggle = document.getElementById("menu-toggle");
const overlayButton = document.getElementById("overlay");

const maskCanvas = document.createElement("canvas");
const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });

let segmentation = null;
let latestMask = null;
let loops = [];
let burstFragments = [];
let timeSinceSeed = params.seedInterval;
let timeSinceGrowthStart = 0;
let step = 0;
let lastFrameTime = performance.now();
let processing = false;
let forceReseed = false;
let stopped = false;

function setStatus(message) {
  statusEl.textContent = message;
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function resampleContour(points, spacing) {
  if (points.length < 2) return points;
  const resampled = [{ ...points[0] }];
  let carry = 0;

  for (let i = 1; i < points.length; i += 1) {
    const a = points[i - 1];
    const b = points[i];
    const segLen = distance(a, b);
    if (segLen === 0) continue;

    const dirX = (b.x - a.x) / segLen;
    const dirY = (b.y - a.y) / segLen;
    let d = spacing - carry;

    while (d <= segLen) {
      resampled.push({ x: a.x + dirX * d, y: a.y + dirY * d });
      d += spacing;
    }
    carry = d - segLen;
  }

  return resampled;
}

function applyDifferentialGrowth(points) {
  if (points.length < 2) return points;

  const count = points.length;
  const forces = Array.from({ length: count }, () => ({ x: 0, y: 0 }));

  for (let i = 0; i < count; i += 1) {
    const curr = points[i];
    const prev = points[i > 0 ? i - 1 : count - 1];
    const next = points[i < count - 1 ? i + 1 : 0];
    forces[i].x += (prev.x - curr.x) * params.relaxStrength;
    forces[i].y += (prev.y - curr.y) * params.relaxStrength;
    forces[i].x += (next.x - curr.x) * params.relaxStrength;
    forces[i].y += (next.y - curr.y) * params.relaxStrength;
  }

  const radius = params.repelRadius;
  const radiusSq = radius * radius;
  const cellSize = Math.max(4, radius);
  const grid = new Map();

  for (let i = 0; i < count; i += 1) {
    const p = points[i];
    const key = `${Math.floor(p.x / cellSize)},${Math.floor(p.y / cellSize)}`;
    if (!grid.has(key)) grid.set(key, []);
    grid.get(key).push(i);
  }

  for (let i = 0; i < count; i += 1) {
    const a = points[i];
    const cellX = Math.floor(a.x / cellSize);
    const cellY = Math.floor(a.y / cellSize);

    for (let gx = cellX - 1; gx <= cellX + 1; gx += 1) {
      for (let gy = cellY - 1; gy <= cellY + 1; gy += 1) {
        const cell = grid.get(`${gx},${gy}`);
        if (!cell) continue;

        for (const j of cell) {
          if (j <= i) continue;
          const b = points[j];
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const distSq = dx * dx + dy * dy;
          if (distSq === 0 || distSq > radiusSq) continue;

          const dist = Math.sqrt(distSq);
          const force = (1 - dist / radius) * params.repelStrength;
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          forces[i].x -= fx;
          forces[i].y -= fy;
          forces[j].x += fx;
          forces[j].y += fy;
        }
      }
    }
  }

  for (let i = 0; i < count; i += 1) {
    points[i].x += forces[i].x * params.timeStep;
    points[i].y += forces[i].y * params.timeStep;
  }

  const candidates = [];
  for (let i = 0; i < count; i += 1) {
    const a = points[i];
    const b = points[(i + 1) % count];
    let insert = distance(a, b) > params.maxEdge;

    if (count > 2) {
      const prev = points[i > 0 ? i - 1 : count - 1];
      const next = points[(i + 1) % count];
      const v1x = prev.x - a.x;
      const v1y = prev.y - a.y;
      const v2x = next.x - a.x;
      const v2y = next.y - a.y;
      const len1 = Math.hypot(v1x, v1y);
      const len2 = Math.hypot(v2x, v2y);
      if (len1 > 0 && len2 > 0) {
        const cosTheta = (v1x * v2x + v1y * v2y) / (len1 * len2);
        if (cosTheta < params.curvatureThreshold) insert = true;
      }
    }

    if (insert) candidates.push(i);
  }

  const maxInserts = Math.min(
    params.maxNewPoints,
    params.maxPoints - points.length,
    candidates.length,
  );
  const selected = new Set();
  while (selected.size < maxInserts) {
    const idx = candidates[Math.floor(Math.random() * candidates.length)];
    selected.add(idx);
  }

  const newPoints = [];
  for (let i = 0; i < count; i += 1) {
    const a = points[i];
    const b = points[(i + 1) % count];
    newPoints.push(a);
    if (selected.has(i)) {
      newPoints.push({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
    }
  }

  return newPoints;
}

function smooth(points) {
  if (points.length < 3) return points;
  const count = points.length;
  const smoothed = [];

  for (let i = 0; i < count; i += 1) {
    const prev = points[i > 0 ? i - 1 : count - 1];
    const curr = points[i];
    const next = points[(i + 1) % count];
    const targetX = (prev.x + curr.x + next.x) / 3;
    const targetY = (prev.y + curr.y + next.y) / 3;
    smoothed.push({
      x: curr.x + (targetX - curr.x) * params.smoothStrength,
      y: curr.y + (targetY - curr.y) * params.smoothStrength,
    });
  }

  return smoothed;
}

function loopCentroid(points) {
  if (!points.length) return { x: 0, y: 0 };
  const sum = points.reduce((acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y }), { x: 0, y: 0 });
  return { x: sum.x / points.length, y: sum.y / points.length };
}

function createBurstFragments(sourceLoops) {
  const fragments = [];

  for (const loop of sourceLoops) {
    if (loop.length < 4) continue;
    const center = loopCentroid(loop);
    const stride = Math.max(3, params.fragmentPoints - 3);
    const fragLen = Math.max(3, params.fragmentPoints);

    for (let start = 0; start < loop.length; start += stride) {
      const chunk = loop.slice(start, start + fragLen);
      if (chunk.length < 2) continue;
      const fragCenter = loopCentroid(chunk);
      let dx = fragCenter.x - center.x;
      let dy = fragCenter.y - center.y;
      const norm = Math.hypot(dx, dy);
      if (norm < 0.00001) {
        const angle = Math.random() * Math.PI * 2;
        dx = Math.cos(angle);
        dy = Math.sin(angle);
      } else {
        dx /= norm;
        dy /= norm;
      }

      const speed = params.burstSpeed * (0.7 + Math.random() * 0.6);
      const jitter = -params.burstJitter + Math.random() * params.burstJitter * 2;
      fragments.push({
        points: chunk.map((p) => ({ ...p })),
        velocity: { x: dx * speed - dy * jitter, y: dy * speed + dx * jitter },
        ttl: params.dissolveTime * (0.8 + Math.random() * 0.45),
        age: 0,
      });
    }
  }

  return fragments;
}

function updateFragments(dt) {
  const drag = params.burstDrag ** Math.max(dt * 60, 1);
  burstFragments = burstFragments.filter((fragment) => {
    fragment.age += dt;
    if (fragment.age >= fragment.ttl) return false;
    for (const point of fragment.points) {
      point.x += fragment.velocity.x * dt;
      point.y += fragment.velocity.y * dt;
    }
    fragment.velocity.x *= drag;
    fragment.velocity.y *= drag;
    return true;
  });
}

function drawFragments() {
  ctx.lineWidth = params.fragmentThickness;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  for (const fragment of burstFragments) {
    if (fragment.points.length < 2) continue;
    const life = Math.max(0, 1 - fragment.age / fragment.ttl);
    const value = Math.floor(255 * life ** 1.7);
    if (value <= 0) continue;
    ctx.strokeStyle = `rgb(${value}, ${value}, ${value})`;
    drawLoop(fragment.points, false);
  }
}

function drawLoop(points, closed) {
  if (!points.length) return;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i += 1) {
    ctx.lineTo(points[i].x, points[i].y);
  }
  if (closed) ctx.closePath();
  ctx.stroke();
}

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(window.innerWidth * dpr));
  const height = Math.max(1, Math.floor(window.innerHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function videoFit() {
  const vw = video.videoWidth || 640;
  const vh = video.videoHeight || 480;
  const scale = Math.max(canvas.width / vw, canvas.height / vh);
  const width = vw * scale;
  const height = vh * scale;
  const x = (canvas.width - width) / 2;
  const y = (canvas.height - height) / 2;
  return { x, y, width, height, scale };
}

function drawMirroredVideo(opacity) {
  const fit = videoFit();
  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, canvas.width - fit.x - fit.width, fit.y, fit.width, fit.height);
  ctx.restore();
}

function imageToCanvasPoint(px, py, maskWidth, maskHeight) {
  const fit = videoFit();
  const videoX = (px / maskWidth) * fit.width;
  const videoY = (py / maskHeight) * fit.height;
  return {
    x: canvas.width - (fit.x + videoX),
    y: fit.y + videoY,
  };
}

function extractValidLoops(mask) {
  const maskWidth = 192;
  const maskHeight = Math.max(1, Math.round(maskWidth * (video.videoHeight / video.videoWidth)));
  maskCanvas.width = maskWidth;
  maskCanvas.height = maskHeight;
  maskCtx.clearRect(0, 0, maskWidth, maskHeight);
  maskCtx.drawImage(mask, 0, 0, maskWidth, maskHeight);

  const image = maskCtx.getImageData(0, 0, maskWidth, maskHeight);
  const binary = new Uint8Array(maskWidth * maskHeight);
  for (let i = 0; i < binary.length; i += 1) {
    binary[i] = image.data[i * 4] > 26 ? 1 : 0;
  }

  const components = findComponents(binary, maskWidth, maskHeight)
    .filter((component) => component.area * (canvas.width / maskWidth) * (canvas.height / maskHeight) >= params.minContourArea)
    .sort((a, b) => b.area - a.area)
    .slice(0, params.maxContours);

  return components
    .map((component) => componentToLoop(component, binary, maskWidth, maskHeight))
    .filter((loop) => loop.length > 10)
    .map((loop) => resampleContour(loop, params.minEdge));
}

function findComponents(binary, width, height) {
  const visited = new Uint8Array(binary.length);
  const components = [];
  const queue = [];

  for (let idx = 0; idx < binary.length; idx += 1) {
    if (!binary[idx] || visited[idx]) continue;

    queue.length = 0;
    queue.push(idx);
    visited[idx] = 1;
    const pixels = [];

    for (let head = 0; head < queue.length; head += 1) {
      const current = queue[head];
      pixels.push(current);
      const x = current % width;
      const y = Math.floor(current / width);
      const neighbors = [
        current - 1,
        current + 1,
        current - width,
        current + width,
      ];

      for (const next of neighbors) {
        if (next < 0 || next >= binary.length || visited[next] || !binary[next]) continue;
        const nx = next % width;
        const ny = Math.floor(next / width);
        if (Math.abs(nx - x) + Math.abs(ny - y) !== 1) continue;
        visited[next] = 1;
        queue.push(next);
      }
    }

    components.push({ pixels, area: pixels.length });
  }

  return components;
}

function componentToLoop(component, binary, width, height) {
  const boundary = [];
  const boundarySet = new Set();

  for (const idx of component.pixels) {
    const x = idx % width;
    const y = Math.floor(idx / width);
    const edge =
      x === 0 ||
      y === 0 ||
      x === width - 1 ||
      y === height - 1 ||
      !binary[idx - 1] ||
      !binary[idx + 1] ||
      !binary[idx - width] ||
      !binary[idx + width];

    if (edge) {
      boundary.push({ x, y });
      boundarySet.add(`${x},${y}`);
    }
  }

  if (!boundary.length) return [];

  const ordered = traceBoundary(boundary, boundarySet);
  return ordered.map((point) => imageToCanvasPoint(point.x + 0.5, point.y + 0.5, width, height));
}

function traceBoundary(boundary, boundarySet) {
  const start = boundary.reduce((best, point) => {
    if (point.y < best.y) return point;
    if (point.y === best.y && point.x < best.x) return point;
    return best;
  }, boundary[0]);

  const visited = new Set();
  const path = [{ ...start }];
  visited.add(`${start.x},${start.y}`);

  let current = start;
  let previousDirection = { x: 1, y: 0 };

  while (visited.size < boundary.length) {
    const next = chooseNextBoundaryPoint(current, previousDirection, boundary, boundarySet, visited);
    if (!next) break;

    previousDirection = { x: next.x - current.x, y: next.y - current.y };
    current = next;
    path.push({ ...current });
    visited.add(`${current.x},${current.y}`);

    if (path.length > boundary.length * 2) break;
  }

  return path.length > 10 ? path : boundary;
}

function chooseNextBoundaryPoint(current, previousDirection, boundary, boundarySet, visited) {
  const candidates = [];

  for (let dy = -1; dy <= 1; dy += 1) {
    for (let dx = -1; dx <= 1; dx += 1) {
      if (dx === 0 && dy === 0) continue;
      const x = current.x + dx;
      const y = current.y + dy;
      const key = `${x},${y}`;
      if (boundarySet.has(key) && !visited.has(key)) {
        candidates.push({ x, y, dx, dy, distance: Math.hypot(dx, dy) });
      }
    }
  }

  if (!candidates.length) {
    let nearest = null;
    for (const point of boundary) {
      const key = `${point.x},${point.y}`;
      if (visited.has(key)) continue;
      const dx = point.x - current.x;
      const dy = point.y - current.y;
      const distanceToPoint = Math.hypot(dx, dy);
      if (distanceToPoint > 3) continue;
      const candidate = { ...point, dx, dy, distance: distanceToPoint };
      if (!nearest || scoreBoundaryCandidate(candidate, previousDirection) < scoreBoundaryCandidate(nearest, previousDirection)) {
        nearest = candidate;
      }
    }
    return nearest;
  }

  candidates.sort(
    (a, b) => scoreBoundaryCandidate(a, previousDirection) - scoreBoundaryCandidate(b, previousDirection),
  );
  return candidates[0];
}

function scoreBoundaryCandidate(candidate, previousDirection) {
  const prevLength = Math.hypot(previousDirection.x, previousDirection.y) || 1;
  const nextLength = Math.hypot(candidate.dx, candidate.dy) || 1;
  const dot = (previousDirection.x * candidate.dx + previousDirection.y * candidate.dy) / (prevLength * nextLength);
  return candidate.distance + (1 - dot) * 0.35;
}

function reseedFromMask() {
  if (!latestMask) return;
  const newLoops = extractValidLoops(latestMask);
  if (newLoops.length) {
    burstFragments.push(...createBurstFragments(loops));
    loops = newLoops;
    timeSinceGrowthStart = 0;
  } else if (loops.length) {
    burstFragments.push(...createBurstFragments(loops));
    loops = [];
    timeSinceGrowthStart = 0;
  }
  timeSinceSeed = 0;
}

function render(now) {
  if (stopped) return;
  resizeCanvas();

  const dt = Math.min(0.05, (now - lastFrameTime) / 1000 || 1 / 30);
  lastFrameTime = now;
  timeSinceSeed += dt;
  timeSinceGrowthStart += dt;
  updateFragments(dt);

  if (forceReseed || timeSinceSeed >= params.seedInterval) {
    reseedFromMask();
    forceReseed = false;
  }

  if (loops.length && timeSinceGrowthStart >= params.growthDelay) {
    for (let i = 0; i < params.iterations; i += 1) {
      loops = loops.map(applyDifferentialGrowth);
      if (step % params.smoothEvery === 0) {
        loops = loops.map(smooth);
      }
      step += 1;
    }
  }

  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (params.cameraBackground && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    drawMirroredVideo(1);
  }

  drawFragments();
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = params.outlineThickness;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const loop of loops) {
    drawLoop(loop, true);
  }

  requestAnimationFrame(render);
}

async function processCamera() {
  if (processing || stopped || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  processing = true;
  try {
    await segmentation.send({ image: video });
  } catch (error) {
    setStatus(error.message || "Camera processing failed.");
  } finally {
    processing = false;
  }
}

async function start() {
  if (!window.SelfieSegmentation) {
    setStatus("MediaPipe could not be loaded.");
    return;
  }

  segmentation = new SelfieSegmentation({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`,
  });
  segmentation.setOptions({ modelSelection: 1 });
  segmentation.onResults((results) => {
    latestMask = results.segmentationMask;
    setStatus("");
  });

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    setInterval(processCamera, 33);
    requestAnimationFrame(render);
  } catch (error) {
    setStatus(error.message || "Camera permission was not granted.");
  }
}

menuToggle.addEventListener("click", () => {
  const isOpen = controls.classList.toggle("open");
  menuToggle.setAttribute("aria-expanded", String(isOpen));
});

document.getElementById("reseed").addEventListener("click", () => {
  burstFragments.push(...createBurstFragments(loops));
  loops = [];
  timeSinceGrowthStart = 0;
  forceReseed = true;
});

overlayButton.addEventListener("click", () => {
  params.cameraBackground = !params.cameraBackground;
  overlayButton.setAttribute("aria-pressed", String(params.cameraBackground));
});

document.getElementById("fullscreen").addEventListener("click", () => {
  if (document.fullscreenElement) {
    document.exitFullscreen();
  } else {
    document.documentElement.requestFullscreen();
  }
});

window.addEventListener("resize", resizeCanvas);
window.addEventListener("keydown", (event) => {
  if (event.key === "r" || event.key === "R") {
    burstFragments.push(...createBurstFragments(loops));
    loops = [];
    timeSinceGrowthStart = 0;
    forceReseed = true;
  }
  if (event.key === "d" || event.key === "D") {
    params.cameraBackground = !params.cameraBackground;
    overlayButton.setAttribute("aria-pressed", String(params.cameraBackground));
  }
  if (event.key === "f" || event.key === "F") {
    document.getElementById("fullscreen").click();
  }
  if (event.key === "Escape" || event.key === "q" || event.key === "Q") {
    stopped = true;
    const stream = video.srcObject;
    if (stream) stream.getTracks().forEach((track) => track.stop());
    setStatus("Camera stopped.");
  }
});

document.querySelectorAll("[data-param]").forEach((slider) => {
  const output = document.querySelector(`output[for="${slider.id}"]`);
  const decimals = slider.step.includes(".") ? 2 : 0;

  const update = () => {
    const value = Number(slider.value);
    params[slider.dataset.param] = value;
    if (output) output.textContent = value.toFixed(decimals);
  };

  slider.addEventListener("input", update);
  update();
});

resizeCanvas();
start();
