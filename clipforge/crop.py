"""Dynamic 9:16 reframing.

We sample frames every ``crop.sample_every_n_frames``, look for faces (OpenCV Haar or DNN)
and fall back to the centroid of motion when there is no face. The resulting horizontal
positions are smoothed and speed-limited so the crop glides instead of snapping, then
reduced to a handful of keyframes and compiled into a single ffmpeg ``crop`` expression.

When two faces share a wide shot, the framing alternates between them on utterance
boundaries (we know when someone is talking from the word timestamps, and ``min_hold_s``
stops it turning into a ping-pong match).

Everything except ``sample_positions`` is pure and unit-tested.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .logging_utils import get_logger
from .transcribe import Transcript

log = get_logger("crop")


@dataclass(slots=True)
class Sample:
    t: float          # source time
    x: float          # horizontal centre, 0..1 of frame width
    faces: int = 0    # how many faces were found in this frame
    second_x: float | None = None  # centre of the second face, if any


@dataclass(slots=True)
class CropPlan:
    """Keyframed horizontal position of the crop window, in 0..1 of the frame width."""

    keyframes: list[tuple[float, float]] = field(default_factory=list)  # (out_time, centre)
    dynamic: bool = True
    reason: str = ""
    samples: int = 0

    def centre_at(self, t: float) -> float:
        if not self.keyframes:
            return 0.5
        if t <= self.keyframes[0][0]:
            return self.keyframes[0][1]
        if t >= self.keyframes[-1][0]:
            return self.keyframes[-1][1]
        for (t0, x0), (t1, x1) in zip(self.keyframes, self.keyframes[1:], strict=False):
            if t0 <= t <= t1:
                span = max(1e-6, t1 - t0)
                return x0 + (x1 - x0) * (t - t0) / span
        return self.keyframes[-1][1]


# --------------------------------------------------------------------------- pure helpers


def smooth(values: list[float], window: int) -> list[float]:
    """Centred moving average — this is what stops the crop from jerking."""
    if window <= 1 or len(values) < 2:
        return list(values)
    a = np.asarray(values, dtype=float)
    half = max(1, window // 2)
    padded = np.pad(a, (half, half), mode="edge")
    kernel = np.ones(half * 2 + 1) / (half * 2 + 1)
    return list(np.convolve(padded, kernel, mode="valid")[: len(a)])


def limit_speed(
    times: list[float], values: list[float], max_per_s: float, frame_width: int
) -> list[float]:
    """Cap how fast the crop may travel, in pixels per second."""
    if not values:
        return []
    max_frac_per_s = max_per_s / max(1, frame_width)
    out = [values[0]]
    for i in range(1, len(values)):
        dt = max(1e-6, times[i] - times[i - 1])
        allowed = max_frac_per_s * dt
        delta = values[i] - out[-1]
        if delta > allowed:
            delta = allowed
        elif delta < -allowed:
            delta = -allowed
        out.append(out[-1] + delta)
    return out


def simplify(
    times: list[float], values: list[float], min_delta: float = 0.008, min_gap_s: float = 0.5
) -> list[tuple[float, float]]:
    """Drop keyframes that would not move the crop noticeably; always keep both endpoints."""
    if not times:
        return []
    keys: list[tuple[float, float]] = [(times[0], values[0])]
    for t, x in zip(times[1:], values[1:], strict=False):
        last_t, last_x = keys[-1]
        if abs(x - last_x) >= min_delta and (t - last_t) >= min_gap_s:
            keys.append((t, x))
    if keys[-1][0] != times[-1]:
        keys.append((times[-1], values[-1]))
    return keys


def build_crop_expr(
    keyframes: list[tuple[float, float]], in_w: int, crop_w: int
) -> str:
    """Compile keyframes into an ffmpeg ``x`` expression with linear interpolation.

    The expression is clamped to [0, in_w - crop_w] so the crop window can never leave the
    frame, whatever the tracker produced.
    """
    max_x = max(0, in_w - crop_w)
    if not keyframes:
        return str(max_x // 2)

    def px(centre: float) -> float:
        return min(max_x, max(0.0, centre * in_w - crop_w / 2))

    if len(keyframes) == 1:
        return f"{px(keyframes[0][1]):.1f}"

    expr = f"{px(keyframes[-1][1]):.1f}"  # value after the last keyframe
    for (t0, x0), (t1, x1) in reversed(list(zip(keyframes, keyframes[1:], strict=False))):
        p0, p1 = px(x0), px(x1)
        span = max(1e-6, t1 - t0)
        segment = f"({p0:.1f}+({p1 - p0:.1f})*(t-{t0:.3f})/{span:.3f})"
        expr = f"if(lt(t,{t1:.3f}),{segment},{expr})"
    first_t, first_x = keyframes[0]
    return f"if(lt(t,{first_t:.3f}),{px(first_x):.1f},{expr})"


def pick_speaker_track(
    samples: list[Sample], transcript: Transcript, cfg: Config
) -> list[Sample]:
    """In two-face shots, follow whoever is likely speaking, switching on utterance edges.

    We cannot tell *which* mouth is moving, so we alternate between the two faces at
    utterance boundaries and hold each framing for at least ``min_hold_s``. That reads as
    shot-reverse-shot, which is what the eye expects from a dialogue scene.
    """
    sw = cfg.edit.crop.speaker_switch
    if not sw.enabled or not samples:
        return samples

    boundaries = [u.start for u in transcript.utterances] if transcript.utterances else []
    if not boundaries:
        return samples

    out: list[Sample] = []
    use_second = False
    last_switch = -1e9
    b_idx = 0
    for s in samples:
        while b_idx < len(boundaries) and boundaries[b_idx] <= s.t:
            if s.t - last_switch >= sw.min_hold_s:
                use_second = not use_second
                last_switch = s.t
            b_idx += 1
        if s.faces >= 2 and s.second_x is not None and use_second:
            out.append(Sample(s.t, s.second_x, s.faces, s.x))
        else:
            out.append(s)
    return out


# --------------------------------------------------------------------------- detection


def _load_face_detector(cfg: Config):
    import cv2

    crop_cfg = cfg.edit.crop
    if crop_cfg.detector == "dnn" and crop_cfg.dnn_proto and crop_cfg.dnn_model:
        net = cv2.dnn.readNetFromCaffe(
            str(cfg.resolve(crop_cfg.dnn_proto)), str(cfg.resolve(crop_cfg.dnn_model))
        )
        return ("dnn", net)
    path = cv2.data.haarcascades + crop_cfg.haar_cascade
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError(f"could not load Haar cascade: {path}")
    return ("haar", cascade)


def _detect_faces(kind: str, detector, frame, min_size: int) -> list[tuple[int, int, int, int]]:
    import cv2

    if kind == "dnn":
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        detector.setInput(blob)
        detections = detector.forward()
        boxes = []
        for i in range(detections.shape[2]):
            if float(detections[0, 0, i, 2]) < 0.6:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            if (x2 - x1) >= min_size:
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = detector.detectMultiScale(gray, 1.1, 5, minSize=(min_size, min_size))
    return [tuple(map(int, f)) for f in found]


def sample_positions(
    video_path: str,
    start_s: float,
    end_s: float,
    cfg: Config,
    sample_width: int = 480,
) -> list[Sample]:
    """Decode the segment at reduced size and track faces (or motion) horizontally."""
    import cv2

    crop_cfg = cfg.edit.crop
    fps_out = max(0.5, cfg.edit.fps / max(1, crop_cfg.sample_every_n_frames))
    cmd = [
        cfg.runtime.ffmpeg,
        "-v", "error",
        "-ss", f"{start_s:.3f}",
        "-t", f"{end_s - start_s:.3f}",
        "-i", video_path,
        "-vf", f"fps={fps_out},scale={sample_width}:-2",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 and not proc.stdout:
        raise RuntimeError(f"crop sampling failed: {proc.stderr.decode(errors='ignore')[-400:]}")

    probe_h = None
    # Frame height is unknown up front (scale=-2 keeps the aspect); derive it from the size.
    total = len(proc.stdout)
    for h in range(sample_width, 0, -2):
        if total % (sample_width * h * 3) == 0:
            probe_h = h
            break
    if not probe_h:
        raise RuntimeError("could not infer sampled frame height")

    frame_bytes = sample_width * probe_h * 3
    n_frames = total // frame_bytes
    buf = np.frombuffer(proc.stdout, dtype=np.uint8)

    kind, detector = _load_face_detector(cfg)
    min_size = max(8, int(crop_cfg.min_face_size * sample_width / max(1, cfg.edit.target_w)))

    samples: list[Sample] = []
    prev_gray = None
    for i in range(n_frames):
        frame = buf[i * frame_bytes : (i + 1) * frame_bytes].reshape(probe_h, sample_width, 3)
        t = start_s + i / fps_out
        faces = _detect_faces(kind, detector, frame, min_size)
        if faces:
            faces.sort(key=lambda f: f[2] * f[3], reverse=True)
            cx = (faces[0][0] + faces[0][2] / 2) / sample_width
            second = None
            if len(faces) > 1:
                second = (faces[1][0] + faces[1][2] / 2) / sample_width
            samples.append(Sample(t, float(cx), len(faces), second))
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            column_energy = diff.sum(axis=0).astype(float)
            if column_energy.sum() > 0:
                cx = float((column_energy * np.arange(sample_width)).sum() / column_energy.sum())
                samples.append(Sample(t, cx / sample_width, 0, None))
        prev_gray = gray

    return samples


# --------------------------------------------------------------------------- orchestration


def plan_crop(
    video_path: str,
    start_s: float,
    end_s: float,
    cfg: Config,
    transcript: Transcript,
    retime,
) -> CropPlan:
    """Build the crop plan for one segment, in OUTPUT time (post dead-air trimming)."""
    crop_cfg = cfg.edit.crop
    if crop_cfg.mode == "center":
        return CropPlan(keyframes=[(0.0, 0.5)], dynamic=False, reason="center crop (configured)")

    try:
        samples = sample_positions(video_path, start_s, end_s, cfg)
    except Exception as exc:
        log.warning("crop tracking failed (%s) — falling back to center crop", exc)
        return CropPlan(keyframes=[(0.0, 0.5)], dynamic=False, reason=f"fallback: {exc}")

    if not samples:
        return CropPlan(
            keyframes=[(0.0, 0.5)], dynamic=False, reason="no faces or motion detected"
        )

    samples = pick_speaker_track(samples, transcript, cfg)
    samples = [s for s in samples if not retime.is_cut(s.t)]
    if not samples:
        return CropPlan(keyframes=[(0.0, 0.5)], dynamic=False, reason="all samples were trimmed")

    out_times = [retime.map_time(s.t) for s in samples]
    xs = smooth([s.x for s in samples], crop_cfg.smoothing_window)
    xs = limit_speed(out_times, xs, crop_cfg.max_pan_px_per_s, cfg.edit.target_w)
    keys = simplify(out_times, xs)
    return CropPlan(
        keyframes=keys, dynamic=True, reason="face/motion tracking", samples=len(samples)
    )
