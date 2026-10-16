#!/usr/bin/env python3
"""Live webcam license-plate detector, confidence scorer, and human-feedback learner.

Focuses strictly on locating license plates, enhancing the plate image,
reading all printed contents, computing an accuracy confidence score, and
learning from human intervention (press Y to confirm or N to reject).
Plate reading runs on a background thread so the video feed stays fast.

Everything this script reads or writes stays inside this project folder.
"""

import argparse
from collections import deque
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets.coco_classes import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model
from yolox.utils.boxes import postprocess


def parse_args():
    """Read command-line options for the camera window."""
    parser = argparse.ArgumentParser(
        description="Detect and read license plates on the webcam. Quit with q or Esc."
    )
    parser.add_argument("--cam", type=int, default=0, help="camera index")
    parser.add_argument(
        "--name",
        default="yolox-s",
        help="model name, for example yolox-s, yolox-m, yolox-tiny, yolox-nano",
    )
    parser.add_argument(
        "--ckpt",
        default=str(ROOT / "weights" / "yolox_s.pth"),
        help="checkpoint path inside this project",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "gpu"),
        help="cpu, or gpu when CUDA is available",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="also write the annotated video under YOLOX_outputs/",
    )
    return parser.parse_args()


def resolve_inside_project(path):
    """Turn a checkpoint path into an absolute path inside this folder."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    resolved = resolved.resolve()
    if ROOT != resolved and ROOT not in resolved.parents:
        raise SystemExit(f"Refusing to use a path outside this folder: {resolved}")
    return resolved


def open_camera(cam_id):
    """Open a webcam and return it after the first real frame arrives."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(cam_id, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
        cap.release()
    raise SystemExit(
        f"Could not open camera {cam_id}. No webcam is available to this program.\n"
        "Plug in a camera, allow desktop apps to use it in Windows privacy settings, "
        "then try again. For a second camera: .\\run_camera.bat --cam 1"
    )


def load_model(name, ckpt_path, conf, nms, device):
    """Build a YOLOX detector, load weights, and fuse Conv+BN layers for faster inference."""
    exp = get_exp(None, name)
    exp.test_conf = conf
    exp.nmsthre = nms
    exp.test_size = (416, 416)
    model = exp.get_model()
    model.eval()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model = fuse_model(model)

    if device == "gpu":
        if not torch.cuda.is_available():
            raise SystemExit("No CUDA GPU is available. Run with --device cpu.")
        model.cuda()
    return model, exp


VEHICLE_NAMES = {"car", "truck", "bus", "motorcycle"}


def detect_vehicles(model, exp, frame, preproc, device):
    """Find only vehicle bounding boxes to help locate plates on distant cars or trucks."""
    ratio = min(exp.test_size[0] / frame.shape[0], exp.test_size[1] / frame.shape[1])
    tensor, _ = preproc(frame, None, exp.test_size)
    tensor = torch.from_numpy(tensor).unsqueeze(0).float()
    if device == "gpu":
        tensor = tensor.cuda()

    with torch.inference_mode():
        outputs = model(tensor)
        outputs = postprocess(
            outputs, exp.num_classes, exp.test_conf, exp.nmsthre, class_agnostic=True
        )

    if outputs[0] is None:
        return []

    output = outputs[0].cpu()
    boxes = output[:, 0:4] / ratio
    scores = output[:, 4] * output[:, 5]
    classes = output[:, 6]
    vehicles = []
    height, width = frame.shape[:2]
    for box, score, cls_id in zip(boxes, scores, classes):
        if float(score) < exp.test_conf:
            continue
        name = COCO_CLASSES[int(cls_id)]
        if name not in VEHICLE_NAMES:
            continue
        x0, y0, x1, y1 = [int(value) for value in box[:4]]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 - x0 >= 40 and y1 - y0 >= 40:
            vehicles.append((x0, y0, x1, y1))
    return vehicles


def analyze(model, exp, frame, preproc, device, run_vehicle_pass=False):
    """Focus only on license plate regions without drawing unrelated COCO objects."""
    from read_plate import detect_plate_regions

    drawn = frame.copy()
    plate_candidates = detect_plate_regions(frame)
    vehicles = detect_vehicles(model, exp, frame, preproc, device) if run_vehicle_pass else []
    return drawn, plate_candidates, vehicles


class PlateWatcher:
    """Enhance, score, and read license plates on a background thread with human feedback."""

    def __init__(self):
        """Start the background plate reader."""
        from read_plate import (
            compute_confidence,
            enhance_plate_image,
            read_plate_in_frame,
            record_human_feedback,
        )

        self._read = read_plate_in_frame
        self._enhance = enhance_plate_image
        self._feedback_fn = record_human_feedback
        self._compute_conf = compute_confidence
        self._lock = threading.Lock()
        self._pending = None
        self._stop = False
        self.reading = None
        self.enhanced_crop = None
        self.updated = 0.0
        self.feedback_banner = None
        self.feedback_banner_time = 0.0
        self._recent_tokens = deque(maxlen=12)
        self._announced = None
        self._thread = threading.Thread(target=self._loop, name="plate-reader", daemon=True)
        self._thread.start()

    def submit(self, frame, vehicles):
        """Queue one frame if the reader is idle. Older queued frames are skipped."""
        with self._lock:
            if self._pending is None:
                self._pending = (frame.copy(), list(vehicles))

    def snapshot(self):
        """Return (reading, enhanced_crop, feedback_banner)."""
        with self._lock:
            banner = None
            if self.feedback_banner and time.perf_counter() - self.feedback_banner_time < 5.0:
                banner = self.feedback_banner
            if self.reading is None or time.perf_counter() - self.updated > 8:
                return None, None, banner
            return self.reading, self.enhanced_crop, banner

    def apply_human_feedback(self, is_correct):
        """Record human confirmation (Y) or rejection (N) for the active plate reading."""
        with self._lock:
            current = dict(self.reading) if self.reading else None
            recent = list(self._recent_tokens)
        if not current or not current.get("normalized"):
            with self._lock:
                self.feedback_banner = "No active plate reading to confirm or reject yet."
                self.feedback_banner_time = time.perf_counter()
            return

        extra_items = list(current.get("all_items") or [])
        for tok in recent:
            extra_items.append({"text": tok})
        _, msg = self._feedback_fn(current, items=extra_items, is_correct=is_correct)
        print(f"[Human Feedback] {msg}")

        with self._lock:
            self.feedback_banner = msg
            self.feedback_banner_time = time.perf_counter()
            if not is_correct:
                self.reading = None
                self.enhanced_crop = None
                self._announced = None
            elif self.reading:
                self.reading["human_confirmations"] = int(self.reading.get("human_confirmations", 0)) + 1
                conf, level, details = self._compute_conf(
                    self.reading, self.reading.get("all_items") or []
                )
                self.reading["confidence"] = conf
                self.reading["confidence_level"] = level
                self.reading["confidence_details"] = details
                self.updated = time.perf_counter()

    def stop(self):
        """Ask the reader thread to exit and wait briefly for it."""
        self._stop = True
        self._thread.join(timeout=1.0)

    def _loop(self):
        """Pull queued frames, enhance and read any visible plate, and log new plates."""
        while not self._stop:
            job = None
            with self._lock:
                if self._pending is not None:
                    job = self._pending
                    self._pending = None
            if job is None:
                time.sleep(0.03)
                continue
            frame, vehicles = job
            try:
                reading = self._read(frame, vehicles)
            except Exception as exc:
                print(f"Plate reader error: {exc}")
                continue
            if not reading or not reading.get("normalized"):
                continue

            crop = frame
            if reading.get("plate_box"):
                px0, py0, px1, py1 = reading["plate_box"]
                fh, fw = frame.shape[:2]
                pad_x = int((px1 - px0) * 0.08)
                pad_y = int((py1 - py0) * 0.12)
                cx0, cy0 = max(0, px0 - pad_x), max(0, py0 - pad_y)
                cx1, cy1 = min(fw, px1 + pad_x), min(fh, py1 + pad_y)
                if cx1 - cx0 >= 40 and cy1 - cy0 >= 20:
                    crop = frame[cy0:cy1, cx0:cx1]
            enhanced_preview = self._enhance(crop)

            with self._lock:
                self._recent_tokens.append(reading.get("raw_ocr_number") or reading["number"])
                # Keep higher-confidence reading if the same plate was just read a moment ago
                if (
                    self.reading
                    and time.perf_counter() - self.updated < 2.5
                    and self.reading.get("normalized") == reading["normalized"]
                    and self.reading.get("confidence", 0.0) > reading.get("confidence", 0.0)
                ):
                    reading["confidence"] = self.reading["confidence"]
                    reading["confidence_level"] = self.reading["confidence_level"]
                self.reading = reading
                self.enhanced_crop = enhanced_preview
                self.updated = time.perf_counter()

            if reading["normalized"] != self._announced:
                self._announced = reading["normalized"]
                from read_plate import location_text

                place = location_text(reading) or "location not printed clearly"
                contents = " | ".join(reading.get("contents") or [])
                conf = reading.get("confidence", 0.0)
                level = reading.get("confidence_level", "")
                print(
                    f"Plate {reading['number']} | Confidence: {conf:.1f}% ({level}) | "
                    f"{place} | Contents: {contents}"
                )


def paint_plate(frame, reading, enhanced_crop=None, candidate_boxes=None, feedback_banner=None):
    """Draw license plate boxes, confidence score, contents, and human feedback controls."""
    from read_plate import location_text

    fh, fw = frame.shape[:2]
    _label(
        frame,
        "Human Feedback: [Y] Confirm Correct   [N] Reject Wrong   [Q] Quit",
        (10, fh - 14),
        scale=0.58,
        color=(200, 230, 255),
    )
    if feedback_banner:
        _label(frame, feedback_banner, (10, fh - 46), scale=0.62, color=(100, 255, 180))

    if reading is None:
        for x0, y0, x1, y1 in (candidate_boxes or [])[:2]:
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 180, 255), 2)
        _label(frame, "Looking for a license plate...", (10, 70))
        return frame

    for item in reading.get("content_items") or []:
        ix0, iy0 = int(item["x"]), int(item["y"])
        ix1, iy1 = int(item["x"] + item["w"]), int(item["y"] + item["h"])
        cv2.rectangle(frame, (ix0, iy0), (ix1, iy1), (220, 180, 40), 2)

    items = reading.get("number_items") or []
    if items:
        x0 = int(min(item["x"] for item in items))
        y0 = int(min(item["y"] for item in items))
        x1 = int(max(item["x"] + item["w"] for item in items))
        y1 = int(max(item["y"] + item["h"] for item in items))
        cv2.rectangle(frame, (x0, y0), (x1, y1), (80, 220, 80), 3)

    conf = reading.get("confidence", 0.0)
    level = reading.get("confidence_level", "")
    parts = [f"{reading['number']} ({conf:.1f}% {level})"]
    if reading.get("plate_type"):
        parts.append(reading["plate_type"])
    place = location_text(reading)
    if place:
        parts.append(place)
    if reading.get("expiration"):
        parts.append(f"Exp {reading['expiration']}")
    _label(frame, "  |  ".join(parts), (10, 70))

    details = []
    if reading.get("slogan"):
        details.append(f'"{reading["slogan"]}"')
    if reading.get("sticker_numbers"):
        details.append(f"Sticker: {', '.join(reading['sticker_numbers'][:2])}")
    if details:
        _label(frame, "  |  ".join(details), (10, 112), scale=0.65)

    if enhanced_crop is not None and enhanced_crop.size > 0:
        thumb_w = min(260, int(fw * 0.28))
        thumb_h = max(40, int(enhanced_crop.shape[0] * (thumb_w / max(enhanced_crop.shape[1], 1))))
        if thumb_h + 40 < fh and thumb_w + 20 < fw:
            thumb = cv2.resize(enhanced_crop, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            x0 = fw - thumb_w - 12
            y0 = 36
            cv2.rectangle(
                frame, (x0 - 3, y0 - 22), (x0 + thumb_w + 3, y0 + thumb_h + 3), (0, 0, 0), -1
            )
            cv2.putText(
                frame,
                "Enhanced Plate",
                (x0 + 4, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (80, 220, 80),
                1,
            )
            frame[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = thumb

    return frame


def _label(frame, text, origin, scale=0.8, color=(80, 220, 80)):
    """Draw one line of text on a black background so it stays readable."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    x, y = origin
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(
        frame,
        (x - 4, y - text_height - 8),
        (x + text_width + 6, y + baseline + 4),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)


def main():
    """Open the webcam, locate and enhance license plates, and overlay plate contents."""
    args = parse_args()
    ckpt_path = resolve_inside_project(args.ckpt)
    if not ckpt_path.is_file():
        raise SystemExit(
            f"Missing checkpoint: {ckpt_path}\n"
            "Keep yolox_s.pth inside the weights folder, then run this again."
        )

    print(f"Loading fused {args.name} from {ckpt_path} (focused on license plates)")
    model, exp = load_model(args.name, ckpt_path, args.conf, args.nms, args.device)
    preproc = ValTransform(legacy=False)
    cap = open_camera(args.cam)

    window = "License Plate Reader"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    writer = None
    if args.save:
        out_dir = ROOT / "YOLOX_outputs" / "camera"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / time.strftime("camera_%Y%m%d_%H%M%S.mp4")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        print(f"Saving video to {out_path}")

    print(
        "Camera is open. Point it at a license plate.\n"
        "Press Y to confirm a reading, N to reject a wrong reading, or Q / Esc to quit."
    )
    watcher = PlateWatcher()
    frame_idx = 0
    last_vehicles = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera stopped sending frames.")
                break

            started = time.perf_counter()
            run_vehicle_pass = frame_idx % 8 == 0
            result, candidates, vehicles = analyze(
                model, exp, frame, preproc, args.device, run_vehicle_pass=run_vehicle_pass
            )
            if run_vehicle_pass:
                last_vehicles = vehicles
            watcher.submit(frame, last_vehicles)
            elapsed = time.perf_counter() - started
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                result,
                f"{fps:.1f} FPS",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            reading, enhanced_crop, feedback_banner = watcher.snapshot()
            paint_plate(result, reading, enhanced_crop, candidates, feedback_banner)
            if writer is not None:
                writer.write(result)
            cv2.imshow(window, result)
            frame_idx += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("y"), ord("Y")):
                watcher.apply_human_feedback(is_correct=True)
            elif key in (ord("n"), ord("N")):
                watcher.apply_human_feedback(is_correct=False)
    finally:
        watcher.stop()
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
