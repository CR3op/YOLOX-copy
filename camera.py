#!/usr/bin/env python3
"""Live webcam detection with the local YOLOX checkout.

When a license plate is visible, the window shows its number and the
state or county printed on it. Plate reading runs beside the video so
the picture keeps moving.

Everything this script reads or writes stays inside this project folder.
"""

import argparse
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets.coco_classes import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils.boxes import postprocess
from yolox.utils.visualize import vis


def parse_args():
    """Read command-line options for the camera window."""
    parser = argparse.ArgumentParser(
        description="Run YOLOX on the webcam. Quit with q or Esc."
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
    """Turn a checkpoint path into an absolute path inside this folder.

    Paths outside the project are rejected so weights and outputs stay local.
    """
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    resolved = resolved.resolve()
    if ROOT != resolved and ROOT not in resolved.parents:
        raise SystemExit(f"Refusing to use a path outside this folder: {resolved}")
    return resolved


def open_camera(cam_id):
    """Open a webcam and return it after the first real frame arrives.

    On Windows, DirectShow is tried first, then Media Foundation. A backend
    can report that it opened and still deliver no picture, so each one is
    checked with an actual read.
    """
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
    """Build a YOLOX detector and load the pretrained checkpoint.

    Returns the model and its experiment config. The checkpoint is read on
    CPU first so a missing GPU does not fail the load.
    """
    exp = get_exp(None, name)
    exp.test_conf = conf
    exp.nmsthre = nms
    model = exp.get_model()
    model.eval()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)

    if device == "gpu":
        if not torch.cuda.is_available():
            raise SystemExit("No CUDA GPU is available. Run with --device cpu.")
        model.cuda()
    return model, exp


# COCO classes that can carry a license plate. The plate reader checks these crops
# when the plate does not fill the whole camera frame.
VEHICLE_NAMES = {"car", "truck", "bus", "motorcycle"}


def analyze(model, exp, frame, preproc, device):
    """Run YOLOX on one frame.

    Returns the frame with detection boxes drawn, and a list of vehicle
    boxes as (x0, y0, x1, y1). The original frame is left unchanged.
    """
    ratio = min(exp.test_size[0] / frame.shape[0], exp.test_size[1] / frame.shape[1])
    tensor, _ = preproc(frame, None, exp.test_size)
    tensor = torch.from_numpy(tensor).unsqueeze(0).float()
    if device == "gpu":
        tensor = tensor.cuda()

    with torch.no_grad():
        outputs = model(tensor)
        outputs = postprocess(
            outputs, exp.num_classes, exp.test_conf, exp.nmsthre, class_agnostic=True
        )

    if outputs[0] is None:
        return frame, []

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
    drawn = vis(frame.copy(), boxes, scores, classes, exp.test_conf, COCO_CLASSES)
    return drawn, vehicles


class PlateWatcher:
    """Read plates on a side thread so the camera window stays live.

    OCR is much slower than drawing a frame. This class keeps only the newest
    unread frame, reads it in the background, and remembers the last plate for
    a few seconds so the label does not flicker off between reads.
    """

    def __init__(self):
        """Start the background reader."""
        from read_plate import read_plate_in_frame

        self._read = read_plate_in_frame
        self._lock = threading.Lock()
        self._pending = None
        self._stop = False
        self.reading = None
        self.updated = 0.0
        self._announced = None
        self._thread = threading.Thread(target=self._loop, name="plate-reader", daemon=True)
        self._thread.start()

    def submit(self, frame, vehicles):
        """Queue one frame if the reader is idle. Older queued frames are skipped."""
        with self._lock:
            if self._pending is None:
                self._pending = (frame.copy(), list(vehicles))

    def snapshot(self):
        """Return the latest plate reading, or None once it is more than 8 seconds old."""
        with self._lock:
            if self.reading is None:
                return None
            if time.perf_counter() - self.updated > 8:
                return None
            return self.reading

    def stop(self):
        """Ask the reader thread to exit and wait briefly for it."""
        self._stop = True
        self._thread.join(timeout=1.0)

    def _loop(self):
        """Pull queued frames and print each new plate number once."""
        while not self._stop:
            job = None
            with self._lock:
                if self._pending is not None:
                    job = self._pending
                    self._pending = None
            if job is None:
                time.sleep(0.05)
                continue
            frame, vehicles = job
            try:
                reading = self._read(frame, vehicles)
            except Exception as exc:
                print(f"Plate reader error: {exc}")
                continue
            if not reading or not reading.get("normalized"):
                continue
            with self._lock:
                self.reading = reading
                self.updated = time.perf_counter()
            if reading["normalized"] != self._announced:
                self._announced = reading["normalized"]
                from read_plate import location_text

                place = location_text(reading) or "location not printed clearly"
                print(f"Plate {reading['number']} | {place}")


def paint_plate(frame, reading):
    """Draw the plate number, issuing location, and a box around the number."""
    from read_plate import location_text

    if reading is None:
        _label(frame, "Looking for a plate...", (10, 70))
        return frame
    items = reading.get("number_items") or []
    if items:
        x0 = int(min(item["x"] for item in items))
        y0 = int(min(item["y"] for item in items))
        x1 = int(max(item["x"] + item["w"] for item in items))
        y1 = int(max(item["y"] + item["h"] for item in items))
        cv2.rectangle(frame, (x0, y0), (x1, y1), (80, 220, 80), 3)
    place = location_text(reading)
    text = reading["number"] if not place else f"{reading['number']}  |  {place}"
    _label(frame, text, (10, 70))
    return frame


def _label(frame, text, origin):
    """Draw one line of text on a black background so it stays readable."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
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
    cv2.putText(frame, text, (x, y), font, scale, (80, 220, 80), thickness)


def main():
    """Open the webcam, detect objects, and overlay any plate that is read."""
    args = parse_args()
    ckpt_path = resolve_inside_project(args.ckpt)
    if not ckpt_path.is_file():
        raise SystemExit(
            f"Missing checkpoint: {ckpt_path}\n"
            "Download yolox_s.pth into the weights folder, then run this again."
        )

    print(f"Loading {args.name} from {ckpt_path}")
    model, exp = load_model(args.name, ckpt_path, args.conf, args.nms, args.device)
    preproc = ValTransform(legacy=False)
    cap = open_camera(args.cam)

    window = "YOLOX camera"
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

    print("Camera is open. Point it at a license plate. Press q or Esc to quit.")
    watcher = PlateWatcher()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera stopped sending frames.")
                break

            started = time.perf_counter()
            result, vehicles = analyze(model, exp, frame, preproc, args.device)
            watcher.submit(frame, vehicles)
            elapsed = time.perf_counter() - started
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                result,
                f"{fps:.1f} FPS",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            paint_plate(result, watcher.snapshot())
            if writer is not None:
                writer.write(result)
            cv2.imshow(window, result)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        watcher.stop()
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
