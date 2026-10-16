# Using This Project & Architecture Documentation

This folder is a self-contained computer vision application built on top of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), OpenCV, and the native Windows OCR engine (`winocr`). It is specialized for **license plate detection, multi-stage image enhancement, full plate content recognition, confidence scoring, and human-in-the-loop (HITL) learning** from both still photos and a live webcam feed.

Everything this project installs, caches, reads, or writes stays strictly inside this project directory (`d:\programming\vecspace computer vision model`):
- **Virtual environment**: `.venv` (`PYTHONNOUSERSITE=1`)
- **Model weights**: `weights\yolox_s.pth`
- **Local caches**: `.cache` (`TORCH_HOME` and `XDG_CACHE_HOME`)
- **Input photos & learned feedback memory**: `datasets\plates`, `datasets`, and `datasets\plate_feedback.json`
- **Outputs**: `YOLOX_outputs\plates` and `YOLOX_outputs\camera`

---

## Project Summary

The system solves three core challenges in real-world license plate recognition:
1. **Speed & Focus**: Standard object detectors spend heavy CPU time on every frame detecting 80 everyday object classes (people, chairs, cups, etc.) and still do not read text. This project narrows the pipeline strictly to **license plates**, combining a ~1.6–2.1 ms OpenCV morphological plate-region detector with a fused, periodic YOLOX vehicle fallback and an asynchronous background OCR worker (`PlateWatcher`).
2. **Reading Muddy, Low-Contrast, or Small Plate Text**: Real license plates often have mud splatters, glare, shadows, or small secondary print (plate classification, state slogans, county stickers, expiration dates, and sticker serial numbers). The image enhancement pipeline in `read_plate.py` cleans and enhances both the full plate and targeted sub-regions before passing them to a persistent local WinRT OCR engine.
3. **Confidence Scoring & Human-in-the-Loop Learning**: Rather than guessing blindly, the algorithm calculates a **0–100% accuracy confidence score** broken down across five measurable factors, and allows a human operator to confirm (`--confirm` / `Y`), reject (`--reject` / `N`), or teach (`--teach`) plate readings so the model continuously improves its accuracy and repairs noisy frames.

---

## Recent Changes & Benefits

### 1. License-Plate-Only Focus
- **Removed Unrelated Object Clutter**: `camera.py` no longer calls `vis(...)` to draw bounding boxes for all 80 COCO classes. Only candidate license plate regions, recognized plate text boxes, and the enhanced plate inset are rendered on screen.
- **Fast Morphological Plate Localization (`detect_plate_regions`)**: Locates rectangular plate candidates directly in the frame by downscaling to 480px width, applying a horizontal `cv2.MORPH_BLACKHAT` transform, computing `cv2.Sobel` X-gradients (which respond strongly to dense vertical character strokes on a rectangular plate), closing gaps, and filtering contours by aspect ratio (`1.3` to `5.8`).
- **Vehicle-Only YOLOX Fallback (`detect_vehicles`)**: When YOLOX runs, it filters exclusively for vehicles that carry plates (`car`, `truck`, `bus`, `motorcycle`) so distant bumper areas can still be checked.

### 2. Multi-Stage Image Enhancement Pipeline (`read_plate.py`)
- **Chroma-Guided Mud & Stain Suppression (`enhance_plate_image`)**: Identifies warm brown/tan mud splatters (`R - B` warmth combined with mid-luminance masking) and lifts those pixels toward the reflective plate background without erasing neutral black character ink.
- **LAB Luminance Stretching & CLAHE (`_enhance_soft`)**: Stretches the 2nd–98th luminance percentiles in `LAB` color space and blends Contrast Limited Adaptive Histogram Equalization (`cv2.createCLAHE`) so dark, shadowed, or washed-out plates become high-contrast and legible.
- **Edge-Preserving Denoising & Unsharp Masking**: Applies `cv2.bilateralFilter` to smooth dirt grain while keeping character edges sharp, followed by Gaussian unsharp masking (`cv2.addWeighted`) to crisp up blurry strokes.
- **Midtone Gamma Darkening (`_enhance_dark_ink`)**: Applies non-linear gamma darkening (`gamma = 1.8`) on the sub-header band so thin or mud-obscured print—such as **`TRAILER`** beneath **`TEXAS`**—becomes readable.
- **Saved Enhanced Output**: Every run of `.\run_plate.bat` saves both the cleaned/enhanced plate image (`*_enhanced.jpg`) and the annotated result (`*.jpg`) in `YOLOX_outputs\plates`.

### 3. Complete Plate Content Extraction
Instead of extracting only the 7-character license number, `read_plate.py` extracts and structures **all printed contents** on the plate:
- **Plate Number**: e.g., `291-764M` (with automatic cleanup of state-silhouette separators and OCR punctuation noise)
- **Issuing State & Country**: e.g., `Texas, United States`
- **County (from registration sticker)**: e.g., `Starr County`
- **Plate Type / Classification**: e.g., `Trailer`, `Truck`, `Passenger`, `Commercial`
- **State Slogan**: e.g., `The Lone Star State`
- **Sticker Expiration Date**: e.g., `11/25`
- **Sticker Serial Numbers**: e.g., `31001425, 96764981, 690361K`
- **Full Contents Summary**: Ordered list of all recognized text elements on the plate.

---

## Confidence Scoring & Human-in-the-Loop Learning

### How the Confidence Score Works (`compute_confidence`)

Every plate reading is evaluated on a **0–100% scale** across five weighted signals:

| Factor | Max Points | How It Is Calculated |
| --- | --- | --- |
| **1. Multi-Pass OCR Agreement** | **35.0 pts** | `read_text()` runs 5 distinct crop/enhancement passes (`full_raw`, `band_raw`, `band_soft_clahe`, `sub_dark_ink`, `sticker_raw`). `dedupe()` tracks how many passes independently read the same plate number: 1 pass = `18 pts`, 2 passes = `28 pts`, 3+ passes = `35 pts`. |
| **2. Plate Format Validity** | **25.0 pts** | Verifies standard 6–7 character length (`10 pts`), mixed letters + digits (`6 pts`), and clean alphanumeric pattern formatting such as `###-###A` (`9 pts`) without OCR punctuation noise. |
| **3. Contextual Contents Match** | **20.0 pts** | Awards corroboration points when surrounding plate contents are also detected in the image: Issuing State (`+7 pts`), Plate Type (`+4 pts`), State Slogan (`+4 pts`), and County or Expiration Date (`+5 pts`). |
| **4. Image Clarity & Contrast** | **10.0 pts** | Measures pixel luminance standard deviation (`np.std(gray)`, up to `5 pts`) and Sobel edge sharpness (`cv2.Sobel`, up to `5 pts`) inside the plate number bounding box. |
| **5. Human Feedback Boost** | **+15.0 pts** | Starts at `0.0 pts` before human confirmation so unverified reads stay below `90%`. Once confirmed by a human, adds `+8.0` to `+15.0 pts` based on confirmation count and match similarity. |

**Confidence Levels**:
- **90.0% – 99.9%**: `Verified High` (human-confirmed) or `High`
- **75.0% – 89.9%**: `Good` (strong automated reading; e.g., `87.4%` on the muddy `291-764M` plate before human verification)
- **55.0% – 74.9%**: `Moderate (Needs Human Check)`
- **Below 55.0%**: `Low (Uncertain)`

### How Human Intervention Helps the Algorithm Learn (`record_human_feedback`)

When you confirm, reject, or teach a plate reading, the algorithm updates `datasets\plate_feedback.json` (kept strictly inside this project folder) and improves future reads in four ways:

1. **OCR Visual-Confusion & Alias Learning (`ocr_similarity`)**:
   - On a moving webcam or muddy plate, OCR often misreads characters (such as reading a center state silhouette as `0`/`+`/`4`, or splitting a trailing `M` into `/1` or `11`, producing `291064M` or `291+764/1`).
   - Using weighted visual-confusion edit distance (`OCR_CONFUSION_GROUPS`) and learned aliases stored on confirmation, `_apply_learned_memory()` automatically repairs noisy OCR reads (e.g., `291+764/1` -> `291-764M`).
2. **Backfilling Blurry Secondary Contents (`_apply_learned_memory`)**:
   - When a live camera frame is too angled or motion-blurred to read tiny sticker text, matching a human-verified plate automatically backfills its known `state`, `county`, `plate_type`, `slogan`, `expiration`, and `sticker_numbers`.
3. **Adaptive Enhancement Pass Weights (`pass_weights`)**:
   - Each time you confirm a reading, the enhancement passes (`band_soft_clahe`, `band_raw`, `full_raw`, etc.) that produced the winning plate number receive a priority weight boost (`+0.05`, up to `1.50x`), training the scorer to favor the most effective image-enhancement filters.
4. **Suppressing Rejected False Positives (`rejected_tokens`)**:
   - When you reject an inaccurate read, that token is recorded in `rejected_tokens` (and verified sticker serials are also excluded) so `_is_non_number_text()` blocks it from being chosen as a plate number again.

---

## Performance Improvements: How FPS Jumped from ~20 FPS to ~600 FPS

Previously, the live camera loop ran at roughly **11–20 FPS** (`~50–90 ms` per frame). After the optimizations, normal camera frames process in **~1.6 ms (~500–600 FPS)**, with an overall average above **100+ FPS** even including periodic deep-learning passes.

| Bottleneck (Before) | Optimization Applied (After) | Impact on Speed |
| --- | --- | --- |
| **Full 640x640 YOLOX on every single frame** (`~89 ms/frame`) | Replaced per-frame full-scene neural inference with `detect_plate_regions()` (OpenCV morphology at 480px width), and scheduled YOLOX vehicle fallback only once every 8th frame (`frame_idx % 8 == 0`). | Non-YOLOX frames drop from `~89 ms` to **`~1.6 ms` (~600 FPS)**. |
| **Unfused Conv2d + BatchNorm2d layers at 640x640** | Applied `fuse_model(model)` at startup, switched to `torch.inference_mode()`, and lowered vehicle fallback input size to `416x416`. | YOLOX pass itself is **2.2x faster** (`~89 ms` -> `~39 ms`). |
| **Drawing all 80 COCO classes (`vis`)** | Removed 80-class box/mask rendering and filtered strictly to license plates. | Eliminates per-frame overlay overhead and visual clutter. |
| **Re-creating WinRT `OcrEngine` & `picklify()` on every crop** (`~192 ms/plate`) | Cached a single persistent `OcrEngine("en")` and `asyncio` event loop in `_recognize_bgr()`, reading `word.bounding_rect` directly in C++/WinRT without recursive Python `dir()` reflection. | Per-crop OCR drops to **`~22 ms`** (`~110 ms` total for 5 enhanced passes). |
| **Synchronous OCR blocking** | `PlateWatcher` runs all multi-pass image enhancement and OCR on a dedicated daemon thread (`plate-reader`), keeping only the newest frame. | Display loop never waits on OCR (`0 ms` main-thread stall). |

---

## How to Run

### 1. Read a License Plate Photo

Put photos in `datasets\plates` (or directly in `datasets\`). Then run:

```bat
.\run_plate.bat
```

Or pass a specific image inside this folder:

```bat
.\run_plate.bat "datasets\license plate.jpg"
```

Example output (before human confirmation):
```text
datasets\license plate.jpg
  Plate number: 291-764M
  Confidence score: 87.4% (Good)
    - OCR pass agreement: 35.0/35.0 pts (3 passes: band_raw, band_soft_clahe, full_raw)
    - Plate format validity: 25.0/25.0 pts
    - Contextual contents match: 20.0/20.0 pts
    - Image clarity & contrast: 7.4/10.0 pts
    - Human feedback boost: +0.0/15.0 pts (confirmations: 0)
  Issued in: Starr County, Texas, United States
  Plate type: Trailer
  Slogan: The Lone Star State
  Sticker expiration: 11/25
  Sticker numbers: 31001425, 96764981, 690361K
  Recognized contents: TEXAS | TRAILER | 291-764M | The Lone Star State | Starr County | EXP 11/25 | 31001425 | 96764981 | 690361K
  Enhanced photo: YOLOX_outputs\plates\license plate_enhanced.jpg
  Marked photo: YOLOX_outputs\plates\license plate.jpg
```

### 2. Provide Human Feedback (CLI or Live Camera)

All human feedback is saved locally inside `datasets\plate_feedback.json`:
- **Confirm a reading is accurate** (boosts confidence to `95.4% Verified High` and learns plate contents + pass weights):
  ```bat
  .\run_plate.bat --confirm
  ```
- **Reject a wrong reading** (adds the misread token to `rejected_tokens` and re-evaluates):
  ```bat
  .\run_plate.bat --reject
  ```
- **Teach the true plate number** (saves the true plate number and maps noisy OCR tokens in the photo as aliases):
  ```bat
  .\run_plate.bat --teach "291-764M"
  ```
- **In the live camera window (`.\run_camera.bat`)**:
  - Press **`Y`** to confirm the displayed reading on screen.
  - Press **`N`** to reject a false reading.

### 3. Run the Live Webcam Reader

Plug in a webcam and run:

```bat
.\run_camera.bat
```

- A window titled **License Plate Reader** opens.
- When a plate is detected and read, the window highlights all recognized text regions in cyan, outlines the main plate number in green, displays the parsed plate summary, confidence score, and slogan at the top-left, and shows an **Enhanced Plate** preview in the top-right corner.
- Press **`Y`** to confirm a correct read, **`N`** to reject a wrong read, or **`Q`** / **`Esc`** to quit.
- For a second camera, run `.\run_camera.bat --cam 1`. To record annotated video under `YOLOX_outputs\camera`, add `--save`.

---

## Key Files

| File | Role |
| --- | --- |
| `run_plate.bat` | Runs `read_plate.py` inside the local `.venv` with isolated cache paths. |
| `read_plate.py` | Plate region localization, mud/contrast/sharpness image enhancement, cached WinRT OCR, 5-factor confidence scoring, and human-feedback learning. |
| `run_camera.bat` | Launches `camera.py` inside the local `.venv`. |
| `camera.py` | High-FPS webcam loop focused on license plates, background `PlateWatcher` thread, live confidence & enhanced-plate overlay, and `Y`/`N` human feedback keys. |
| `datasets\plate_feedback.json` | Local human-in-the-loop learning memory (verified plates, OCR aliases, rejected tokens, and enhancement pass weights). |
| `weights\yolox_s.pth` | Local YOLOX-S weights used for vehicle-crop fallback. |
| `YOLOX_outputs\plates` | Saved enhanced plate images (`*_enhanced.jpg`) and annotated plate detections (`*.jpg`). |
