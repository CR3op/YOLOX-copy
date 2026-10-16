# Using this project

This folder is a local copy of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) plus a webcam window and a license-plate reader. Everything it installs or writes stays inside this folder: the virtual environment, model weights, caches, and output images.

The pipeline is focused strictly on license plates. It locates rectangular plate regions with fast OpenCV morphology (and fused YOLOX vehicle-crop fallback), applies image enhancement (mud/stain suppression, LAB CLAHE contrast balancing, bilateral denoising, gamma midtone darkening, and unsharp masking), and reads all printed plate contents with the local Windows OCR engine in `read_plate.py`. Nothing is downloaded from outside this folder.

## Run the camera

Plug in a webcam and allow desktop apps to use it under Windows privacy settings. From this folder:

```bat
.\run_camera.bat
```

A window titled **YOLOX camera** opens. It draws boxes around detected objects. When a plate is readable, the window shows the plate number and the issuing place, for example `291-764M  |  Starr County, Texas`. Hold the plate fairly close and steady. Press **q** or **Esc** to close the window and release the camera.

If the camera is not the first device:

```bat
.\run_camera.bat --cam 1
```

To also save the annotated video under `YOLOX_outputs/camera`:

```bat
.\run_camera.bat --save
```

## Read a plate photo

Put photos in `datasets\plates`. Photos already placed directly in `datasets` are read too. Then run:

```bat
.\run_plate.bat
```

Or pass one image inside this folder:

```bat
.\run_plate.bat "datasets\license plate.jpg"
```

The command prints the plate number and issuing place, and writes a marked copy to `YOLOX_outputs\plates`.

## What the scripts do

| File | Role |
| --- | --- |
| `run_camera.bat` | Starts the webcam with the local virtual environment. |
| `camera.py` | Shows the camera, runs YOLOX, and overlays plate reads. |
| `run_plate.bat` | Reads plate photos with the local virtual environment. |
| `read_plate.py` | Finds the plate number, state, and county sticker text. |
| `tools/demo.py` | Upstream YOLOX demo, adjusted for Windows webcams and current PyTorch. |

`camera.py` keeps the video moving by reading plates on a background thread. A reading stays on screen for about 8 seconds, then clears if the plate is gone.

## Folder layout

| Path | Contents |
| --- | --- |
| `.venv` | Python packages for this project only. |
| `weights\yolox_s.pth` | Pretrained YOLOX-S checkpoint. |
| `datasets\plates` | Plate photos to read. |
| `YOLOX_outputs` | Marked photos and optional camera recordings. |
| `.cache` | Torch and pip caches kept inside the project. |

## Recreate the environment

From this folder, with Python 3.12:

```bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements-camera.txt
.venv\Scripts\python.exe -m pip install -r requirements-plate.txt
```

Download the detector weights into `weights\yolox_s.pth` from the YOLOX-S link in the benchmark table of `README.md`.

Plate reading uses the Windows English OCR language. If it is missing, install it from an administrator PowerShell:

```powershell
Add-WindowsCapability -Online -Name "Language.OCR~~~en-US~0.0.1.0"
```

## Upstream YOLOX

Training, evaluation, and export instructions for the original project are in `README.md` below the "Using this copy" section.
