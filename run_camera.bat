@echo off
rem Open the webcam with object detection and license-plate reading.
rem Uses the virtual environment inside this folder only.
setlocal
cd /d "%~dp0"

set "PYTHONUNBUFFERED=1"
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH=%~dp0"
set "TORCH_HOME=%~dp0.cache\torch"
set "XDG_CACHE_HOME=%~dp0.cache"
set "MPLCONFIGDIR=%~dp0.cache\matplotlib"

if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Local environment is missing. Create it inside this folder first:
    echo   py -3.12 -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    echo   .venv\Scripts\python.exe -m pip install -r requirements-camera.txt
    exit /b 1
)

"%~dp0.venv\Scripts\python.exe" "%~dp0camera.py" %*
