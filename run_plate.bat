@echo off
rem Read plate numbers and issuing locations from photos in this folder.
rem Uses the virtual environment inside this folder only.
setlocal
cd /d "%~dp0"

set "PYTHONNOUSERSITE=1"
set "PYTHONPATH=%~dp0"
set "TORCH_HOME=%~dp0.cache\torch"
set "XDG_CACHE_HOME=%~dp0.cache"

if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Local environment is missing. Run this project from the folder that contains .venv.
    exit /b 1
)

"%~dp0.venv\Scripts\python.exe" "%~dp0read_plate.py" %*
