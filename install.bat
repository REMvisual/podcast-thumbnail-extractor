@echo off
setlocal enabledelayedexpansion
REM ==============================================================================
REM PodLab ^| Thumbnail Extractor - Windows installer
REM Creates venv, installs dependencies, downloads starter models.
REM Re-runnable; skips completed steps.
REM ==============================================================================

pushd "%~dp0"

echo.
echo =======================================================
echo  PodLab ^| Thumbnail Extractor - Installer
echo =======================================================
echo.

REM --- Python check ---
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10 or later from https://python.org
    popd
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Python: %PYVER%

REM --- venv ---
if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: venv creation failed.
        popd
        exit /b 1
    )
) else (
    echo venv exists, skipping.
)

REM --- Dependencies ---
echo Installing dependencies...
call "venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: dependency install failed.
    popd
    exit /b 1
)

REM --- Starter models ---
echo.
echo Downloading starter models (one-time, ~885 MB)...
python tools\download_models.py
if errorlevel 1 (
    echo WARNING: model download failed. You can still use face detection and heuristic mode.
    echo Re-run install.bat to retry, or use config UI to train your own models.
)

echo.
echo =======================================================
echo  Install complete.
echo  Start: run.bat          Configure: config.bat
echo =======================================================
popd
exit /b 0
