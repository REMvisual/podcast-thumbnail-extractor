@echo off
REM PodLab Thumbnail Extractor - run.bat
REM Portable launcher. Anchors to this script's directory via %~dp0.
REM Port override: set THUMBNAIL_EXTRACTOR_PORT before invocation.

pushd "%~dp0"
if "%THUMBNAIL_EXTRACTOR_PORT%"=="" set THUMBNAIL_EXTRACTOR_PORT=5000
title PodLab Thumbnail Extractor (port %THUMBNAIL_EXTRACTOR_PORT%)
call venv\Scripts\activate.bat
python src\app.py
popd
