@echo off
REM PodLab Thumbnail Extractor - kill.bat
REM Kills whatever is listening on THUMBNAIL_EXTRACTOR_PORT (default 5000).
REM Clean exit when nothing is running.

setlocal
if "%THUMBNAIL_EXTRACTOR_PORT%"=="" set THUMBNAIL_EXTRACTOR_PORT=5000

for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":%THUMBNAIL_EXTRACTOR_PORT% .*LISTENING"') do (
    taskkill /PID %%p /F >nul 2>&1
)

echo Killed anything on port %THUMBNAIL_EXTRACTOR_PORT%.
endlocal
