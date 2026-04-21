@echo off
REM PodLab Thumbnail Extractor - config.bat
REM Starts the server (via run.bat) and opens the browser to /#config after a short delay.

pushd "%~dp0"
if "%THUMBNAIL_EXTRACTOR_PORT%"=="" set THUMBNAIL_EXTRACTOR_PORT=5000
REM Open browser to /#config after a short delay so the server can come up.
start "" /B cmd /C "timeout /T 3 /NOBREAK >nul && start http://localhost:%THUMBNAIL_EXTRACTOR_PORT%/#config"
call "%~dp0run.bat"
popd
