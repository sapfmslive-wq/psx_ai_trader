@echo off
REM PSX AI Trader - Live Collector (Phase 9)
REM Runs the continuous collector loop until Ctrl+C is pressed.

setlocal

if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: virtual environment not found. Run install.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo ==============================================
echo PSX AI TRADER - LIVE COLLECTOR
echo Press Ctrl+C to stop gracefully.
echo ==============================================
echo.

python -m src.main run-collector %*

echo.
echo Collector stopped.
pause

endlocal
