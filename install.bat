@echo off
REM PSX AI Trader - Phase 1 installer (Windows)
REM Creates a virtual environment, installs dependencies, and initializes
REM the project directory structure.

setlocal

echo ==============================================
echo PSX AI TRADER - INSTALL (Phase 1)
echo ==============================================

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: python was not found on PATH. Install Python 3.10+ first.
    pause
    exit /b 1
)

echo.
echo [1/5] Creating virtual environment (.venv) ...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: failed to create virtual environment.
    pause
    exit /b 1
)

echo.
echo [2/5] Activating virtual environment ...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: failed to activate virtual environment.
    pause
    exit /b 1
)

echo.
echo [3/5] Installing dependencies from requirements.txt ...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed.
    pause
    exit /b 1
)

echo.
echo [4/5] Initializing project directories and .env ...
python -m src.main init
if errorlevel 1 (
    echo WARNING: init reported problems - see output above.
)

echo.
echo [5/5] Initializing database ...
python -m src.main init-db
if errorlevel 1 (
    echo WARNING: init-db reported problems - see output above.
)

echo.
echo ==============================================
echo INSTALL COMPLETE
echo Next: python -m src.main health
echo Then: pytest -q
echo ==============================================
pause

endlocal
