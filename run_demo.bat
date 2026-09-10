@echo off
title AI ANPR Trajectory & Adaptive Traffic Control
cd /d "%~dp0"
echo =========================================================================
echo  City-Wide AI Engine: Multi-Camera ANPR Trajectory & Adaptive Traffic Control
echo  SIH 2026 Working Prototype
echo =========================================================================
echo.

REM 1. Check if configured SIH virtual environment exists
if exist "C:\sih2026-1\phase1\.venv\Scripts\python.exe" (
    echo Using environment: C:\sih2026-1\phase1\.venv
    set "PYTHON_EXE=C:\sih2026-1\phase1\.venv\Scripts\python.exe"
    goto :RUN
)

REM 2. Check local virtual environment
if exist ".venv\Scripts\python.exe" (
    echo Using local environment: .venv
    set "PYTHON_EXE=.venv\Scripts\python.exe"
    goto :RUN
)

REM 3. Fallback to system Python
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo Using system Python
    set "PYTHON_EXE=python"
    goto :RUN
)

REM 4. Fallback to py launcher
where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo Using Python launcher (py)
    set "PYTHON_EXE=py"
    goto :RUN
)

echo ERROR: No Python installation detected on this system.
echo Please install Python 3.10+ and run: pip install -r requirements.txt
pause
exit /b 1

:RUN
echo Starting Streamlit Interactive Dashboard at http://localhost:8501 ...
"%PYTHON_EXE%" -m streamlit run app.py --server.port 8501
pause
