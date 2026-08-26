@echo off
setlocal
cd /d "%~dp0"
title VOD BLADE

set VENV_PYTHON=%~dp0.venv\Scripts\python.exe

if not exist "%VENV_PYTHON%" (
    echo [VOD BLADE] Virtual environment not found at "%VENV_PYTHON%".
    echo Run setup first, from this folder:
    echo     python -m venv .venv
    echo     .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo [VOD BLADE] Starting app.py ...
echo [VOD BLADE] Once it's up, open http://localhost:7863 in your browser.
echo.
"%VENV_PYTHON%" app.py

echo.
echo [VOD BLADE] Server stopped (exit code %ERRORLEVEL%).
pause
