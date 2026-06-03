@echo off
title Python GBC Emulator
cd /d "%~dp0"

echo -----------------------------------------
echo   Python Game Boy / Game Boy Color Emulator
echo -----------------------------------------
echo.

:: Check if Python is installed
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.9+ from https://python.org
    echo          and make sure "Add Python to PATH" is checked during install.
    pause
    exit /b 1
)

:: Install dependencies if needed
echo Checking dependencies...
python -c "import pygame; import numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing pygame and numpy...
    python -m pip install -r requirements.txt --quiet
    if %errorlevel% neq 0 (
        echo [WARN]  Could not auto-install dependencies.
        echo          Run:  pip install pygame numpy
    )
)

echo.
echo Starting emulator...
echo.
echo   Controls:
echo     Arrow keys - D-pad
echo     Z          - A button
echo     X          - B button
echo     Right Shift- Select
echo     Enter      - Start
echo     Escape     - Return to menu / Quit
echo.
echo   In menu:  Arrow keys to navigate, Enter to select, Esc to go back
echo.

python "%~dp0gbc_emulator_skeleton.py"
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Emulator crashed. See the error message above.
    pause
)
