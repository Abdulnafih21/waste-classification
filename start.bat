@echo off
cd /d "%~dp0"
echo ========================================================
echo        Waste Classification AI - Startup Script
echo ========================================================
echo.

:: Check if Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in PATH!
    echo Please install Python 3.8+ and make sure to check "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

:: Install required packages if missing
echo Checking dependencies...
python -m pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [WARNING] Dependency installation failed or had warnings. Attempting to start app anyway...
)

echo.
echo Starting Waste Classification Web App...
echo (It will open in your default browser at http://127.0.0.1:5000)
echo.

python app.py
pause
