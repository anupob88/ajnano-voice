@echo off
REM ajnano-voice setup for Windows (.121 MooYor)
echo === ajnano-voice Setup ===

REM Check Python
python --version >nul 2>&1 || (echo ERROR: Python not found && exit /b 1)

REM Check CUDA
nvidia-smi >nul 2>&1 || (echo WARNING: nvidia-smi not found — GPU may not be available)

REM Create venv if needed
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate + install
call venv\Scripts\activate.bat
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo === Setup complete ===
echo Run: venv\Scripts\activate.bat ^&^& python server.py --port 8808
