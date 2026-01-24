@echo off
REM Script to start the Python ASL recognition API server on Windows

echo Starting ASL Recognition API Server...
echo ======================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "api_server.py" (
    echo Error: api_server.py not found. Please run this script from the 'asl' directory
    pause
    exit /b 1
)

REM Check if model files exist
if not exist "asl_model.keras" if not exist "asl_model.h5" (
    echo Warning: Model file not found. Please train the model first:
    echo   python train_model.py
    echo.
    set /p continue="Continue anyway? (y/n) "
    if /i not "%continue%"=="y" (
        exit /b 1
    )
)

REM Check if class_names.pkl exists
if not exist "class_names.pkl" (
    echo Error: class_names.pkl not found. Please train the model first:
    echo   python train_model.py
    pause
    exit /b 1
)

REM Install dependencies if needed
echo Checking dependencies...
python -c "import flask" 2>nul
if errorlevel 1 (
    echo Installing Flask and dependencies...
    pip install flask flask-cors
)

REM Start the server
echo.
echo Starting server on http://127.0.0.1:5000
echo Press Ctrl+C to stop
echo.

python api_server.py

pause
