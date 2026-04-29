@echo off
REM Train 3-Class KOI Model with GPU (GTX 1050 Ti)
REM =============================================

echo.
echo ========================================================================
echo  Training 3-Class KOI Model with GPU Acceleration (GTX 1050 Ti)
echo ========================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo Starting GPU training...
echo.
echo NOTE: This will use your NVIDIA GTX 1050 Ti GPU if available.
echo       If GPU is not available, it will fall back to CPU training.
echo.
echo Press Ctrl+C to cancel, or
pause

REM Run the GPU training script
python train_3class_gpu.py

if errorlevel 1 (
    echo.
    echo ========================================================================
    echo  TRAINING FAILED!
    echo ========================================================================
    echo.
    echo Troubleshooting tips:
    echo 1. Make sure cumulative_2025.10.03_00.50.03.csv exists
    echo 2. Check GPU drivers are installed: nvidia-smi
    echo 3. Verify CUDA toolkit is installed
    echo 4. Try CPU training instead: python train_3class_cpu.py
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo  TRAINING COMPLETED SUCCESSFULLY!
echo ========================================================================
echo.
echo Your model is saved in: backend\saved_models\
echo.
echo Next steps:
echo 1. Test the model: python test_3class_api.py
echo 2. Start server: cd backend ^&^& python app.py
echo.
pause
