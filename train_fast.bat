@echo off
REM FAST GPU Training for GTX 1050 Ti
REM ==================================
REM Optimized for speed: 3-8 minutes training time

echo.
echo ========================================================================
echo  FAST 3-Class KOI Model Training - GTX 1050 Ti GPU
echo ========================================================================
echo.
echo  This will train your model using:
echo  - NVIDIA GTX 1050 Ti GPU acceleration
echo  - 150 estimators per model (FAST mode)
echo  - Real-time progress tracking
echo  - Expected time: 3-8 minutes
echo.
echo  TIP: Open another terminal and run 'nvidia-smi -l 1' to monitor GPU!
echo.
echo ========================================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Starting FAST GPU training...
echo.
pause

REM Run fast training
python train_3class_gpu_fast.py

if errorlevel 1 (
    echo.
    echo ========================================================================
    echo  TRAINING FAILED!
    echo ========================================================================
    echo.
    echo Troubleshooting:
    echo 1. Check GPU: nvidia-smi
    echo 2. Check setup: python check_gpu_ready.py
    echo 3. Try CPU training: python train_3class_cpu.py
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo  SUCCESS! Model trained and saved!
echo ========================================================================
echo.
echo Next steps:
echo 1. Test model:    python test_3class_api.py
echo 2. Start server:  cd backend ^&^& python app.py
echo 3. View API:      http://localhost:8000/docs
echo.
pause
