@echo off
REM Start the KOI Classification API Server
REM Checks if model is trained before starting

echo ======================================================================
echo KOI Classification API Server
echo ======================================================================
echo.

REM Check if model exists
if not exist "backend\saved_models\stacking_model.pkl" (
    echo ERROR: Model not found!
    echo.
    echo Please train the model first by running:
    echo   train_model.bat
    echo.
    echo Or manually:
    echo   python train_3class_model.py
    echo.
    pause
    exit /b 1
)

echo Model found: backend\saved_models\stacking_model.pkl
echo.
echo Starting server...
echo.
echo The API will be available at:
echo   - http://localhost:8000
echo   - http://localhost:8000/docs (Interactive API docs)
echo.
echo Press Ctrl+C to stop the server
echo.
echo ======================================================================
echo.

cd backend
python app.py

REM If server stops
echo.
echo Server stopped.
pause
