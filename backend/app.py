"""
FastAPI Backend for KOI Classification
Production-ready API with model caching and comprehensive error handling
"""
from fastapi import FastAPI, HTTPException, status, Request, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import time
import pandas as pd
import io
import joblib
import os

from models import (
    KOIFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse,
    CSVPredictionResponse,
    ModelStatistics,
    HyperparameterConfig,
    TrainingRequest,
    TrainingResponse,
    TrainingStatusResponse,
    DataIngestionRequest,
    DataIngestionResponse,
    PlanetData,
    PlanetDataResponse,
    PlanetListResponse,
    FlexiblePlanetData,
    FlexiblePlanetResponse,
    FlexiblePlanetListResponse,
    CSVUploadResponse,
    SimplePlanetInput,
    SimplePlanetPredictionResponse
)
from ml_model import KOIModelPredictor
from config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables for tracking
training_jobs = {}  # Store training job status
training_data_path = Path("data/training_data.csv")  # Path to accumulated training data
planet_storage_path = Path("data/planets")  # Directory for individual planet data
planet_storage_path.mkdir(parents=True, exist_ok=True)  # Create directory if not exists
flexible_planet_storage_path = Path("data/flexible_planets")  # Directory for flexible planet data
flexible_planet_storage_path.mkdir(parents=True, exist_ok=True)  # Create directory

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (loaded once at startup for performance)
model_predictor: KOIModelPredictor = None

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.on_event("startup")
async def startup_event():
    """Load model on startup for better performance"""
    global model_predictor
    try:
        logger.info("🚀 Starting KOI Classification API...")
        
        model_path = Path(settings.MODEL_PATH)
        metadata_path = Path(settings.METADATA_PATH)
        
        logger.info(f"📂 Model path: {model_path}")
        logger.info(f"📂 Metadata path: {metadata_path}")
        
        if not model_path.exists():
            logger.error(f"❌ Model file not found: {model_path}")
            logger.warning("⚠️ API will start but predictions will fail until model is loaded")
            return
        
        logger.info("🔄 Loading model...")
        model_predictor = KOIModelPredictor(
            model_path=str(model_path),
            metadata_path=str(metadata_path) if metadata_path.exists() else None
        )
        
        model_info = model_predictor.get_model_info()
        logger.info("✅ Model loaded successfully!")
        logger.info(f"   Model type: {model_info.get('model_type')}")
        logger.info(f"   Features: {model_info.get('feature_count')}")
        logger.info(f"   GPU accelerated: {model_info.get('gpu_accelerated', 'Unknown')}")
        logger.info(f"   Version: {model_info.get('model_version')}")
        logger.info("🎉 API is ready to accept requests!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.warning("⚠️ API will start but predictions will fail")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "KOI Classification API - Exoplanet Research Tool",
        "version": settings.API_VERSION,
        "status": "running",
        "description": "API for classifying Kepler Objects of Interest (KOI) into CONFIRMED, CANDIDATE, or FALSE POSITIVE and managing ML models",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "prediction": {
                "single": "/predict",
                "batch": "/predict/batch",
                "csv_upload": "/predict/csv"
            },
            "model": {
                "info": "/model/info",
                "statistics": "/model/statistics",
                "hyperparameters": "/model/hyperparameters"
            },
            "training": {
                "start": "/training/start",
                "status": "/training/status/{job_id}",
                "history": "/training/history"
            },
            "data": {
                "ingest": "/data/ingest",
                "info": "/data/info"
            },
            "planets": {
                "save": "/planets/save",
                "get": "/planets/{planet_id}",
                "list": "/planets/list",
                "delete": "/planets/{planet_id}"
            },
            "flexible_planets": {
                "upload_csv": "/flexible-planets/upload-csv",
                "save_single": "/flexible-planets/save",
                "get": "/flexible-planets/{planet_id}",
                "list": "/flexible-planets/list",
                "delete": "/flexible-planets/{planet_id}"
            }
        },
        "features": [
            "Single and batch predictions",
            "CSV file upload for bulk predictions",
            "Model statistics and accuracy metrics",
            "Hyperparameter viewing and configuration",
            "Model retraining with new data",
            "Training progress tracking",
            "Data ingestion and management",
            "Individual planet data storage and retrieval",
            "🆕 Flexible planet data with color, name, distance, etc.",
            "🆕 Smart CSV column detection and mapping",
            "🆕 Support for any planet property structure"
        ]
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    Returns model status and information
    """
    if not model_predictor or not model_predictor.is_loaded():
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_type=None,
            model_version=None
        )
    
    model_info = model_predictor.get_model_info()
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_type=model_info.get("model_type"),
        model_version=model_info.get("model_version"),
        feature_count=model_info.get("feature_count"),
        gpu_accelerated=model_info.get("gpu_accelerated")
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: KOIFeatures):
    """
    Predict KOI disposition for a single candidate
    
    - **features**: Dictionary of KOI feature values
    
    Returns:
    - **prediction**: Predicted class (0, 1, or 2)
    - **prediction_label**: Human-readable label (CONFIRMED, CANDIDATE, or FALSE POSITIVE)
    - **confidence**: Prediction confidence (0-1)
    - **probabilities**: Probability for each class
    """
    if not model_predictor or not model_predictor.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check /health endpoint."
        )
    
    try:
        # Make prediction
        prediction, label, confidence, probabilities = model_predictor.predict(request.features)
        
        # Get model version
        model_info = model_predictor.get_model_info()
        model_version = model_info.get("model_version", "unknown")
        
        return PredictionResponse(
            prediction=prediction,
            prediction_label=label,
            confidence=confidence,
            probabilities=probabilities,
            model_version=model_version
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict KOI disposition for multiple candidates
    
    - **features_list**: List of feature dictionaries
    
    Returns batch predictions with confidence scores
    """
    if not model_predictor or not model_predictor.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check /health endpoint."
        )
    
    try:
        # Make batch predictions
        results = model_predictor.predict_batch(request.features_list)
        
        model_info = model_predictor.get_model_info()
        model_version = model_info.get("model_version", "unknown")
        
        predictions = [
            PredictionResponse(
                prediction=pred,
                prediction_label=label,
                confidence=conf,
                probabilities=probs,
                model_version=model_version
            )
            for pred, label, conf, probs in results
        ]
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.post("/predict/csv", response_model=CSVPredictionResponse, tags=["Prediction"])
async def predict_csv(file: UploadFile = File(...), max_rows: int = 10000):
    """
    Upload a CSV file and get predictions for all rows
    
    **Performance Optimized**: Handles large files efficiently with row limits
    
    The CSV file should have the same structure as the training dataset.
    The target column (koi_disposition) will be automatically removed if present.
    Missing features will be filled with default values.
    
    - **file**: CSV file with KOI data
    - **max_rows**: Maximum rows to process (default: 10000, prevents timeouts on large files)
    
    Returns:
    - **predictions**: List of predictions for each row
    - **total_count**: Total number of rows processed
    - **success_count**: Number of successful predictions
    - **failed_count**: Number of failed predictions
    - **processing_time**: Time taken to process the file (seconds)
    - **errors**: List of errors (if any) with row numbers
    
    **Note**: For files with >10,000 rows, only the first 10,000 will be processed by default.
    Increase max_rows parameter if needed, but be aware of longer processing times.
    """
    if not model_predictor or not model_predictor.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check /health endpoint."
        )
    
    start_time = time.time()
    
    # Validate file type
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded. Please select a valid CSV file."
        )
    
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file format. Only CSV files are accepted. You uploaded: {file.filename}"
        )
    
    try:
        # Read CSV file with encoding detection
        contents = await file.read()
        
        # Try multiple encodings
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
        df = None
        encoding_used = None
        
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(io.BytesIO(contents), encoding=encoding)
                encoding_used = encoding
                logger.info(f"✅ Successfully decoded CSV with {encoding} encoding")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                # If it's not an encoding error, try next encoding
                if "codec" not in str(e).lower() and "decode" not in str(e).lower():
                    raise
                continue
        
        if df is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to decode CSV file. Please ensure it's a valid CSV file with UTF-8, Latin-1, or Windows-1252 encoding. The file may be corrupted or in an unsupported format."
            )
        
        logger.info(f"📄 Received CSV file: {file.filename} (encoding: {encoding_used})")
        logger.info(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
        
        if len(df) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CSV file is empty - no data rows found"
            )
        
        # Apply row limit for safety (prevent infinite loops and timeouts)
        MAX_ROWS = max_rows if max_rows > 0 else 10000
        original_row_count = len(df)
        if len(df) > MAX_ROWS:
            logger.warning(f"⚠️ CSV has {len(df)} rows. Limiting to first {MAX_ROWS} rows for performance.")
            df = df.head(MAX_ROWS)
        
        # Remove target column if present
        target_columns = ['koi_disposition', 'disposition', 'label', 'target']
        for col in target_columns:
            if col in df.columns:
                df = df.drop(columns=[col])
                logger.info(f"   Removed target column: {col}")
        
        # Convert DataFrame rows to list of dictionaries
        features_list = df.to_dict('records')
        
        # Make predictions with progress logging
        predictions = []
        errors = []
        model_info = model_predictor.get_model_info()
        model_version = model_info.get("model_version", "unknown")
        
        total_rows = len(features_list)
        batch_size = 100  # Log progress every 100 rows
        
        logger.info(f"🔄 Starting prediction for {total_rows} rows...")
        
        for idx, features in enumerate(features_list):
            try:
                # Log progress every batch_size rows
                if idx > 0 and idx % batch_size == 0:
                    progress_pct = (idx / total_rows) * 100
                    logger.info(f"📊 Progress: {idx}/{total_rows} ({progress_pct:.1f}%)")
                
                prediction, label, confidence, probabilities = model_predictor.predict(features)
                predictions.append(
                    PredictionResponse(
                        prediction=prediction,
                        prediction_label=label,
                        confidence=confidence,
                        probabilities=probabilities,
                        model_version=model_version
                    )
                )
            except Exception as e:
                errors.append({
                    "row": idx,
                    "error": str(e)
                })
                # Only log first 10 errors to avoid log spam
                if len(errors) <= 10:
                    logger.warning(f"Failed to predict row {idx}: {e}")
        
        # Calculate statistics
        total_count = len(features_list)
        success_count = len(predictions)
        failed_count = len(errors)
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Processed {total_count} rows in {processing_time:.2f}s")
        logger.info(f"   Success: {success_count}, Failed: {failed_count}")
        
        # Add warning if rows were limited
        if original_row_count > MAX_ROWS:
            errors.append({
                "row": "WARNING",
                "error": f"CSV contained {original_row_count} rows but only first {MAX_ROWS} were processed. Consider splitting large files."
            })
        
        return CSVPredictionResponse(
            predictions=predictions,
            total_count=total_count,
            success_count=success_count,
            failed_count=failed_count,
            processing_time=round(processing_time, 3),
            errors=errors
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV file is empty"
        )
    except pd.errors.ParserError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid CSV format"
        )
    except Exception as e:
        logger.error(f"Error processing CSV file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CSV processing failed: {str(e)}"
        )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """
    Get detailed information about the loaded model
    
    Returns model type, version, training info, and performance metrics
    """
    if not model_predictor or not model_predictor.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return model_predictor.get_model_info()


@app.get("/model/statistics", response_model=ModelStatistics, tags=["Model"])
async def get_model_statistics():
    """
    Get model performance statistics and accuracy metrics
    
    Returns:
    - Accuracy, precision, recall, F1 score
    - Confusion matrix
    - Class distribution
    - Cross-validation scores
    - Training information
    
    **For researchers**: Use this to evaluate model performance
    **For novices**: Shows how well the model predicts exoplanets
    """
    if not model_predictor or not model_predictor.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Load metadata
        metadata_path = Path(settings.MODEL_METADATA_PATH)
        if not metadata_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model metadata not found. Retrain the model to generate statistics."
            )
        
        import joblib
        metadata = joblib.load(metadata_path)
        
        return ModelStatistics(
            accuracy=metadata.get('test_accuracy', 0.0),
            precision=metadata.get('precision', 0.0),
            recall=metadata.get('recall', 0.0),
            f1_score=metadata.get('f1_score', 0.0),
            confusion_matrix=metadata.get('confusion_matrix', [[0, 0], [0, 0]]),
            class_distribution=metadata.get('class_distribution', {}),
            training_date=metadata.get('training_date'),
            training_samples=metadata.get('training_samples', 0),
            cross_validation_scores=metadata.get('cv_scores', [])
        )
    
    except Exception as e:
        logger.error(f"Error getting model statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model statistics: {str(e)}"
        )


@app.get("/model/hyperparameters", response_model=HyperparameterConfig, tags=["Model"])
async def get_hyperparameters():
    """
    Get current model hyperparameters
    
    Returns the hyperparameter configuration used to train the current model.
    
    **For researchers**: Review and compare different configurations
    **For novices**: See what settings control the model's behavior
    """
    if not model_predictor or not model_predictor.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        metadata_path = Path(settings.MODEL_METADATA_PATH)
        if not metadata_path.exists():
            # Return default hyperparameters
            return HyperparameterConfig()
        
        import joblib
        metadata = joblib.load(metadata_path)
        hyperparams = metadata.get('hyperparameters', {})
        
        return HyperparameterConfig(**hyperparams) if hyperparams else HyperparameterConfig()
    
    except Exception as e:
        logger.error(f"Error getting hyperparameters: {e}")
        return HyperparameterConfig()  # Return defaults on error


@app.post("/training/start", response_model=TrainingResponse, tags=["Training"])
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Start model training/retraining with optional custom hyperparameters
    
    This endpoint allows you to:
    - Retrain the model with new data
    - Adjust hyperparameters (learning rate, tree depth, etc.)
    - Include or exclude existing training data
    
    Training runs in the background. Use the returned job_id to check progress.
    
    **For researchers**: Fine-tune model parameters for better accuracy
    **For novices**: Improve the model with new discoveries
    """
    try:
        # Generate unique job ID
        job_id = f"train_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Check if training data exists
        if request.use_existing_data and not training_data_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No existing training data found. Upload data first via /data/ingest"
            )
        
        # Initialize job status
        training_jobs[job_id] = {
            "status": "starting",
            "progress": 0.0,
            "current_step": "Initializing training...",
            "logs": ["Training job created"],
            "started_at": datetime.now(),
            "completed_at": None,
            "error": None
        }
        
        # Get hyperparameters
        hyperparams = request.hyperparameters if request.hyperparameters else HyperparameterConfig()
        
        # Start training in background
        background_tasks.add_task(
            run_training_job,
            job_id,
            hyperparams,
            request.use_existing_data
        )
        
        logger.info(f"🚀 Started training job: {job_id}")
        
        return TrainingResponse(
            status="started",
            job_id=job_id,
            message=f"Model training started. Use /training/status/{job_id} to check progress.",
            estimated_time=300  # Estimate 5 minutes
        )
    
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start training: {str(e)}"
        )


@app.get("/training/status/{job_id}", response_model=TrainingStatusResponse, tags=["Training"])
async def get_training_status(job_id: str):
    """
    Check the status of a training job
    
    Use the job_id returned from /training/start to monitor progress.
    
    **For researchers**: Monitor training progress and check for convergence
    **For novices**: See if your model is ready to use
    """
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found"
        )
    
    job_status = training_jobs[job_id]
    
    return TrainingStatusResponse(
        job_id=job_id,
        status=job_status["status"],
        progress=job_status["progress"],
        current_step=job_status.get("current_step"),
        logs=job_status.get("logs", [])[-10:],  # Last 10 log entries
        started_at=job_status.get("started_at"),
        completed_at=job_status.get("completed_at"),
        error=job_status.get("error")
    )


@app.get("/training/history", tags=["Training"])
async def get_training_history():
    """
    Get history of all training jobs
    
    **For researchers**: Track model evolution and experiments
    **For novices**: See when the model was last updated
    """
    return {
        "total_jobs": len(training_jobs),
        "jobs": [
            {
                "job_id": job_id,
                "status": job["status"],
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at")
            }
            for job_id, job in training_jobs.items()
        ]
    }


@app.post("/data/ingest", response_model=DataIngestionResponse, tags=["Data Management"])
async def ingest_data(
    file: UploadFile = File(...),
    request: DataIngestionRequest = DataIngestionRequest(),
    background_tasks: BackgroundTasks = None
):
    """
    Upload new training data to improve the model
    
    Upload a CSV file with labeled exoplanet data. The data will be:
    - Validated for correct format
    - Added to the training dataset
    - Optionally used to retrain the model immediately
    
    CSV should include:
    - Feature columns (koi_period, koi_duration, etc.)
    - Target column (koi_disposition: CANDIDATE, CONFIRMED, or FALSE POSITIVE)
    
    **For researchers**: Contribute new observations to improve predictions
    **For novices**: Add newly confirmed exoplanets to make the model smarter
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are accepted"
        )
    
    try:
        # Read uploaded CSV with encoding detection
        contents = await file.read()
        
        # Try multiple encodings
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
        new_data = None
        encoding_used = None
        
        for encoding in encodings_to_try:
            try:
                new_data = pd.read_csv(io.BytesIO(contents), encoding=encoding)
                encoding_used = encoding
                logger.info(f"✅ Successfully decoded CSV with {encoding} encoding")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                # If it's not an encoding error, try next encoding
                if "codec" not in str(e).lower() and "decode" not in str(e).lower():
                    raise
                continue
        
        if new_data is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to decode CSV file. Please ensure it's a valid CSV with UTF-8, Latin-1, or Windows-1252 encoding."
            )
        
        logger.info(f"📊 Received data file: {file.filename} (encoding: {encoding_used})")
        logger.info(f"   Rows: {len(new_data)}, Columns: {len(new_data.columns)}")
        
        # Validate data has target column
        target_columns = ['koi_disposition', 'disposition', 'label', 'target']
        target_col = None
        for col in target_columns:
            if col in new_data.columns:
                target_col = col
                break
        
        if not target_col:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"CSV must contain a target column: {target_columns}"
            )
        
        # Create data directory if it doesn't exist
        training_data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle data storage
        rows_added = len(new_data)
        
        if request.append_to_existing and training_data_path.exists():
            # Append to existing data
            existing_data = pd.read_csv(training_data_path)
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            combined_data.to_csv(training_data_path, index=False)
            total_rows = len(combined_data)
            logger.info(f"✅ Appended {rows_added} rows. Total: {total_rows}")
        else:
            # Replace existing data
            new_data.to_csv(training_data_path, index=False)
            total_rows = rows_added
            logger.info(f"✅ Stored {rows_added} rows (replaced existing data)")
        
        # Optionally trigger retraining
        training_job_id = None
        if request.auto_retrain and background_tasks:
            job_id = f"train_{time.strftime('%Y%m%d_%H%M%S')}_auto"
            training_jobs[job_id] = {
                "status": "starting",
                "progress": 0.0,
                "current_step": "Auto-training after data ingestion...",
                "logs": ["Auto-training triggered by data ingestion"],
                "started_at": datetime.now(),
                "completed_at": None,
                "error": None
            }
            background_tasks.add_task(
                run_training_job,
                job_id,
                HyperparameterConfig(),
                True
            )
            training_job_id = job_id
            logger.info(f"🚀 Auto-training started: {job_id}")
        
        return DataIngestionResponse(
            status="success",
            message=f"Successfully added {rows_added} new samples",
            rows_added=rows_added,
            total_rows=total_rows,
            training_triggered=request.auto_retrain,
            training_job_id=training_job_id
        )
    
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV file is empty"
        )
    except pd.errors.ParserError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid CSV format"
        )
    except Exception as e:
        logger.error(f"Error ingesting data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data ingestion failed: {str(e)}"
        )


@app.get("/data/info", tags=["Data Management"])
async def get_data_info():
    """
    Get information about the current training dataset
    
    **For researchers**: Understand dataset composition
    **For novices**: See how much data the model learned from
    """
    if not training_data_path.exists():
        return {
            "status": "no_data",
            "message": "No training data available",
            "total_samples": 0,
            "class_distribution": {},
            "last_updated": None
        }
    
    try:
        data = pd.read_csv(training_data_path)
        
        # Find target column
        target_col = None
        for col in ['koi_disposition', 'disposition', 'label', 'target']:
            if col in data.columns:
                target_col = col
                break
        
        class_dist = {}
        if target_col:
            class_dist = data[target_col].value_counts().to_dict()
        
        # Get file modification time
        import os
        last_updated = datetime.fromtimestamp(os.path.getmtime(training_data_path))
        
        return {
            "status": "available",
            "total_samples": len(data),
            "total_features": len(data.columns) - 1 if target_col else len(data.columns),
            "class_distribution": class_dist,
            "last_updated": last_updated,
            "file_path": str(training_data_path)
        }
    
    except Exception as e:
        logger.error(f"Error getting data info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve data info: {str(e)}"
        )


@app.post("/planets/save", response_model=PlanetDataResponse, tags=["Planet Data"])
async def save_planet_data(planet: PlanetData):
    """
    Save data about a single planet/KOI observation
    
    This endpoint allows you to store detailed information about individual
    exoplanet candidates or confirmed planets for later retrieval.
    
    **Use Cases:**
    - Store observations from telescope data
    - Keep track of interesting candidates
    - Build a personal catalog of exoplanets
    - Share data with collaborators
    
    **For Researchers:** Maintain detailed records with custom notes
    **For Novices:** Save your favorite exoplanet discoveries
    """
    try:
        # Generate planet_id if not provided
        if not planet.planet_id:
            # Generate unique ID based on timestamp
            planet.planet_id = f"PLT_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Create planet data dict
        planet_dict = planet.model_dump()
        
        # Save to JSON file
        planet_file = planet_storage_path / f"{planet.planet_id}.json"
        
        with open(planet_file, 'w') as f:
            # Convert datetime to string for JSON serialization
            planet_dict_json = planet_dict.copy()
            planet_dict_json['submitted_at'] = planet_dict['submitted_at'].isoformat()
            import json
            json.dump(planet_dict_json, f, indent=2)
        
        logger.info(f"💾 Saved planet data: {planet.planet_id}")
        
        return PlanetDataResponse(
            status="success",
            message=f"Planet data saved successfully",
            planet_id=planet.planet_id,
            data=planet
        )
    
    except Exception as e:
        logger.error(f"Error saving planet data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save planet data: {str(e)}"
        )


@app.get("/planets/{planet_id}", response_model=PlanetDataResponse, tags=["Planet Data"])
async def get_planet_data(planet_id: str):
    """
    Retrieve data about a specific planet by ID
    
    Use the planet_id from the save operation to retrieve the stored data.
    
    **Use Cases:**
    - Retrieve previously saved observations
    - Share planet data by ID
    - Review stored information
    - Export data for analysis
    
    **For Researchers:** Access detailed observations for papers
    **For Novices:** Look up your saved planets
    """
    try:
        planet_file = planet_storage_path / f"{planet_id}.json"
        
        if not planet_file.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Planet with ID '{planet_id}' not found"
            )
        
        # Load from JSON file
        import json
        with open(planet_file, 'r') as f:
            planet_dict = json.load(f)
        
        # Convert ISO string back to datetime
        planet_dict['submitted_at'] = datetime.fromisoformat(planet_dict['submitted_at'])
        
        planet_data = PlanetData(**planet_dict)
        
        logger.info(f"📖 Retrieved planet data: {planet_id}")
        
        return PlanetDataResponse(
            status="success",
            message="Planet data retrieved successfully",
            planet_id=planet_id,
            data=planet_data
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving planet data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve planet data: {str(e)}"
        )


@app.get("/planets/list", response_model=PlanetListResponse, tags=["Planet Data"])
async def list_planets(
    limit: int = 100,
    offset: int = 0,
    disposition: Optional[str] = None
):
    """
    List all stored planet data with optional filtering
    
    **Query Parameters:**
    - limit: Maximum number of results (default: 100)
    - offset: Number of results to skip (default: 0)
    - disposition: Filter by classification (CANDIDATE, CONFIRMED, FALSE POSITIVE)
    
    **Use Cases:**
    - Browse your saved planets
    - Filter by confirmation status
    - Export catalog to CSV
    - Review all observations
    
    **For Researchers:** Manage your observation database
    **For Novices:** See all your saved discoveries
    """
    try:
        # Get all planet JSON files
        planet_files = list(planet_storage_path.glob("*.json"))
        
        planets = []
        import json
        
        for planet_file in planet_files:
            try:
                with open(planet_file, 'r') as f:
                    planet_dict = json.load(f)
                
                # Convert ISO string back to datetime
                planet_dict['submitted_at'] = datetime.fromisoformat(planet_dict['submitted_at'])
                
                planet_data = PlanetData(**planet_dict)
                
                # Apply disposition filter if provided
                if disposition and planet_data.disposition != disposition:
                    continue
                
                planets.append(planet_data)
            
            except Exception as e:
                logger.warning(f"Failed to load planet file {planet_file}: {e}")
                continue
        
        # Sort by submission date (newest first)
        planets.sort(key=lambda p: p.submitted_at, reverse=True)
        
        # Apply pagination
        total_count = len(planets)
        planets_page = planets[offset:offset + limit]
        
        logger.info(f"📋 Listed {len(planets_page)} planets (total: {total_count})")
        
        return PlanetListResponse(
            total_count=total_count,
            planets=planets_page
        )
    
    except Exception as e:
        logger.error(f"Error listing planets: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list planets: {str(e)}"
        )


@app.delete("/planets/{planet_id}", tags=["Planet Data"])
async def delete_planet_data(planet_id: str):
    """
    Delete a planet's stored data
    
    **Warning:** This action cannot be undone!
    
    **Use Cases:**
    - Remove duplicate entries
    - Clean up test data
    - Delete outdated observations
    
    **For Researchers:** Manage your database
    **For Novices:** Remove unwanted entries
    """
    try:
        planet_file = planet_storage_path / f"{planet_id}.json"
        
        if not planet_file.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Planet with ID '{planet_id}' not found"
            )
        
        # Delete the file
        planet_file.unlink()
        
        logger.info(f"🗑️ Deleted planet data: {planet_id}")
        
        return {
            "status": "success",
            "message": f"Planet '{planet_id}' deleted successfully",
            "planet_id": planet_id,
            "timestamp": datetime.now()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting planet data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete planet data: {str(e)}"
        )


@app.post("/planets/{planet_id}/predict", response_model=PredictionResponse, tags=["Planet Data"])
async def predict_stored_planet(planet_id: str):
    """
    Get ML prediction for a stored planet
    
    Loads the planet data and runs it through the classification model.
    
    **Use Cases:**
    - Classify previously saved observations
    - Update predictions with new model
    - Batch process saved planets
    
    **For Researchers:** Evaluate candidates with updated models
    **For Novices:** Check if your planet is likely confirmed
    """
    try:
        # Load planet data
        planet_file = planet_storage_path / f"{planet_id}.json"
        
        if not planet_file.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Planet with ID '{planet_id}' not found"
            )
        
        import json
        with open(planet_file, 'r') as f:
            planet_dict = json.load(f)
        
        # Extract features for prediction
        features = {}
        
        # Add main features
        feature_fields = [
            'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration', 
            'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol', 
            'koi_steff', 'koi_slogg', 'koi_srad', 'koi_kepmag'
        ]
        
        for field in feature_fields:
            if planet_dict.get(field) is not None:
                features[field] = planet_dict[field]
        
        # Add additional features
        if planet_dict.get('additional_features'):
            features.update(planet_dict['additional_features'])
        
        if not features:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No features found for prediction"
            )
        
        # Make prediction
        if not model_predictor or not model_predictor.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        prediction, label, confidence, probabilities = model_predictor.predict(features)
        
        model_info = model_predictor.get_model_info()
        model_version = model_info.get("model_version", "unknown")
        
        logger.info(f"🔮 Predicted stored planet {planet_id}: {label} ({confidence:.2%})")
        
        return PredictionResponse(
            prediction=prediction,
            prediction_label=label,
            confidence=confidence,
            probabilities=probabilities,
            model_version=model_version
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting stored planet: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/planets/predict-and-save", response_model=SimplePlanetPredictionResponse, tags=["Planet Data"])
async def predict_and_save_planet(planet: SimplePlanetInput):
    """
    Predict classification for a new planet and save it to My Planets
    
    This endpoint:
    1. Accepts simple planet input with key features
    2. Makes a prediction using the ML model
    3. Saves the planet data with prediction to your planet collection
    4. Returns the prediction result and planet ID
    
    **Key Features to Provide:**
    - koi_period: Orbital period (days)
    - koi_depth: Transit depth (ppm)
    - koi_prad: Planetary radius (Earth radii)
    - koi_teq: Equilibrium temperature (K)
    - koi_steff: Stellar effective temperature (K)
    - koi_srad: Stellar radius (solar radii)
    
    **Use Cases:**
    - Quick planet classification
    - Add discovered planets to your collection
    - Test planet parameters
    
    **Response includes:**
    - Prediction label (CONFIRMED, CANDIDATE, FALSE POSITIVE)
    - Confidence score
    - Probabilities for each class
    - Saved planet ID for future reference
    """
    try:
        # Check if model is loaded
        if not model_predictor or not model_predictor.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please wait for model initialization."
            )
        
        # Extract features from input
        features = {}
        planet_dict = planet.dict(exclude_unset=True, exclude={'planet_name', 'notes'})
        
        # Add provided features
        for key, value in planet_dict.items():
            if value is not None:
                features[key] = value
        
        if not features:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No features provided. Please provide at least some planet parameters."
            )
        
        logger.info(f"🔮 Making prediction with {len(features)} features")
        
        # Make prediction
        prediction, label, confidence, probabilities = model_predictor.predict(features)
        
        # Generate unique planet ID
        import uuid
        from datetime import datetime
        planet_id = f"PLT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        # Prepare planet data for storage
        planet_data = {
            "planet_id": planet_id,
            "planet_name": planet.planet_name or f"Planet {planet_id}",
            "notes": planet.notes,
            "disposition": label,  # Save the prediction
            "prediction_confidence": confidence,
            "prediction_probabilities": probabilities,
            "submitted_at": datetime.now().isoformat(),
            "predicted_at": datetime.now().isoformat(),
        }
        
        # Add all input features
        for key, value in planet_dict.items():
            if value is not None:
                planet_data[key] = value
        
        # Save to planet storage
        import json
        planet_file = planet_storage_path / f"{planet_id}.json"
        with open(planet_file, 'w') as f:
            json.dump(planet_data, f, indent=2)
        
        logger.info(f"✅ Planet saved: {planet_id} | Prediction: {label} ({confidence:.2%})")
        
        # Prepare response
        return SimplePlanetPredictionResponse(
            status="success",
            message=f"Planet analyzed and saved successfully as '{planet_data['planet_name']}'",
            planet_id=planet_id,
            prediction=prediction,
            prediction_label=label,
            confidence=confidence,
            probabilities=probabilities,
            planet_data={
                "planet_name": planet_data['planet_name'],
                "prediction": label,
                "confidence": f"{confidence:.2%}",
                **{k: v for k, v in planet_dict.items() if v is not None}
            },
            saved_at=datetime.now()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict-and-save: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to predict and save planet: {str(e)}"
        )


@app.get("/planets/predict", response_model=SimplePlanetPredictionResponse, tags=["Planet Prediction"])
async def get_planet_prediction_by_name(planet_name: str):
    """
    Get prediction for a saved planet by its name
    
    **Endpoint:** `GET /planets/predict?planet_name=Kepler-442b`
    
    **Use Case:**
    - Retrieve prediction for a planet from "My Planets" page
    - Only requires the planet name
    
    **Parameters:**
    - `planet_name`: Name of the saved planet (query parameter)
    
    **Example:**
    ```
    GET /planets/predict?planet_name=Kepler-442b
    ```
    
    **Response:**
    ```json
    {
      "status": "success",
      "message": "Prediction retrieved successfully",
      "planet_id": "planet_1234567890",
      "prediction": 0,
      "prediction_label": "CONFIRMED",
      "confidence": 0.95,
      "probabilities": {
        "CONFIRMED": 0.95,
        "CANDIDATE": 0.04,
        "FALSE POSITIVE": 0.01
      },
      "planet_data": {...},
      "saved_at": "2024-01-15T10:30:00"
    }
    ```
    """
    try:
        # Search through all saved planets to find matching name
        import json
        
        if not planet_storage_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No planets have been saved yet"
            )
        
        # Find planet file by name
        found_planet = None
        found_planet_id = None
        
        for planet_file in planet_storage_path.glob("planet_*.json"):
            with open(planet_file, 'r') as f:
                planet_data = json.load(f)
                if planet_data.get('planet_name') == planet_name:
                    found_planet = planet_data
                    found_planet_id = planet_file.stem
                    break
        
        if not found_planet:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Planet '{planet_name}' not found. Make sure you've saved it first using POST /planets/predict-and-save"
            )
        
        # Extract prediction information
        prediction = found_planet.get('prediction')
        prediction_label = found_planet.get('disposition', 'UNKNOWN')
        confidence = found_planet.get('prediction_confidence', 0.0)
        probabilities = found_planet.get('prediction_probabilities', {})
        saved_at = found_planet.get('predicted_at') or found_planet.get('submitted_at')
        
        logger.info(f"📊 Retrieved prediction for planet: {planet_name} ({found_planet_id}) | {prediction_label}")
        
        return SimplePlanetPredictionResponse(
            status="success",
            message=f"Prediction retrieved successfully for '{planet_name}'",
            planet_id=found_planet_id,
            prediction=prediction,
            prediction_label=prediction_label,
            confidence=confidence,
            probabilities=probabilities,
            planet_data=found_planet,
            saved_at=datetime.fromisoformat(saved_at) if saved_at else datetime.now()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving prediction by name: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve prediction: {str(e)}"
        )


# Background training function
async def run_training_job(job_id: str, hyperparams: HyperparameterConfig, use_existing_data: bool):
    """
    Background task to train/retrain the model
    """
    try:
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["progress"] = 5.0
        training_jobs[job_id]["logs"].append("Loading training data...")
        
        # Load data
        if not training_data_path.exists():
            raise FileNotFoundError("Training data not found")
        
        data = pd.read_csv(training_data_path)
        training_jobs[job_id]["logs"].append(f"Loaded {len(data)} samples")
        training_jobs[job_id]["progress"] = 10.0
        
        # Find target column
        target_col = None
        for col in ['koi_disposition', 'disposition', 'label', 'target']:
            if col in data.columns:
                target_col = col
                break
        
        if not target_col:
            raise ValueError("No target column found in data")
        
        # Prepare data
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        import lightgbm as lgb
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        training_jobs[job_id]["logs"].append("Preprocessing data...")
        training_jobs[job_id]["current_step"] = "Data preprocessing"
        training_jobs[job_id]["progress"] = 20.0
        
        # Separate features and target
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Handle missing values
        X = X.fillna(X.median(numeric_only=True))
        
        # One-hot encode categorical features
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        
        training_jobs[job_id]["logs"].append(f"Train: {len(X_train)}, Test: {len(X_test)}")
        training_jobs[job_id]["progress"] = 30.0
        
        # Configure base estimators
        training_jobs[job_id]["logs"].append("Configuring models...")
        training_jobs[job_id]["current_step"] = "Configuring LightGBM and XGBoost"
        
        device = 'gpu' if hyperparams.use_gpu else 'cpu'
        
        lgb_params = {
            'n_estimators': hyperparams.lgbm_n_estimators,
            'learning_rate': hyperparams.lgbm_learning_rate,
            'max_depth': hyperparams.lgbm_max_depth,
            'num_leaves': hyperparams.lgbm_num_leaves,
            'device': device,
            'verbose': -1,
            'random_state': 42
        }
        
        xgb_params = {
            'n_estimators': hyperparams.xgb_n_estimators,
            'learning_rate': hyperparams.xgb_learning_rate,
            'max_depth': hyperparams.xgb_max_depth,
            'tree_method': 'hist',
            'device': 'cuda' if hyperparams.use_gpu else 'cpu',
            'random_state': 42
        }
        
        # Train models
        training_jobs[job_id]["logs"].append("Training LightGBM...")
        training_jobs[job_id]["current_step"] = "Training LightGBM"
        training_jobs[job_id]["progress"] = 40.0
        
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        
        training_jobs[job_id]["logs"].append("Training XGBoost...")
        training_jobs[job_id]["current_step"] = "Training XGBoost"
        training_jobs[job_id]["progress"] = 60.0
        
        xgb_model = xgb.XGBClassifier(**xgb_params)
        
        # Create stacking ensemble
        training_jobs[job_id]["logs"].append("Creating stacking ensemble...")
        training_jobs[job_id]["current_step"] = "Stacking ensemble"
        training_jobs[job_id]["progress"] = 70.0
        
        stacking_clf = StackingClassifier(
            estimators=[
                ('lgbm', lgb_model),
                ('xgb', xgb_model)
            ],
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=hyperparams.cv_folds,
            n_jobs=1,
            verbose=0
        )
        
        # Train stacking model
        training_jobs[job_id]["logs"].append("Training stacking classifier...")
        training_jobs[job_id]["current_step"] = f"Stacking with {hyperparams.cv_folds}-fold CV"
        training_jobs[job_id]["progress"] = 80.0
        
        stacking_clf.fit(X_train, y_train)
        
        # Evaluate
        training_jobs[job_id]["logs"].append("Evaluating model...")
        training_jobs[job_id]["current_step"] = "Model evaluation"
        training_jobs[job_id]["progress"] = 90.0
        
        y_pred = stacking_clf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()
        
        # Save model
        training_jobs[job_id]["logs"].append("Saving model...")
        training_jobs[job_id]["current_step"] = "Saving model"
        training_jobs[job_id]["progress"] = 95.0
        
        model_path = Path(settings.MODEL_PATH)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(stacking_clf, model_path)
        
        # Save metadata
        metadata = {
            'model_type': 'StackingClassifier',
            'feature_names': list(X_encoded.columns),
            'label_encoder': le,
            'test_accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': conf_matrix,
            'class_distribution': dict(pd.Series(y).value_counts()),
            'training_date': datetime.now(),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'hyperparameters': hyperparams.model_dump(),
            'gpu_accelerated': hyperparams.use_gpu
        }
        
        metadata_path = Path(settings.MODEL_METADATA_PATH)
        joblib.dump(metadata, metadata_path)
        
        # Reload model in predictor
        training_jobs[job_id]["logs"].append("Reloading model...")
        model_predictor._load_model()
        
        # Complete
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["progress"] = 100.0
        training_jobs[job_id]["current_step"] = "Completed"
        training_jobs[job_id]["completed_at"] = datetime.now()
        training_jobs[job_id]["logs"].append(f"✅ Training completed! Accuracy: {accuracy:.4f}")
        
        logger.info(f"✅ Training job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Training job {job_id} failed: {e}")
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)
        training_jobs[job_id]["completed_at"] = datetime.now()
        training_jobs[job_id]["logs"].append(f"❌ Error: {str(e)}")


# ==============================================================================
# FLEXIBLE PLANET DATA ENDPOINTS - Smart CSV Detection & Single Entry
# ==============================================================================

@app.post("/flexible-planets/upload-csv", response_model=CSVUploadResponse)
async def upload_flexible_planets_csv(
    file: UploadFile = File(...),
    auto_map_columns: bool = True
):
    """
    Upload a CSV file with flexible planet data. 
    Automatically detects columns and maps them to planet properties.
    
    Features:
    - Intelligently reads CSV structure
    - Maps common column names (color, name, distance, radius, etc.)
    - Handles different column formats
    - Saves each planet as individual JSON file
    
    Example CSV columns it can handle:
    - name, planet_name, kepler_name
    - color, colour, planet_color
    - distance, distance_ly, dist
    - radius, planet_radius, rad
    - mass, planet_mass
    - temperature, temp, surface_temp
    - etc.
    """
    try:
        # Validate file
        if not file or not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file uploaded. Please select a valid CSV file."
            )
        
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file format. Only CSV files (.csv) are accepted. You uploaded: {file.filename}"
            )
        
        # Read CSV content with encoding detection
        contents = await file.read()
        
        # Try multiple encodings
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
        df = None
        encoding_used = None
        
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(io.BytesIO(contents), encoding=encoding)
                encoding_used = encoding
                logger.info(f"✅ Successfully decoded CSV with {encoding} encoding")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                # If it's not an encoding error, try next encoding
                if "codec" not in str(e).lower() and "decode" not in str(e).lower():
                    raise
                continue
        
        if df is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to decode CSV file. Please ensure it's a valid CSV file with UTF-8, Latin-1, or Windows-1252 encoding. The file may be corrupted or in an unsupported format."
            )
        
        if df.empty or len(df) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The CSV file is empty. Please upload a CSV file with data rows."
            )
        
        logger.info(f"📊 CSV uploaded with {len(df)} rows and {len(df.columns)} columns (encoding: {encoding_used})")
        logger.info(f"📋 Columns detected: {list(df.columns)}")
        
        # Detect and map column names
        column_mapping = _detect_column_mapping(df.columns.tolist())
        
        logger.info(f"🗺️ Column mapping: {column_mapping}")
        
        # Save planets
        saved_planet_ids = []
        
        for idx, row in df.iterrows():
            # Create flexible planet data from row
            planet_data = _create_flexible_planet_from_row(row, column_mapping)
            
            # Generate unique ID if not present
            if not planet_data.get("planet_id"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                planet_data["planet_id"] = f"FLX_{timestamp}_{idx}"
            
            # Add metadata
            planet_data["data_source"] = f"CSV Upload: {file.filename}"
            planet_data["submitted_at"] = datetime.now().isoformat()
            
            # Save to file
            planet_id = planet_data["planet_id"]
            file_path = flexible_planet_storage_path / f"{planet_id}.json"
            
            with open(file_path, 'w') as f:
                import json
                json.dump(planet_data, f, indent=2, default=str)
            
            saved_planet_ids.append(planet_id)
        
        logger.info(f"✅ Saved {len(saved_planet_ids)} planets from CSV")
        
        return CSVUploadResponse(
            status="success",
            message=f"Successfully uploaded and saved {len(saved_planet_ids)} planets",
            total_planets=len(saved_planet_ids),
            columns_detected=list(df.columns),
            columns_mapped=column_mapping,
            planets_saved=saved_planet_ids,
            timestamp=datetime.now()
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="CSV file is empty - no data found. Please upload a CSV file with headers and data rows."
        )
    except pd.errors.ParserError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse CSV file: {str(e)}. Please ensure your CSV is properly formatted with consistent columns."
        )
    except UnicodeDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"CSV file encoding error: {str(e)}. Please save your CSV file with UTF-8 encoding."
        )
    except Exception as e:
        logger.error(f"Error uploading CSV: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process CSV file: {str(e)}. Please check the file format and try again."
        )


@app.post("/flexible-planets/save", response_model=FlexiblePlanetResponse)
async def save_flexible_planet(planet: FlexiblePlanetData):
    """
    Save a single planet with flexible properties (color, name, distance, etc.)
    
    Accepts any combination of planet properties:
    - Visual: color, appearance
    - Distance: distance, distance_unit, constellation
    - Physical: radius, mass, density, gravity
    - Orbital: orbital_period, semi_major_axis, eccentricity
    - Temperature: temperature, atmosphere
    - Star: star_name, star_type, star_temperature
    - Discovery: discovery_year, discovery_method, classification
    - Habitability: habitable_zone, earth_similarity_index
    - Custom: additional_properties (any extra fields)
    
    Example:
    {
        "name": "Kepler-442b",
        "color": "blue-green",
        "distance": 1206,
        "distance_unit": "light-years",
        "radius": 1.34,
        "temperature": 233,
        "habitable_zone": true
    }
    """
    try:
        # Generate unique ID if not provided
        if not planet.planet_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            planet.planet_id = f"FLX_{timestamp}"
        
        # Set metadata
        planet.submitted_at = datetime.now()
        planet.data_source = planet.data_source or "Direct API Input"
        
        # Convert to dict for storage
        planet_dict = planet.model_dump()
        
        # Save to file
        file_path = flexible_planet_storage_path / f"{planet.planet_id}.json"
        
        with open(file_path, 'w') as f:
            import json
            json.dump(planet_dict, f, indent=2, default=str)
        
        logger.info(f"✅ Saved flexible planet: {planet.planet_id}")
        
        return FlexiblePlanetResponse(
            status="success",
            message=f"Planet '{planet.name or planet.planet_id}' saved successfully",
            planet_id=planet.planet_id,
            data=planet,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error saving flexible planet: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save planet: {str(e)}"
        )


@app.get("/flexible-planets/{planet_id}", response_model=FlexiblePlanetResponse)
async def get_flexible_planet(planet_id: str):
    """
    Retrieve a flexible planet by ID
    """
    try:
        file_path = flexible_planet_storage_path / f"{planet_id}.json"
        
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Planet '{planet_id}' not found"
            )
        
        with open(file_path, 'r') as f:
            import json
            planet_dict = json.load(f)
        
        # Convert datetime strings back to datetime objects
        if 'submitted_at' in planet_dict and planet_dict['submitted_at']:
            planet_dict['submitted_at'] = datetime.fromisoformat(planet_dict['submitted_at'])
        
        planet_data = FlexiblePlanetData(**planet_dict)
        
        return FlexiblePlanetResponse(
            status="success",
            message=f"Planet '{planet_id}' retrieved successfully",
            planet_id=planet_id,
            data=planet_data,
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving flexible planet: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve planet: {str(e)}"
        )


@app.get("/flexible-planets/list", response_model=FlexiblePlanetListResponse)
async def list_flexible_planets(
    limit: int = 100,
    offset: int = 0,
    name_filter: Optional[str] = None,
    classification: Optional[str] = None,
    habitable_only: bool = False
):
    """
    List all flexible planets with pagination and filtering
    
    Query parameters:
    - limit: Maximum number of results (default: 100)
    - offset: Number of results to skip (default: 0)
    - name_filter: Filter by planet name (partial match)
    - classification: Filter by type (e.g., 'Super-Earth', 'Gas Giant')
    - habitable_only: Show only planets in habitable zone
    """
    try:
        # Get all planet files
        planet_files = list(flexible_planet_storage_path.glob("*.json"))
        
        planets = []
        for file_path in planet_files:
            try:
                with open(file_path, 'r') as f:
                    import json
                    planet_dict = json.load(f)
                
                # Convert datetime strings
                if 'submitted_at' in planet_dict and planet_dict['submitted_at']:
                    planet_dict['submitted_at'] = datetime.fromisoformat(planet_dict['submitted_at'])
                
                planet = FlexiblePlanetData(**planet_dict)
                
                # Apply filters
                if name_filter and planet.name:
                    if name_filter.lower() not in planet.name.lower():
                        continue
                
                if classification and planet.classification:
                    if classification.lower() != planet.classification.lower():
                        continue
                
                if habitable_only:
                    if not planet.habitable_zone:
                        continue
                
                planets.append(planet)
                
            except Exception as e:
                logger.warning(f"Failed to load planet from {file_path}: {e}")
                continue
        
        # Sort by submission date (newest first)
        planets.sort(
            key=lambda x: x.submitted_at if x.submitted_at else datetime.min,
            reverse=True
        )
        
        # Pagination
        total_count = len(planets)
        paginated_planets = planets[offset:offset + limit]
        
        logger.info(f"📋 Listed {len(paginated_planets)} of {total_count} flexible planets")
        
        return FlexiblePlanetListResponse(
            total_count=total_count,
            planets=paginated_planets,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error listing flexible planets: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list planets: {str(e)}"
        )


@app.delete("/flexible-planets/{planet_id}")
async def delete_flexible_planet(planet_id: str):
    """
    Delete a flexible planet by ID
    """
    try:
        file_path = flexible_planet_storage_path / f"{planet_id}.json"
        
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Planet '{planet_id}' not found"
            )
        
        file_path.unlink()
        
        logger.info(f"🗑️ Deleted flexible planet: {planet_id}")
        
        return {
            "status": "success",
            "message": f"Planet '{planet_id}' deleted successfully",
            "planet_id": planet_id,
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting flexible planet: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete planet: {str(e)}"
        )


# Helper functions for column detection and mapping
def _detect_column_mapping(columns: List[str]) -> Dict[str, str]:
    """
    Intelligently detect and map CSV columns to planet properties
    """
    mapping = {}
    
    # Normalize column names for matching
    columns_lower = [col.lower().strip() for col in columns]
    
    # Define possible mappings (flexible matching)
    field_patterns = {
        "name": ["name", "planet_name", "planet", "kepler_name", "common_name", "designation"],
        "color": ["color", "colour", "planet_color", "appearance_color"],
        "distance": ["distance", "distance_ly", "dist", "distance_light_years"],
        "radius": ["radius", "planet_radius", "rad", "prad", "koi_prad"],
        "mass": ["mass", "planet_mass", "pmass"],
        "orbital_period": ["orbital_period", "period", "koi_period", "orbital_period_days"],
        "temperature": ["temperature", "temp", "surface_temp", "koi_teq", "equilibrium_temp"],
        "star_name": ["star_name", "star", "host_star", "stellar_name"],
        "star_type": ["star_type", "stellar_type", "spectral_type"],
        "star_temperature": ["star_temperature", "star_temp", "koi_steff", "stellar_temp"],
        "discovery_year": ["discovery_year", "year", "disc_year"],
        "discovery_method": ["discovery_method", "method", "detection_method"],
        "classification": ["classification", "type", "planet_type", "class"],
        "disposition": ["disposition", "status", "koi_disposition"],
        "constellation": ["constellation", "const"],
        "right_ascension": ["right_ascension", "ra", "r.a."],
        "declination": ["declination", "dec", "decl"],
        "semi_major_axis": ["semi_major_axis", "sma", "orbital_distance", "koi_sma"],
        "eccentricity": ["eccentricity", "ecc", "orbital_eccentricity"],
        "density": ["density", "dens", "planet_density"],
        "gravity": ["gravity", "surface_gravity", "grav"],
        "atmosphere": ["atmosphere", "atm", "atmospheric_composition"],
        "habitable_zone": ["habitable_zone", "habitable", "hz", "in_hz"],
        "earth_similarity_index": ["earth_similarity_index", "esi", "similarity"],
        "common_name": ["common_name", "popular_name", "nickname"]
    }
    
    # Match columns to fields
    for field, patterns in field_patterns.items():
        for col, col_lower in zip(columns, columns_lower):
            for pattern in patterns:
                if pattern in col_lower or col_lower in pattern:
                    mapping[col] = field
                    break
            if col in mapping:
                break
    
    return mapping


def _create_flexible_planet_from_row(row: pd.Series, column_mapping: Dict[str, str]) -> Dict[str, Any]:
    """
    Create a flexible planet dict from a CSV row using column mapping
    """
    planet_dict = {}
    additional_props = {}
    
    for col_name, value in row.items():
        # Skip NaN values
        if pd.isna(value):
            continue
        
        # Get mapped field name
        field_name = column_mapping.get(col_name)
        
        if field_name:
            # Convert to appropriate type
            if field_name in ["radius", "mass", "orbital_period", "temperature", "distance", 
                             "semi_major_axis", "eccentricity", "density", "gravity", 
                             "star_temperature", "right_ascension", "declination", "earth_similarity_index"]:
                try:
                    planet_dict[field_name] = float(value)
                except (ValueError, TypeError):
                    planet_dict[field_name] = str(value)
            elif field_name in ["discovery_year"]:
                try:
                    planet_dict[field_name] = int(value)
                except (ValueError, TypeError):
                    planet_dict[field_name] = str(value)
            elif field_name in ["habitable_zone"]:
                # Convert to boolean
                if isinstance(value, str):
                    planet_dict[field_name] = value.lower() in ["true", "yes", "1", "y"]
                else:
                    planet_dict[field_name] = bool(value)
            else:
                planet_dict[field_name] = str(value)
        else:
            # Store unmapped columns in additional_properties
            additional_props[col_name] = str(value)
    
    # Add additional properties if any
    if additional_props:
        planet_dict["additional_properties"] = additional_props
    
    return planet_dict


# Error handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": str(exc.errors()),
            "body": str(exc.body)
        }
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={"error": "Not Found", "detail": "Endpoint not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 errors"""
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": "An unexpected error occurred"}
    )


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(
        "app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )
