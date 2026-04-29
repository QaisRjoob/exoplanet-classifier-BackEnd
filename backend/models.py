"""
Pydantic models for API request/response validation
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class KOIFeatures(BaseModel):
    """Input features for KOI prediction"""
    # Since your model uses many features after preprocessing,
    # we'll accept a flexible dictionary approach
    features: Dict[str, Any] = Field(
        ..., 
        description="Dictionary of feature names and values"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "koi_period": 15.23,
                    "koi_time0bk": 131.5,
                    "koi_impact": 0.5,
                    "koi_duration": 3.2,
                    "koi_depth": 500.0,
                    "koi_prad": 2.5,
                    "koi_teq": 400.0,
                    "koi_insol": 10.5,
                    "koi_steff": 5800.0,
                    "koi_slogg": 4.5,
                    "koi_srad": 1.0,
                    "koi_kepmag": 14.5
                }
            }
        }


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "prediction": "1",
                "prediction_label": "CANDIDATE",
                "confidence": 0.85,
                "probabilities": {
                    "0": 0.10,
                    "1": 0.85,
                    "2": 0.05
                },
                "timestamp": "2025-10-04T12:00:00",
                "model_version": "1.0.0"
            }
        }
    }
    
    prediction: str = Field(..., description="Predicted disposition (CONFIRMED, CANDIDATE, or FALSE POSITIVE)")
    prediction_label: str = Field(..., description="Human-readable prediction label")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    timestamp: datetime = Field(default_factory=datetime.now)
    model_version: Optional[str] = None


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    features_list: List[Dict[str, Any]] = Field(
        ...,
        description="List of feature dictionaries for batch prediction"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "features_list": [
                    {
                        "koi_period": 15.23,
                        "koi_time0bk": 131.5,
                        "koi_impact": 0.5
                    },
                    {
                        "koi_period": 20.5,
                        "koi_time0bk": 140.2,
                        "koi_impact": 0.3
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse]
    total_count: int
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """Health check response"""
    model_config = {"protected_namespaces": ()}
    
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    model_version: Optional[str] = None
    feature_count: Optional[int] = None
    gpu_accelerated: Optional[bool] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: str
    timestamp: datetime = Field(default_factory=datetime.now)


class CSVPredictionResponse(BaseModel):
    """Response model for CSV file predictions"""
    predictions: List[PredictionResponse]
    total_count: int
    success_count: int
    failed_count: int
    processing_time: float
    errors: List[Dict[str, Any]] = []
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "prediction": "0",
                        "prediction_label": "CONFIRMED",
                        "confidence": 0.92,
                        "probabilities": {"0": 0.92, "1": 0.05, "2": 0.03}
                    }
                ],
                "total_count": 100,
                "success_count": 98,
                "failed_count": 2,
                "processing_time": 1.23,
                "errors": [
                    {"row": 5, "error": "Missing required feature"}
                ],
                "timestamp": "2025-10-04T12:00:00"
            }
        }


class ModelStatistics(BaseModel):
    """Model performance statistics"""
    accuracy: float = Field(..., description="Overall model accuracy")
    precision: float = Field(..., description="Precision score")
    recall: float = Field(..., description="Recall score")
    f1_score: float = Field(..., description="F1 score")
    confusion_matrix: List[List[int]] = Field(..., description="Confusion matrix")
    class_distribution: Dict[str, int] = Field(..., description="Distribution of classes in training data")
    training_date: Optional[datetime] = None
    training_samples: Optional[int] = None
    cross_validation_scores: Optional[List[float]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "accuracy": 0.92,
                "precision": 0.91,
                "recall": 0.89,
                "f1_score": 0.90,
                "confusion_matrix": [[850, 50], [30, 488]],
                "class_distribution": {"CANDIDATE": 900, "CONFIRMED": 518},
                "training_date": "2025-10-04T12:00:00",
                "training_samples": 3307,
                "cross_validation_scores": [0.91, 0.93, 0.92, 0.90, 0.94]
            }
        }


class HyperparameterConfig(BaseModel):
    """Hyperparameter configuration for model training"""
    lgbm_n_estimators: int = Field(default=500, ge=50, le=2000, description="LightGBM number of trees")
    lgbm_learning_rate: float = Field(default=0.1, ge=0.001, le=1.0, description="LightGBM learning rate")
    lgbm_max_depth: int = Field(default=-1, ge=-1, le=50, description="LightGBM max tree depth (-1=unlimited)")
    lgbm_num_leaves: int = Field(default=31, ge=2, le=256, description="LightGBM number of leaves")
    
    xgb_n_estimators: int = Field(default=500, ge=50, le=2000, description="XGBoost number of trees")
    xgb_learning_rate: float = Field(default=0.1, ge=0.001, le=1.0, description="XGBoost learning rate")
    xgb_max_depth: int = Field(default=6, ge=1, le=20, description="XGBoost max tree depth")
    
    cv_folds: int = Field(default=5, ge=2, le=10, description="Cross-validation folds")
    use_gpu: bool = Field(default=True, description="Use GPU acceleration if available")
    
    class Config:
        json_schema_extra = {
            "example": {
                "lgbm_n_estimators": 500,
                "lgbm_learning_rate": 0.1,
                "lgbm_max_depth": -1,
                "lgbm_num_leaves": 31,
                "xgb_n_estimators": 500,
                "xgb_learning_rate": 0.1,
                "xgb_max_depth": 6,
                "cv_folds": 5,
                "use_gpu": True
            }
        }


class TrainingRequest(BaseModel):
    """Request to train/retrain the model with new data"""
    hyperparameters: Optional[HyperparameterConfig] = Field(default=None, description="Custom hyperparameters (optional)")
    use_existing_data: bool = Field(default=True, description="Include existing training data")
    
    class Config:
        json_schema_extra = {
            "example": {
                "hyperparameters": {
                    "lgbm_n_estimators": 600,
                    "cv_folds": 5,
                    "use_gpu": True
                },
                "use_existing_data": True
            }
        }


class TrainingResponse(BaseModel):
    """Response for training request"""
    status: str = Field(..., description="Training status: started, completed, failed")
    job_id: str = Field(..., description="Training job ID for tracking")
    message: str = Field(..., description="Status message")
    estimated_time: Optional[int] = Field(None, description="Estimated time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "started",
                "job_id": "train_20251004_120000",
                "message": "Model training started with 3307 samples",
                "estimated_time": 300,
                "timestamp": "2025-10-04T12:00:00"
            }
        }


class TrainingStatusResponse(BaseModel):
    """Response for training status check"""
    job_id: str
    status: str = Field(..., description="Status: running, completed, failed")
    progress: float = Field(..., ge=0.0, le=100.0, description="Progress percentage")
    current_step: Optional[str] = None
    logs: List[str] = []
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "train_20251004_120000",
                "status": "running",
                "progress": 45.5,
                "current_step": "Training XGBoost (fold 3/5)",
                "logs": ["Started training...", "LightGBM completed", "XGBoost training..."],
                "started_at": "2025-10-04T12:00:00",
                "completed_at": None,
                "error": None
            }
        }


class DataIngestionRequest(BaseModel):
    """Request to add new training data"""
    append_to_existing: bool = Field(default=True, description="Append to existing data or replace")
    auto_retrain: bool = Field(default=False, description="Automatically retrain after ingestion")
    
    class Config:
        json_schema_extra = {
            "example": {
                "append_to_existing": True,
                "auto_retrain": False
            }
        }


class DataIngestionResponse(BaseModel):
    """Response for data ingestion"""
    status: str
    message: str
    rows_added: int
    total_rows: int
    timestamp: datetime = Field(default_factory=datetime.now)
    training_triggered: bool = False
    training_job_id: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Successfully added 100 new samples",
                "rows_added": 100,
                "total_rows": 3407,
                "timestamp": "2025-10-04T12:00:00",
                "training_triggered": False,
                "training_job_id": None
            }
        }


class PlanetData(BaseModel):
    """Single planet/KOI observation data"""
    planet_id: Optional[str] = Field(None, description="Unique identifier (auto-generated if not provided)")
    koi_name: Optional[str] = Field(None, description="KOI designation (e.g., K00001.01)")
    
    # Orbital parameters
    koi_period: Optional[float] = Field(None, description="Orbital period (days)")
    koi_time0bk: Optional[float] = Field(None, description="Transit epoch (BJD)")
    koi_impact: Optional[float] = Field(None, description="Impact parameter")
    koi_duration: Optional[float] = Field(None, description="Transit duration (hours)")
    koi_depth: Optional[float] = Field(None, description="Transit depth (ppm)")
    
    # Planet properties
    koi_prad: Optional[float] = Field(None, description="Planetary radius (Earth radii)")
    koi_teq: Optional[float] = Field(None, description="Equilibrium temperature (K)")
    koi_insol: Optional[float] = Field(None, description="Insolation flux (Earth flux)")
    
    # Stellar properties
    koi_steff: Optional[float] = Field(None, description="Stellar effective temperature (K)")
    koi_slogg: Optional[float] = Field(None, description="Stellar surface gravity (log10(cm/s^2))")
    koi_srad: Optional[float] = Field(None, description="Stellar radius (solar radii)")
    koi_kepmag: Optional[float] = Field(None, description="Kepler magnitude")
    
    # Additional data (all other features as flexible dict)
    additional_features: Optional[Dict[str, Any]] = Field(default={}, description="Additional feature values")
    
    # Metadata
    disposition: Optional[str] = Field(None, description="Classification: CANDIDATE, CONFIRMED, FALSE POSITIVE")
    notes: Optional[str] = Field(None, description="Additional notes or comments")
    submitted_by: Optional[str] = Field(None, description="User/researcher who submitted the data")
    submitted_at: datetime = Field(default_factory=datetime.now, description="Submission timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "planet_id": "PLT_001",
                "koi_name": "K00001.01",
                "koi_period": 15.2345,
                "koi_time0bk": 131.5,
                "koi_impact": 0.5,
                "koi_duration": 3.2,
                "koi_depth": 500.0,
                "koi_prad": 2.5,
                "koi_teq": 400.0,
                "koi_insol": 10.5,
                "koi_steff": 5800.0,
                "koi_slogg": 4.5,
                "koi_srad": 1.0,
                "koi_kepmag": 14.5,
                "disposition": "CANDIDATE",
                "notes": "Interesting candidate for follow-up",
                "submitted_by": "Dr. Smith"
            }
        }


class PlanetDataResponse(BaseModel):
    """Response for planet data operations"""
    status: str
    message: str
    planet_id: str
    data: Optional[PlanetData] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Planet data saved successfully",
                "planet_id": "PLT_001",
                "data": {
                    "planet_id": "PLT_001",
                    "koi_name": "K00001.01",
                    "koi_period": 15.2345,
                    "disposition": "CANDIDATE"
                },
                "timestamp": "2025-10-04T12:00:00"
            }
        }


class PlanetListResponse(BaseModel):
    """Response for listing planets"""
    total_count: int
    planets: List[PlanetData]
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_count": 5,
                "planets": [
                    {
                        "planet_id": "PLT_001",
                        "koi_name": "K00001.01",
                        "koi_period": 15.2345,
                        "disposition": "CANDIDATE"
                    }
                ],
                "timestamp": "2025-10-04T12:00:00"
            }
        }


class FlexiblePlanetData(BaseModel):
    """Flexible model for planet properties (color, name, distance, etc.)"""
    # Basic identification
    planet_id: Optional[str] = Field(None, description="Unique planet identifier")
    name: Optional[str] = Field(None, description="Planet name (e.g., 'Kepler-442b', 'Earth')")
    common_name: Optional[str] = Field(None, description="Common/popular name")
    
    # Visual properties
    color: Optional[str] = Field(None, description="Planet color (e.g., 'blue', 'red', '#FF5733')")
    appearance: Optional[str] = Field(None, description="Visual appearance description")
    
    # Distance and location
    distance: Optional[float] = Field(None, description="Distance from Earth (light-years or parsecs)")
    distance_unit: Optional[str] = Field("light-years", description="Unit of distance measurement")
    constellation: Optional[str] = Field(None, description="Constellation where planet is located")
    right_ascension: Optional[float] = Field(None, description="Right ascension (RA)")
    declination: Optional[float] = Field(None, description="Declination (Dec)")
    
    # Physical properties
    radius: Optional[float] = Field(None, description="Planet radius (Earth radii)")
    mass: Optional[float] = Field(None, description="Planet mass (Earth masses)")
    density: Optional[float] = Field(None, description="Planet density (g/cm³)")
    gravity: Optional[float] = Field(None, description="Surface gravity (m/s²)")
    
    # Orbital properties
    orbital_period: Optional[float] = Field(None, description="Orbital period (days)")
    semi_major_axis: Optional[float] = Field(None, description="Semi-major axis (AU)")
    eccentricity: Optional[float] = Field(None, description="Orbital eccentricity")
    
    # Temperature and atmosphere
    temperature: Optional[float] = Field(None, description="Surface/equilibrium temperature (K)")
    atmosphere: Optional[str] = Field(None, description="Atmospheric composition")
    
    # Star properties
    star_name: Optional[str] = Field(None, description="Host star name")
    star_type: Optional[str] = Field(None, description="Stellar classification (e.g., 'G-type', 'M-dwarf')")
    star_temperature: Optional[float] = Field(None, description="Stellar effective temperature (K)")
    
    # Discovery and classification
    discovery_year: Optional[int] = Field(None, description="Year of discovery")
    discovery_method: Optional[str] = Field(None, description="Detection method (e.g., 'Transit', 'Radial Velocity')")
    classification: Optional[str] = Field(None, description="Planet type (e.g., 'Super-Earth', 'Gas Giant', 'Rocky')")
    disposition: Optional[str] = Field(None, description="Status (CONFIRMED, CANDIDATE, FALSE POSITIVE)")
    
    # Habitability
    habitable_zone: Optional[bool] = Field(None, description="Whether in habitable zone")
    earth_similarity_index: Optional[float] = Field(None, description="ESI score (0-1)")
    
    # Flexible additional data
    additional_properties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Any additional properties")
    
    # Metadata
    notes: Optional[str] = Field(None, description="Additional notes or observations")
    data_source: Optional[str] = Field(None, description="Source of data (e.g., 'NASA Exoplanet Archive', 'User Input')")
    submitted_by: Optional[str] = Field(None, description="Person/team who submitted the data")
    submitted_at: Optional[datetime] = Field(default_factory=datetime.now, description="Submission timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Kepler-442b",
                "color": "blue-green",
                "distance": 1206,
                "distance_unit": "light-years",
                "constellation": "Lyra",
                "radius": 1.34,
                "mass": 2.36,
                "orbital_period": 112.3,
                "temperature": 233,
                "star_name": "Kepler-442",
                "star_type": "K-type",
                "discovery_year": 2015,
                "discovery_method": "Transit",
                "classification": "Super-Earth",
                "disposition": "CONFIRMED",
                "habitable_zone": True,
                "earth_similarity_index": 0.836,
                "notes": "One of the most Earth-like exoplanets discovered"
            }
        }


class FlexiblePlanetResponse(BaseModel):
    """Response for flexible planet data operations"""
    status: str
    message: str
    planet_id: Optional[str] = None
    data: Optional[FlexiblePlanetData] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class FlexiblePlanetListResponse(BaseModel):
    """Response for listing flexible planets"""
    total_count: int
    planets: List[FlexiblePlanetData]
    columns_found: Optional[List[str]] = Field(None, description="Columns found in uploaded CSV")
    timestamp: datetime = Field(default_factory=datetime.now)


class CSVUploadResponse(BaseModel):
    """Response for CSV upload with column detection"""
    status: str
    message: str
    total_planets: int
    columns_detected: List[str]
    columns_mapped: Dict[str, str]
    planets_saved: List[str]
    timestamp: datetime = Field(default_factory=datetime.now)


class SimplePlanetInput(BaseModel):
    """Simple planet input form with key features"""
    
    # Metadata
    planet_name: Optional[str] = Field(None, description="Custom planet name (optional)")
    notes: Optional[str] = Field(None, description="Additional notes (optional)")
    
    # Key features (most important for classification)
    koi_period: Optional[float] = Field(None, description="Orbital period (days)")
    koi_depth: Optional[float] = Field(None, description="Transit depth (ppm)")
    koi_prad: Optional[float] = Field(None, description="Planetary radius (Earth radii)")
    koi_teq: Optional[float] = Field(None, description="Equilibrium temperature (K)")
    koi_insol: Optional[float] = Field(None, description="Insolation flux (Earth flux)")
    koi_model_snr: Optional[float] = Field(None, description="Transit model signal-to-noise ratio")
    koi_steff: Optional[float] = Field(None, description="Stellar effective temperature (K)")
    koi_srad: Optional[float] = Field(None, description="Stellar radius (solar radii)")
    koi_smass: Optional[float] = Field(None, description="Stellar mass (solar masses)")
    
    # Additional common features
    koi_impact: Optional[float] = Field(None, description="Impact parameter")
    koi_duration: Optional[float] = Field(None, description="Transit duration (hours)")
    koi_time0bk: Optional[float] = Field(None, description="Transit epoch (BJD)")
    koi_eccen: Optional[float] = Field(None, description="Orbital eccentricity")
    koi_slogg: Optional[float] = Field(None, description="Stellar surface gravity (log g)")
    koi_kepmag: Optional[float] = Field(None, description="Kepler magnitude")
    
    class Config:
        json_schema_extra = {
            "example": {
                "planet_name": "My Exoplanet Discovery",
                "koi_period": 10.5,
                "koi_depth": 500.0,
                "koi_prad": 2.5,
                "koi_teq": 400.0,
                "koi_insol": 10.5,
                "koi_model_snr": 15.2,
                "koi_steff": 5800.0,
                "koi_srad": 1.0,
                "koi_smass": 1.0,
                "koi_kepmag": 14.5,
                "notes": "Interesting Earth-like candidate"
            }
        }


class SimplePlanetPredictionResponse(BaseModel):
    """Response after predicting and saving a simple planet"""
    status: str
    message: str
    planet_id: str
    prediction: int
    prediction_label: str
    confidence: float
    probabilities: Dict[str, float]
    planet_data: Optional[Dict[str, Any]] = None
    saved_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Planet analyzed and saved successfully",
                "planet_id": "PLT_20251004_120000",
                "prediction": 0,
                "prediction_label": "CONFIRMED",
                "confidence": 0.8523,
                "probabilities": {
                    "CONFIRMED": 0.8523,
                    "CANDIDATE": 0.1201,
                    "FALSE POSITIVE": 0.0276
                },
                "planet_data": {
                    "planet_name": "My Exoplanet Discovery",
                    "koi_period": 10.5,
                    "koi_prad": 2.5
                },
                "saved_at": "2025-10-04T12:00:00"
            }
        }
