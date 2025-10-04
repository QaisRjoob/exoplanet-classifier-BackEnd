"""
ML Model Handler for KOI Classification
Handles model loading, caching, and predictions
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KOIModelPredictor:
    """Handles loading and predictions for KOI classification model"""
    
    def __init__(self, model_path: str, metadata_path: Optional[str] = None):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.model = None
        self.metadata = None
        self.feature_names = None
        self.feature_count = 0
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and metadata"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = joblib.load(self.model_path)
            logger.info("✅ Model loaded successfully")
            logger.info(f"   Model type: {type(self.model).__name__}")
            
            # Try to get feature names from model if available
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = self.model.feature_names_in_.tolist()
                self.feature_count = len(self.feature_names)
                logger.info(f"   Features from model: {self.feature_count}")
            
            # Load metadata if available
            if self.metadata_path and self.metadata_path.exists():
                self.metadata = joblib.load(self.metadata_path)
                logger.info(f"✅ Metadata loaded")
                
                if 'num_features' in self.metadata:
                    self.feature_count = self.metadata['num_features']
                    logger.info(f"   Features from metadata: {self.feature_count}")
                
                if 'gpu_accelerated' in self.metadata:
                    logger.info(f"   GPU accelerated: {self.metadata['gpu_accelerated']}")
            else:
                logger.warning("⚠️ No metadata file found")
                
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def _prepare_features(self, features: Dict) -> pd.DataFrame:
        """
        Prepare features for prediction
        Handles missing features and ensures correct format
        """
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # If we know the expected features, ensure they're all present
        if self.feature_names:
            # Add missing features with 0 (or could use mean/median from training)
            # Using dict comprehension to add all missing columns at once to avoid fragmentation
            missing_cols = {col: 0 for col in self.feature_names if col not in df.columns}
            if missing_cols:
                # Add all missing columns at once using pd.concat to avoid fragmentation
                missing_df = pd.DataFrame([missing_cols])
                df = pd.concat([df, missing_df], axis=1)
            
            # Ensure correct order
            df = df[self.feature_names]
        
        return df
    
    def predict(self, features: Dict) -> Tuple[str, str, float, Dict[str, float]]:
        """
        Make a single prediction
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Tuple of (prediction, label, confidence, probabilities)
        """
        try:
            # Prepare features
            df = self._prepare_features(features)
            
            # Get prediction
            prediction = self.model.predict(df)[0]
            
            # Map prediction to label (3-class classification)
            label_map = {
                0: 'CONFIRMED',
                1: 'CANDIDATE', 
                2: 'FALSE POSITIVE'
            }
            prediction_label = label_map.get(int(prediction), str(prediction))
            
            # Get probabilities if available
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(df)[0]
                confidence = float(np.max(proba))
                
                # Get class labels
                if hasattr(self.model, 'classes_'):
                    classes = self.model.classes_
                else:
                    classes = [0, 1, 2]  # Support 3 classes by default
                
                probabilities = {str(cls): float(prob) for cls, prob in zip(classes, proba)}
            else:
                confidence = 1.0
                probabilities = {str(prediction): 1.0}
            
            return str(prediction), prediction_label, confidence, probabilities
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            raise
    
    def predict_batch(self, features_list: List[Dict]) -> List[Tuple[str, str, float, Dict[str, float]]]:
        """
        Make batch predictions
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            List of (prediction, label, confidence, probabilities) tuples
        """
        results = []
        for features in features_list:
            try:
                results.append(self.predict(features))
            except Exception as e:
                logger.error(f"Error in batch prediction: {e}")
                # Return error prediction
                results.append(("error", "ERROR", 0.0, {"error": 1.0}))
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        info = {
            "model_type": str(type(self.model).__name__),
            "model_loaded": self.model is not None,
            "feature_count": self.feature_count,
        }
        
        if self.metadata:
            info.update({
                "training_date": self.metadata.get('training_date'),
                "model_version": self.metadata.get('model_version', '1.0.0'),
                "test_accuracy": self.metadata.get('test_accuracy'),
                "test_f1_score": self.metadata.get('test_f1_score'),
                "gpu_accelerated": self.metadata.get('gpu_accelerated', False),
                "base_estimators": self.metadata.get('base_estimators', []),
                "training_samples": self.metadata.get('training_samples'),
                "test_samples": self.metadata.get('test_samples'),
            })
        else:
            info["model_version"] = "1.0.0"
        
        return info
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
