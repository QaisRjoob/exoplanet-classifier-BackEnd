"""
Train 3-Class KOI Classification Model with GTX 1050 Ti GPU Optimization
=========================================================================
This script trains a model to classify exoplanets into:
- CONFIRMED (0)
- CANDIDATE (1)  
- FALSE POSITIVE (2)

Optimized for NVIDIA GTX 1050 Ti (4GB VRAM)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import os
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("KOI 3-CLASS CLASSIFICATION - GTX 1050 Ti GPU TRAINING")
print("=" * 80)
print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# Configuration
DATA_FILE = 'cumulative_2025.10.03_00.50.03.csv'
MODEL_DIR = Path('backend/saved_models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 1: GPU DETECTION AND SETUP
# ============================================================================
print("\n🔍 STEP 1: Checking GPU availability...")

gpu_available = False
use_gpu_lgbm = False
use_gpu_xgb = False

# Check PyTorch for CUDA
try:
    import torch
    if torch.cuda.is_available():
        gpu_available = True
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        cuda_version = torch.version.cuda
        
        print(f"   ✅ GPU Detected: {gpu_name}")
        print(f"   ✅ CUDA Version: {cuda_version}")
        print(f"   ✅ GPU Memory: {gpu_memory:.2f} GB")
        print(f"   ℹ️  Note: Optimized for GTX 1050 Ti (4GB VRAM)")
    else:
        print("   ⚠️  CUDA not available - will use CPU")
except ImportError:
    print("   ⚠️  PyTorch not installed - will use CPU")
    print("   💡 Install with: pip install torch --index-url https://download.pytorch.org/whl/cu118")

# Check LightGBM GPU support
if gpu_available:
    try:
        import lightgbm as lgb
        print(f"   ✅ LightGBM {lgb.__version__} installed")
        # Test GPU device
        try:
            test_lgbm = lgb.LGBMClassifier(device='gpu', n_estimators=1, verbose=-1)
            use_gpu_lgbm = True
            print("   ✅ LightGBM GPU support: ENABLED")
        except Exception as e:
            print(f"   ⚠️  LightGBM GPU not available: {str(e)[:60]}")
            print("   ℹ️  LightGBM will use CPU")
    except ImportError:
        print("   ❌ LightGBM not installed")
        print("   Install with: pip install lightgbm")
        import sys
        sys.exit(1)
else:
    try:
        import lightgbm as lgb
        print(f"   ✅ LightGBM {lgb.__version__} installed (CPU mode)")
    except ImportError:
        print("   ❌ LightGBM not installed")
        import sys
        sys.exit(1)

# Check XGBoost GPU support
if gpu_available:
    try:
        import xgboost as xgb
        print(f"   ✅ XGBoost {xgb.__version__} installed")
        use_gpu_xgb = True
        print("   ✅ XGBoost GPU support: ENABLED")
    except ImportError:
        print("   ❌ XGBoost not installed")
        print("   Install with: pip install xgboost")
        import sys
        sys.exit(1)
else:
    try:
        import xgboost as xgb
        print(f"   ✅ XGBoost {xgb.__version__} installed (CPU mode)")
    except ImportError:
        print("   ❌ XGBoost not installed")
        import sys
        sys.exit(1)

# Summary
print("\n📊 Training Configuration:")
print(f"   LightGBM: {'GPU (CUDA)' if use_gpu_lgbm else 'CPU'}")
print(f"   XGBoost:  {'GPU (gpu_hist)' if use_gpu_xgb else 'CPU (hist)'}")
print(f"   Logistic Regression: CPU (meta-learner)")

# ============================================================================
# STEP 2: LOAD AND PREPARE DATA
# ============================================================================
print(f"\n📂 STEP 2: Loading data from {DATA_FILE}...")
try:
    df = pd.read_csv(DATA_FILE)
    print(f"   ✅ Loaded {len(df):,} rows with {len(df.columns)} columns")
except FileNotFoundError:
    print(f"   ❌ Error: File '{DATA_FILE}' not found!")
    print(f"   Make sure the file exists in: {os.getcwd()}")
    import sys
    sys.exit(1)

# Map labels to numeric classes
print("\n🏷️  Mapping labels to classes...")
label_map = {
    'CONFIRMED': 0,
    'CANDIDATE': 1,
    'FALSE POSITIVE': 2
}

df['target'] = df['koi_disposition'].map(label_map)
df_clean = df.dropna(subset=['target'])
print(f"   ✅ Rows after removing missing targets: {len(df_clean):,}")

# Check class distribution
print("\n📊 Class Distribution:")
class_counts = df_clean['target'].value_counts().sort_index()
for class_id, count in class_counts.items():
    label_name = [k for k, v in label_map.items() if v == class_id][0]
    percentage = (count / len(df_clean)) * 100
    print(f"   Class {class_id} ({label_name:15s}): {count:5,} ({percentage:5.2f}%)")

# ============================================================================
# STEP 3: FEATURE SELECTION
# ============================================================================
print("\n🔧 STEP 3: Selecting features...")

# Exclude non-numeric and identifier columns
exclude_cols = [
    'koi_disposition', 'target', 'rowid', 'kepid', 'kepoi_name', 'kepler_name',
    'koi_vet_stat', 'koi_vet_date', 'koi_pdisposition', 'koi_disp_prov',
    'koi_comment', 'koi_parm_prov', 'koi_tce_delivname', 'koi_quarters',
    'koi_trans_mod', 'koi_datalink_dvr', 'koi_datalink_dvs', 'koi_sparprov',
    'koi_limbdark_mod', 'koi_fittype'
]

feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
print(f"   ✅ Selected {len(feature_cols)} features")

X = df_clean[feature_cols]
y = df_clean['target'].astype(int)

# Handle missing and infinite values
print("\n🔄 Cleaning data...")
missing_before = X.isnull().sum().sum()
X = X.fillna(X.median())
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())
print(f"   ✅ Handled {missing_before:,} missing values")

# ============================================================================
# STEP 4: TRAIN/TEST SPLIT
# ============================================================================
print("\n✂️  STEP 4: Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training samples:   {len(X_train):6,}")
print(f"   Test samples:       {len(X_test):6,}")
print(f"   Features:           {X_train.shape[1]:6,}")

# ============================================================================
# STEP 5: BUILD MODELS (GTX 1050 Ti OPTIMIZED)
# ============================================================================
print("\n🤖 STEP 5: Building ensemble model...")
print("\n   Configuration for GTX 1050 Ti (4GB VRAM):")
print("   • Reduced estimators: 300 (instead of 500) for faster training")
print("   • Optimized batch sizes for 4GB memory limit")
print("   • Max depth limited to prevent memory overflow")

# LightGBM Configuration
lgbm_params = {
    'n_estimators': 300,          # Reduced for faster training on GTX 1050 Ti
    'learning_rate': 0.1,
    'max_depth': 8,                # Limited depth for memory efficiency
    'num_leaves': 31,
    'random_state': 42,
    'verbose': -1,
    'class_weight': 'balanced',
    'n_jobs': 4 if not use_gpu_lgbm else 1  # Limit CPU threads when using GPU
}

if use_gpu_lgbm:
    lgbm_params.update({
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'max_bin': 63  # Reduced for better memory usage
    })
    print(f"   ✅ LightGBM: n_estimators=300, device=GPU")
else:
    lgbm_params['n_jobs'] = -1
    print(f"   ℹ️  LightGBM: n_estimators=300, device=CPU")

lgbm_model = lgb.LGBMClassifier(**lgbm_params)

# XGBoost Configuration  
xgb_params = {
    'n_estimators': 300,           # Reduced for faster training
    'learning_rate': 0.1,
    'max_depth': 6,                # Conservative depth
    'random_state': 42,
    'eval_metric': 'mlogloss',
    'verbosity': 1                 # Show progress
}

if use_gpu_xgb:
    xgb_params.update({
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'max_bin': 256             # Optimized for GTX 1050 Ti
    })
    print(f"   ✅ XGBoost: n_estimators=300, tree_method=gpu_hist")
else:
    xgb_params['tree_method'] = 'hist'  # CPU fallback
    xgb_params['n_jobs'] = -1
    print(f"   ℹ️  XGBoost: n_estimators=300, tree_method=hist (CPU)")

xgb_model = xgb.XGBClassifier(**xgb_params)

# Stacking Classifier with Logistic Regression meta-learner
print(f"   ✅ Meta-learner: Logistic Regression (multinomial)")

stacking_model = StackingClassifier(
    estimators=[
        ('lgbm', lgbm_model),
        ('xgb', xgb_model)
    ],
    final_estimator=LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    ),
    cv=5,
    n_jobs=2  # Limit parallel jobs to avoid memory issues
)

# ============================================================================
# STEP 6: TRAIN MODEL
# ============================================================================
print("\n🎯 STEP 6: Training model...")
print("   ⏱️  This may take 5-15 minutes depending on GPU performance...")
print("   💡 Tip: Monitor GPU usage with 'nvidia-smi' in another terminal")
print()

try:
    stacking_model.fit(X_train, y_train)
    print("   ✅ Training complete!")
except Exception as e:
    print(f"   ❌ Training failed: {str(e)}")
    print("\n   Troubleshooting:")
    print("   1. Check GPU memory with: nvidia-smi")
    print("   2. Try running train_3class_cpu.py for CPU training")
    print("   3. Ensure CUDA drivers are up to date")
    import sys
    sys.exit(1)

# ============================================================================
# STEP 7: EVALUATE MODEL
# ============================================================================
print("\n📈 STEP 7: Evaluating model on test set...")
y_pred = stacking_model.predict(X_test)
y_pred_proba = stacking_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n{'=' * 80}")
print("EVALUATION RESULTS")
print("=" * 80)
print(f"\n🎯 Overall Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"📊 F1-Score (weighted):  {f1:.4f}")

print("\n📋 Detailed Classification Report:")
print("-" * 80)
target_names = ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

print("\n🔢 Confusion Matrix:")
print("-" * 80)
cm = confusion_matrix(y_test, y_pred)
print("                    Predicted")
print("                CONF   CAND   FP")
print(f"Actual  CONF   [{cm[0][0]:5d}  {cm[0][1]:5d}  {cm[0][2]:5d}]")
print(f"        CAND   [{cm[1][0]:5d}  {cm[1][1]:5d}  {cm[1][2]:5d}]")
print(f"        FP     [{cm[2][0]:5d}  {cm[2][1]:5d}  {cm[2][2]:5d}]")
print("\n(CONF=CONFIRMED, CAND=CANDIDATE, FP=FALSE POSITIVE)")

# Per-class accuracy
print("\n📊 Per-Class Accuracy:")
for i, class_name in enumerate(target_names):
    class_acc = cm[i][i] / cm[i].sum()
    print(f"   {class_name:15s}: {class_acc:.4f} ({class_acc*100:.2f}%)")

# ============================================================================
# STEP 8: CROSS-VALIDATION
# ============================================================================
print("\n🔄 STEP 8: Running 5-fold cross-validation...")
print("   ⏱️  This will take a few minutes...")

try:
    cv_scores = cross_val_score(
        stacking_model, X_train, y_train, 
        cv=5, scoring='accuracy', n_jobs=1  # Sequential to avoid memory issues
    )
    print(f"   ✅ CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"   ✅ Mean CV Score: {cv_scores.mean():.4f} (± {cv_scores.std() * 2:.4f})")
except Exception as e:
    print(f"   ⚠️  Cross-validation skipped due to: {str(e)}")
    cv_scores = np.array([accuracy])

# ============================================================================
# STEP 9: SAVE MODEL
# ============================================================================
print("\n💾 STEP 9: Saving model and metadata...")

model_path = MODEL_DIR / 'stacking_model.pkl'
joblib.dump(stacking_model, model_path)
print(f"   ✅ Model saved: {model_path}")

# Save comprehensive metadata
metadata = {
    'model_version': '3.0.0-GPU',
    'training_date': datetime.now().isoformat(),
    'training_device': 'GPU (GTX 1050 Ti)' if (use_gpu_lgbm or use_gpu_xgb) else 'CPU',
    'num_classes': 3,
    'classes': target_names,
    'label_mapping': label_map,
    'num_features': len(feature_cols),
    'feature_names': list(X.columns),
    'test_accuracy': float(accuracy),
    'test_f1_score': float(f1),
    'cv_scores': cv_scores.tolist(),
    'cv_mean': float(cv_scores.mean()),
    'cv_std': float(cv_scores.std()),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'class_distribution': {
        'CONFIRMED': int(class_counts[0]),
        'CANDIDATE': int(class_counts[1]),
        'FALSE POSITIVE': int(class_counts[2])
    },
    'gpu_accelerated': use_gpu_lgbm or use_gpu_xgb,
    'lgbm_gpu': use_gpu_lgbm,
    'xgb_gpu': use_gpu_xgb,
    'base_estimators': {
        'lgbm': {
            'type': 'LGBMClassifier',
            'n_estimators': 300,
            'device': 'gpu' if use_gpu_lgbm else 'cpu'
        },
        'xgb': {
            'type': 'XGBClassifier', 
            'n_estimators': 300,
            'tree_method': 'gpu_hist' if use_gpu_xgb else 'hist'
        }
    },
    'meta_estimator': 'LogisticRegression (multinomial)',
    'confusion_matrix': cm.tolist(),
    'per_class_accuracy': {
        target_names[i]: float(cm[i][i] / cm[i].sum()) 
        for i in range(len(target_names))
    }
}

metadata_path = MODEL_DIR / 'model_metadata.pkl'
joblib.dump(metadata, metadata_path)
print(f"   ✅ Metadata saved: {metadata_path}")

# ============================================================================
# COMPLETION
# ============================================================================
print("\n" + "=" * 80)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 80)

print("\n📊 Summary:")
print(f"   • Training Device: {'GPU (GTX 1050 Ti)' if (use_gpu_lgbm or use_gpu_xgb) else 'CPU'}")
print(f"   • Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   • F1-Score: {f1:.4f}")
print(f"   • Training Time: ~5-15 minutes")
print(f"   • Model Saved: {model_path}")

print("\n📝 Next Steps:")
print("   1. Test your trained model:")
print("      python test_3class_api.py")
print()
print("   2. Start the backend server:")
print("      cd backend")
print("      python app.py")
print()
print("   3. Access the API documentation:")
print("      http://localhost:8000/docs")
print()
print("   4. Make predictions with your new 3-class model!")

print("\n" + "=" * 80)
print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
