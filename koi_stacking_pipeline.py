
import pandas as pd
import numpy as np
import time
import warnings
from datetime import datetime
import joblib

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              make_scorer)

# Gradient boosting libraries
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')


# ==================== HELPER FUNCTIONS ====================

def calculate_specificity(y_true, y_pred):
    """
    Calculate specificity (True Negative Rate)
    Specificity = TN / (TN + FP)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)


def print_metrics(y_true, y_pred, dataset_name="Test"):
    """Print comprehensive evaluation metrics"""
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    specificity = calculate_specificity(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n{dataset_name} Set Performance:")
    print(f"  Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Sensitivity: {recall:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  F1 Score:    {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    {cm}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'sensitivity': recall,
        'specificity': specificity,
        'f1_score': f1,
        'confusion_matrix': cm
    }


# ==================== 1. DATA LOADING & PREPROCESSING ====================

print_section("STEP 1: DATA LOADING & PREPROCESSING")

# Load dataset
print("\n[1.1] Loading dataset...")
df = pd.read_csv('cumulative_2025.10.03_00.50.03.csv')
print(f"  Original dataset shape: {df.shape}")

# Display target distribution
print(f"\n[1.2] Target variable distribution:")
print(f"{df['koi_disposition'].value_counts()}")

# Filter rows: keep only 'CONFIRMED' or 'CANDIDATE'
print("\n[1.3] Filtering rows (keeping CONFIRMED and CANDIDATE)...")
df = df[df['koi_disposition'].isin(['CONFIRMED', 'CANDIDATE'])]
print(f"  Dataset shape after filtering: {df.shape}")

# Remove identifier and redundant columns
columns_to_remove = [
    'rowid', 'kepid', 'kepoi_name', 'kepler_name',
    'koi_pdisposition', 'koi_score',
    'koi_teq_err1', 'koi_teq_err2'
]

existing_cols_to_remove = [col for col in columns_to_remove if col in df.columns]
df = df.drop(columns=existing_cols_to_remove, errors='ignore')
print(f"\n[1.4] Removed columns: {existing_cols_to_remove}")
print(f"  Dataset shape after column removal: {df.shape}")

# Encode target variable (CANDIDATE=1, CONFIRMED=0)
print("\n[1.5] Encoding target variable...")
print("  CANDIDATE = 1 (positive class)")
print("  CONFIRMED = 0 (negative class)")
df['koi_disposition'] = df['koi_disposition'].map({'CANDIDATE': 1, 'CONFIRMED': 0})

# Separate features and target
y = df['koi_disposition']
X = df.drop('koi_disposition', axis=1)
print(f"\n  Features shape: {X.shape}")
print(f"  Target shape:   {y.shape}")
print(f"  Target distribution:\n{y.value_counts()}")

# Handle missing values in numeric columns
print("\n[1.6] Handling missing values...")
print(f"  Missing values before: {X.isnull().sum().sum()} total")
numeric_cols = X.select_dtypes(include=[np.number]).columns
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
print(f"  Missing values after numeric imputation: {X.isnull().sum().sum()} total")

# Handle categorical columns with one-hot encoding
print("\n[1.7] Handling categorical columns...")
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"  Categorical columns found: {categorical_cols}")

if categorical_cols:
    # Fill NaN in categorical columns with 'Unknown'
    for col in categorical_cols:
        X[col] = X[col].fillna('Unknown')
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    print(f"  Shape after one-hot encoding: {X.shape}")

# Final NaN check and cleanup
print("\n[1.8] Final missing values check...")
remaining_nans = X.isnull().sum().sum()
print(f"  Remaining missing values: {remaining_nans}")
if remaining_nans > 0:
    X = X.fillna(0)
    print(f"  Filled remaining NaN values with 0")

# Train/test split (70/30 stratified)
print("\n[1.9] Splitting dataset (70% train, 30% test, stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"  Training set: {X_train.shape}")
print(f"  Test set:     {X_test.shape}")

# Feature scaling using StandardScaler
print("\n[1.10] Scaling features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("  ✓ Feature scaling completed")

# Save scaler for production use
scaler_path = 'scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"  ✓ Scaler saved to {scaler_path}")


# ==================== 2. MODEL DEFINITION ====================

print_section("STEP 2: OPTIMAL STACKING CLASSIFIER CONFIGURATION")

print("\n[2.1] Defining Stacking Classifier with optimal hyperparameters...")

# Base estimators (from research paper)
base_estimators = [
    ('lgbm', LGBMClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )),
    ('gb', GradientBoostingClassifier(
        n_estimators=1600,
        learning_rate=0.1,
        random_state=42
    ))
]

# Meta-learner
meta_learner = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Create Stacking Classifier with 5-fold CV
stacking_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1
)

print("\n  Base Estimators:")
print("    1. LightGBM Classifier (n_estimators=500)")
print("    2. Gradient Boosting Classifier (n_estimators=1600, learning_rate=0.1)")
print("\n  Meta-learner:")
print("    Logistic Regression (max_iter=1000)")
print("\n  Cross-validation: 5-fold (internal stacking)")


# ==================== 3. MODEL TRAINING ====================

print_section("STEP 3: MODEL TRAINING ON FULL TRAINING SET")

print("\n[4.1] Training Stacking Classifier on complete training data...")
train_start = time.time()

stacking_model.fit(X_train_scaled, y_train)

train_time = time.time() - train_start
print(f"  ✓ Training completed in {train_time:.2f} seconds")


# ==================== 5. TEST SET EVALUATION ====================

print_section("STEP 5: TEST SET EVALUATION")

print("\n[5.1] Making predictions on test set...")
inference_start = time.time()

y_pred = stacking_model.predict(X_test_scaled)

inference_time = time.time() - inference_start
print(f"  ✓ Inference completed in {inference_time:.4f} seconds")

# Calculate and display metrics
test_metrics = print_metrics(y_test, y_pred, dataset_name="Test")

# Generate classification report
print("\n[5.2] Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['CONFIRMED', 'CANDIDATE']))


# ==================== 6. MODEL PERSISTENCE ====================

print_section("STEP 6: MODEL PERSISTENCE")

# Save the trained model
model_path = 'stacking_model.pkl'
joblib.dump(stacking_model, model_path)
print(f"\n[6.1] ✓ Stacking model saved to: {model_path}")
print(f"[6.2] ✓ Scaler already saved to: {scaler_path}")

# Model metadata
metadata = {
    'model_type': 'StackingClassifier',
    'base_estimators': ['LGBMClassifier', 'GradientBoostingClassifier'],
    'meta_learner': 'LogisticRegression',
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'num_features': X_train_scaled.shape[1],
    'test_accuracy': test_metrics['accuracy'],
    'test_f1_score': test_metrics['f1_score']
}

metadata_path = 'model_metadata.pkl'
joblib.dump(metadata, metadata_path)
print(f"[6.3] ✓ Model metadata saved to: {metadata_path}")


# ==================== 7. PRODUCTION PREDICTION FUNCTION ====================

print_section("STEP 7: PRODUCTION DEPLOYMENT EXAMPLE")

def predict_koi_disposition(features_df, model_path='stacking_model.pkl', scaler_path='scaler.pkl'):
    """
    Production prediction function for KOI classification

    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame containing the same features used in training
        (after preprocessing and one-hot encoding)
    model_path : str
        Path to saved stacking model (.pkl)
    scaler_path : str
        Path to saved scaler (.pkl)

    Returns:
    --------
    predictions : np.ndarray
        Array of predictions (1=CANDIDATE, 0=CONFIRMED)
    probabilities : np.ndarray
        Array of prediction probabilities for each class
    """
    # Load artifacts
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Scale features
    features_scaled = scaler.transform(features_df)

    # Make predictions
    predictions = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)

    return predictions, probabilities


# Example usage on test set
print("\n[7.1] Example prediction function usage:")
print("  Loading saved model and scaler...")

# Take a small sample from test set
sample_indices = np.random.choice(len(X_test), size=5, replace=False)
X_sample = X_test.iloc[sample_indices]
y_sample = y_test.iloc[sample_indices]

# Make predictions using production function
predictions, probabilities = predict_koi_disposition(X_sample, model_path, scaler_path)

print(f"\n  Sample predictions (first 5 test samples):")
for i in range(len(predictions)):
    true_label = 'CANDIDATE' if y_sample.iloc[i] == 1 else 'CONFIRMED'
    pred_label = 'CANDIDATE' if predictions[i] == 1 else 'CONFIRMED'
    confidence = probabilities[i][predictions[i]]
    print(f"    Sample {i+1}: True={true_label:10s} | Pred={pred_label:10s} | Confidence={confidence:.4f}")


# ==================== 8. FINAL SUMMARY ====================

print_section("STEP 8: FINAL SUMMARY & COMPARISON")

print(f"\n📊 Model Performance Summary:")
print(f"  {'='*70}")
print(f"  Test Set Accuracy:        {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
print(f"  Test Set F1 Score:        {test_metrics['f1_score']:.4f}")
print(f"  {'='*70}")

print(f"\n📈 Comparison to Research Paper Benchmark:")
benchmark_accuracy = 0.8308  # 83.08% from paper
accuracy_diff = test_metrics['accuracy'] - benchmark_accuracy
print(f"  Paper Benchmark:          {benchmark_accuracy:.4f} (83.08%)")
print(f"  Our Test Accuracy:        {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
if accuracy_diff >= 0:
    print(f"  Difference:               +{accuracy_diff:.4f} (+{accuracy_diff*100:.2f}%) ✓ IMPROVED")
else:
    print(f"  Difference:               {accuracy_diff:.4f} ({accuracy_diff*100:.2f}%)")

print(f"\n💾 Saved Artifacts:")
print(f"  1. {model_path} - Trained Stacking Classifier")
print(f"  2. {scaler_path} - Feature Scaler")
print(f"  3. {metadata_path} - Model Metadata")

print(f"\n⏱️  Performance Metrics:")
print(f"  Training Time:            {train_time:.2f} seconds")
print(f"  Inference Time (test):    {inference_time:.4f} seconds")
print(f"  Inference per sample:     {inference_time/len(X_test)*1000:.4f} ms")

print(f"\n🔧 Model Configuration:")
print(f"  Base Estimator 1:         LGBMClassifier(n_estimators=500)")
print(f"  Base Estimator 2:         GradientBoostingClassifier(n_estimators=1600, lr=0.1)")
print(f"  Meta-learner:             LogisticRegression(max_iter=1000)")
print(f"  Internal CV:              5-fold stratified")
print(f"  Training samples:         {len(X_train)}")
print(f"  Test samples:             {len(X_test)}")
print(f"  Number of features:       {X_train_scaled.shape[1]}")

print("\n" + "=" * 80)
print("✅ PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 80)
