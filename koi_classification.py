"""
NASA KOI Classification using Ensemble Methods
Implements 5 ensemble algorithms with hyperparameter tuning and k-fold cross-validation
"""

import pandas as pd
import numpy as np
import time
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier,
                               BaggingClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
import joblib

warnings.filterwarnings('ignore')

# ==================== 1. DATA PREPARATION ====================
print("=" * 80)
print("STEP 1: DATA PREPARATION")
print("=" * 80)

# Load dataset
print("\nLoading dataset...")
df = pd.read_csv('cumulative_2025.10.03_00.50.03.csv')
print(f"Original dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# Display initial info
print(f"\nkoi_disposition value counts:\n{df['koi_disposition'].value_counts()}")

# Filter rows: keep only 'confirmed' or 'candidate'
print("\nFiltering rows: keeping only 'confirmed' and 'candidate'...")
df = df[df['koi_disposition'].isin(['CONFIRMED', 'CANDIDATE'])]
print(f"Dataset shape after filtering: {df.shape}")

# Identify columns to remove
columns_to_remove = [
    'rowid', 'kepid', 'kepoi_name', 'kepler_name',
    'koi_pdisposition', 'koi_score',
    'koi_teq_err1', 'koi_teq_err2'
]

# Remove columns that exist in the dataframe
existing_cols_to_remove = [col for col in columns_to_remove if col in df.columns]
df = df.drop(columns=existing_cols_to_remove, errors='ignore')
print(f"\nRemoved columns: {existing_cols_to_remove}")
print(f"Dataset shape after removing columns: {df.shape}")

# Encode target variable
print("\nEncoding target variable...")
print("  candidate = 1")
print("  confirmed = 0")
df['koi_disposition'] = df['koi_disposition'].map({'CANDIDATE': 1, 'CONFIRMED': 0})
print(f"Target distribution:\n{df['koi_disposition'].value_counts()}")

# Separate features and target
y = df['koi_disposition']
X = df.drop('koi_disposition', axis=1)

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Handle missing values FIRST (replace with mean for numeric columns)
print("\nHandling missing values in numeric columns...")
print(f"Missing values before:\n{X.isnull().sum().sum()} total missing values")
numeric_cols = X.select_dtypes(include=[np.number]).columns
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
print(f"Missing values after numeric imputation:\n{X.isnull().sum().sum()} total missing values")

# Handle categorical columns (convert to dummy variables)
print("\nHandling categorical columns...")
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns found: {categorical_cols}")

if categorical_cols:
    # Fill NaN in categorical columns with 'Unknown' before creating dummies
    for col in categorical_cols:
        X[col] = X[col].fillna('Unknown')
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    print(f"Shape after creating dummy variables: {X.shape}")

# Final check for any remaining NaN values
print("\nFinal missing values check...")
print(f"Remaining missing values: {X.isnull().sum().sum()}")
if X.isnull().sum().sum() > 0:
    print("Filling any remaining NaN values with 0...")
    X = X.fillna(0)
    print(f"Missing values after final fill: {X.isnull().sum().sum()}")

# Train/test split (70/30)
print("\nSplitting dataset (70% train, 30% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Scale features
print("\nScaling features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling completed.")

# Save scaler for later use
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved to scaler.pkl")


# ==================== 2. HELPER FUNCTIONS ====================
def calculate_specificity(y_true, y_pred):
    """Calculate specificity (True Negative Rate)"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Evaluate model and return comprehensive metrics"""
    print(f"\nEvaluating {model_name}...")

    # Training
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Prediction
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time

    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)  # Sensitivity
    specificity = calculate_specificity(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Sensitivity': recall,
        'Specificity': specificity,
        'F1 Score': f1,
        'Train Time (s)': train_time,
        'Inference Time (s)': inference_time,
        'Confusion Matrix': cm
    }

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Sensitivity: {recall:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Train Time: {train_time:.2f}s")
    print(f"  Confusion Matrix:\n{cm}")

    return results, model


# ==================== 3. MODEL TRAINING ====================
print("\n" + "=" * 80)
print("STEP 2: MODEL TRAINING AND HYPERPARAMETER TUNING")
print("=" * 80)

results_list = []

# -------------------- 3.1 AdaBoostClassifier --------------------
print("\n" + "-" * 80)
print("3.1 AdaBoostClassifier")
print("-" * 80)

# Baseline
ada_base = AdaBoostClassifier(random_state=42, algorithm='SAMME')
ada_base_results, ada_base_model = evaluate_model(
    ada_base, X_train_scaled, y_train, X_test_scaled, y_test,
    'AdaBoost (Baseline)'
)
results_list.append(ada_base_results)

# Hyperparameter tuning
print("\nTuning AdaBoostClassifier...")
ada_param_grid = {
    'n_estimators': [50, 200, 500, 1000],
    'learning_rate': [0.01, 0.1, 1.0]
}
ada_grid = GridSearchCV(
    AdaBoostClassifier(random_state=42, algorithm='SAMME'),
    ada_param_grid,
    cv=10,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
ada_grid.fit(X_train_scaled, y_train)
print(f"Best parameters: {ada_grid.best_params_}")
print(f"Best CV score: {ada_grid.best_score_:.4f}")

ada_tuned_results, ada_tuned_model = evaluate_model(
    ada_grid.best_estimator_, X_train_scaled, y_train, X_test_scaled, y_test,
    'AdaBoost (Tuned)'
)
results_list.append(ada_tuned_results)


# -------------------- 3.2 RandomForestClassifier --------------------
print("\n" + "-" * 80)
print("3.2 RandomForestClassifier")
print("-" * 80)

# Baseline
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_base_results, rf_base_model = evaluate_model(
    rf_base, X_train_scaled, y_train, X_test_scaled, y_test,
    'RandomForest (Baseline)'
)
results_list.append(rf_base_results)

# Hyperparameter tuning
print("\nTuning RandomForestClassifier...")
rf_param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_features': ['sqrt', 'log2'],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20]
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_grid,
    cv=10,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(X_train_scaled, y_train)
print(f"Best parameters: {rf_grid.best_params_}")
print(f"Best CV score: {rf_grid.best_score_:.4f}")

rf_tuned_results, rf_tuned_model = evaluate_model(
    rf_grid.best_estimator_, X_train_scaled, y_train, X_test_scaled, y_test,
    'RandomForest (Tuned)'
)
results_list.append(rf_tuned_results)


# -------------------- 3.3 BaggingClassifier --------------------
print("\n" + "-" * 80)
print("3.3 BaggingClassifier (Random Subspace)")
print("-" * 80)

# Baseline
bag_base = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    random_state=42,
    n_jobs=-1
)
bag_base_results, bag_base_model = evaluate_model(
    bag_base, X_train_scaled, y_train, X_test_scaled, y_test,
    'Bagging (Baseline)'
)
results_list.append(bag_base_results)

# Hyperparameter tuning
print("\nTuning BaggingClassifier...")
bag_param_grid = {
    'n_estimators': [50, 200, 500],
    'max_samples': [0.5, 0.7, 1.0],
    'max_features': [0.5, 0.7, 1.0]
}
bag_grid = GridSearchCV(
    BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=42, n_jobs=-1),
    bag_param_grid,
    cv=10,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
bag_grid.fit(X_train_scaled, y_train)
print(f"Best parameters: {bag_grid.best_params_}")
print(f"Best CV score: {bag_grid.best_score_:.4f}")

bag_tuned_results, bag_tuned_model = evaluate_model(
    bag_grid.best_estimator_, X_train_scaled, y_train, X_test_scaled, y_test,
    'Bagging (Tuned)'
)
results_list.append(bag_tuned_results)


# -------------------- 3.4 ExtraTreesClassifier --------------------
print("\n" + "-" * 80)
print("3.4 ExtraTreesClassifier")
print("-" * 80)

# Baseline
et_base = ExtraTreesClassifier(random_state=42, n_jobs=-1)
et_base_results, et_base_model = evaluate_model(
    et_base, X_train_scaled, y_train, X_test_scaled, y_test,
    'ExtraTrees (Baseline)'
)
results_list.append(et_base_results)

# Hyperparameter tuning
print("\nTuning ExtraTreesClassifier...")
et_param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_features': ['sqrt', 'log2'],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20]
}
et_grid = GridSearchCV(
    ExtraTreesClassifier(random_state=42, n_jobs=-1),
    et_param_grid,
    cv=10,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
et_grid.fit(X_train_scaled, y_train)
print(f"Best parameters: {et_grid.best_params_}")
print(f"Best CV score: {et_grid.best_score_:.4f}")

et_tuned_results, et_tuned_model = evaluate_model(
    et_grid.best_estimator_, X_train_scaled, y_train, X_test_scaled, y_test,
    'ExtraTrees (Tuned)'
)
results_list.append(et_tuned_results)


# -------------------- 3.5 StackingClassifier --------------------
print("\n" + "-" * 80)
print("3.5 StackingClassifier")
print("-" * 80)

# Test multiple stacking combinations
stacking_configs = [
    {
        'name': 'Stacking (RF+GB)',
        'estimators': [
            ('rf', RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=500, random_state=42))
        ]
    },
    {
        'name': 'Stacking (RF+GB+ET)',
        'estimators': [
            ('rf', RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=500, random_state=42)),
            ('et', ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1))
        ]
    }
]

best_stacking_model = None
best_stacking_score = 0

for config in stacking_configs:
    print(f"\nTesting {config['name']}...")
    stacking = StackingClassifier(
        estimators=config['estimators'],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        n_jobs=-1
    )

    stacking_results, stacking_model = evaluate_model(
        stacking, X_train_scaled, y_train, X_test_scaled, y_test,
        config['name']
    )
    results_list.append(stacking_results)

    if stacking_results['Accuracy'] > best_stacking_score:
        best_stacking_score = stacking_results['Accuracy']
        best_stacking_model = stacking_model


# ==================== 4. RESULTS COMPARISON ====================
print("\n" + "=" * 80)
print("STEP 3: RESULTS COMPARISON")
print("=" * 80)

# Create results DataFrame
results_df = pd.DataFrame(results_list)
results_df = results_df[['Model', 'Accuracy', 'Precision', 'Sensitivity',
                          'Specificity', 'F1 Score', 'Train Time (s)',
                          'Inference Time (s)']]

# Sort by accuracy
results_df = results_df.sort_values('Accuracy', ascending=False)

print("\n" + "=" * 80)
print("FINAL RESULTS TABLE")
print("=" * 80)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('model_comparison_results.csv', index=False)
print("\nResults saved to model_comparison_results.csv")

# Identify best model
best_model_row = results_df.iloc[0]
print(f"\n{'=' * 80}")
print("BEST MODEL")
print("=" * 80)
print(f"Model: {best_model_row['Model']}")
print(f"Accuracy: {best_model_row['Accuracy']:.4f}")
print(f"Precision: {best_model_row['Precision']:.4f}")
print(f"Sensitivity: {best_model_row['Sensitivity']:.4f}")
print(f"Specificity: {best_model_row['Specificity']:.4f}")
print(f"F1 Score: {best_model_row['F1 Score']:.4f}")

# Save best model (using the tuned RF as an example, adjust based on actual best)
if 'Stacking' in best_model_row['Model']:
    joblib.dump(best_stacking_model, 'best_model.pkl')
elif 'RandomForest' in best_model_row['Model'] and 'Tuned' in best_model_row['Model']:
    joblib.dump(rf_tuned_model, 'best_model.pkl')
elif 'ExtraTrees' in best_model_row['Model'] and 'Tuned' in best_model_row['Model']:
    joblib.dump(et_tuned_model, 'best_model.pkl')
elif 'AdaBoost' in best_model_row['Model'] and 'Tuned' in best_model_row['Model']:
    joblib.dump(ada_tuned_model, 'best_model.pkl')
elif 'Bagging' in best_model_row['Model'] and 'Tuned' in best_model_row['Model']:
    joblib.dump(bag_tuned_model, 'best_model.pkl')

print("\nBest model saved to best_model.pkl")
print("\n" + "=" * 80)
print("PIPELINE COMPLETED")
print("=" * 80)
