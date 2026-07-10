"""
utils/preprocessing.py
-----------------------
Reusable, modular preprocessing functions shared by train_model.py and app.py.

Contains:
    - clean_data()          : missing values + duplicates
    - engineer_features()   : BMI category, Age group, simple Risk Score
    - encode_categoricals()  : LabelEncoder for engineered category columns
    - balance_classes()      : SMOTE (falls back to random oversampling if
                                imbalanced-learn isn't installed)
    - split_and_scale()      : train/test split + StandardScaler
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# imbalanced-learn is an optional dependency. If it isn't installed we fall
# back to a simple manual random oversampler so the pipeline still runs.
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False


# ------------------------------------------------------------------
# 1. Cleaning
# ------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and duplicate rows.

    - Numeric columns  -> filled with column median
    - Categorical cols -> filled with column mode
    - Duplicate rows   -> dropped
    """
    df = df.copy()

    # Fill missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    # Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"Removed {removed} duplicate rows")

    return df


# ------------------------------------------------------------------
# 2. Feature Engineering
# ------------------------------------------------------------------
def bmi_category(bmi: float) -> str:
    """Standard WHO BMI categories."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def age_group(age: float) -> str:
    """Simple age bands used commonly in clinical risk scoring."""
    if age < 30:
        return "Young"
    elif age < 45:
        return "Adult"
    elif age < 60:
        return "MiddleAged"
    else:
        return "Senior"


def engineer_features(df: pd.DataFrame, age_col: str = None, bmi_col: str = None,
                       risk_stats: dict = None):
    """Add BMI category, Age group, and a simple composite Risk Score.

    Works generically across datasets: pass the actual column names that
    represent Age / BMI in that dataset (they differ per dataset).
    If a column doesn't exist in this dataset, that feature is skipped.

    risk_stats: optional dict {"cols": [...], "mean": {...}, "std": {...}}.
        - During TRAINING, leave this as None: the mean/std are computed from
          the training data and returned so they can be saved.
        - During PREDICTION (a single new patient row), pass in the stats
          saved from training so the new row is normalized against the same
          reference distribution instead of against itself (which would
          always give std=0 for a single row).

    Returns: (df_with_new_features, risk_stats_used)
    """
    df = df.copy()

    if age_col and age_col in df.columns:
        df["AgeGroup"] = df[age_col].apply(age_group)

    if bmi_col and bmi_col in df.columns:
        df["BMICategory"] = df[bmi_col].apply(bmi_category)

    # Composite risk score = mean of normalized numeric risk-relevant columns.
    # This mimics a simple clinical "risk index" feature engineers often add.
    if risk_stats is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ("target", "Outcome")]
        mean = df[numeric_cols].mean()
        std = df[numeric_cols].std() + 1e-9
        risk_stats = {"cols": list(numeric_cols), "mean": mean.to_dict(), "std": std.to_dict()}
    else:
        numeric_cols = risk_stats["cols"]

    if len(numeric_cols) > 0:
        mean_series = pd.Series(risk_stats["mean"])
        std_series = pd.Series(risk_stats["std"])
        normalized = (df[numeric_cols] - mean_series) / std_series
        df["RiskScore"] = normalized.mean(axis=1)

    return df, risk_stats


# ------------------------------------------------------------------
# 3. Encoding
# ------------------------------------------------------------------
def encode_categoricals(df: pd.DataFrame, categorical_cols):
    """Label-encode categorical columns; returns (df, {col: fitted_encoder})."""
    df = df.copy()
    encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


# ------------------------------------------------------------------
# 4. Class balancing
# ------------------------------------------------------------------
def balance_classes(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """Balance classes using SMOTE. Falls back to simple random oversampling
    of the minority class if imbalanced-learn is not installed."""
    if HAS_SMOTE:
        sm = SMOTE(random_state=random_state)
        X_res, y_res = sm.fit_resample(X, y)
        return X_res, y_res

    # ---- Fallback: naive random oversampling ----
    rng = np.random.RandomState(random_state)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_parts, y_parts = [X], [y]
    for cls, count in zip(classes, counts):
        if count < max_count:
            idx = np.where(y == cls)[0]
            extra_idx = rng.choice(idx, size=max_count - count, replace=True)
            X_parts.append(X[extra_idx])
            y_parts.append(y[extra_idx])

    X_res = np.vstack(X_parts)
    y_res = np.concatenate(y_parts)
    return X_res, y_res


# ------------------------------------------------------------------
# 5. Train/test split + scaling
# ------------------------------------------------------------------
def split_and_scale(X, y, test_size: float = 0.2, random_state: int = 42):
    """Split into train/test then fit a StandardScaler on the TRAIN set only
    (to avoid data leakage) and apply it to both sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
