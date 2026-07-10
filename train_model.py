

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

from utils.preprocessing import (
    clean_data, engineer_features, encode_categoricals,
    balance_classes, split_and_scale
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)


# ==========================================================================
# STEP 1: EDA HELPERS
# ==========================================================================
def run_eda(df: pd.DataFrame, target_col: str, disease_name: str):
    """Print dataset info + save EDA plots (class distribution, correlation
    heatmap) to the reports/ folder."""
    print(f"\n{'='*60}\nEDA REPORT: {disease_name}\n{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nStatistical summary:\n{df.describe().T}")
    print(f"\nClass distribution:\n{df[target_col].value_counts()}")

    # Class distribution plot
    plt.figure(figsize=(5, 4))
    sns.countplot(x=target_col, data=df, palette="Set2")
    plt.title(f"{disease_name} - Class Distribution")
    plt.xlabel("Target (0 = No Disease, 1 = Disease)")
    plt.tight_layout()
    plt.savefig(f"reports/{disease_name}_class_distribution.png", dpi=120)
    plt.close()

    # Correlation heatmap (numeric columns only)
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, annot=numeric_df.shape[1] <= 15)
    plt.title(f"{disease_name} - Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"reports/{disease_name}_correlation_heatmap.png", dpi=120)
    plt.close()


def plot_feature_importance(model, feature_names, disease_name, model_name):
    """Save a feature importance bar chart if the model supports it."""
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])

    if importances is None:
        return

    order = np.argsort(importances)[::-1][:15]  # top 15
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances[order], y=np.array(feature_names)[order], palette="viridis")
    plt.title(f"{disease_name} - Feature Importance ({model_name})")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"reports/{disease_name}_feature_importance.png", dpi=120)
    plt.close()


def plot_roc_curve(y_test, probs_dict, disease_name):
    """probs_dict: {model_name: predicted_probabilities}"""
    plt.figure(figsize=(6, 5))
    for name, probs in probs_dict.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{disease_name} - ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"reports/{disease_name}_roc_curve.png", dpi=120)
    plt.close()


def plot_confusion_matrix(cm, disease_name, model_name):
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.title(f"{disease_name} - Confusion Matrix ({model_name})")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"reports/{disease_name}_confusion_matrix.png", dpi=120)
    plt.close()


# ==========================================================================
# STEP 2: MODEL TRAINING + COMPARISON
# ==========================================================================
def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, kernel="rbf", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, use_label_encoder=False,
            eval_metric="logloss", random_state=42
        )
    else:
        print("[WARNING] xgboost not installed -- skipping XGBoost model. "
              "Install with `pip install xgboost` to include it.")
    return models


def train_and_compare(X_train, X_test, y_train, y_test, feature_names, disease_name):
    """Train all candidate models, evaluate each, save comparison table +
    plots, and return the best model (highest F1 score)."""
    models = get_models()
    results = []
    probs_dict = {}
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs)

        print(f"\n--- {name} ({disease_name}) ---")
        print(f"Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  "
              f"F1={f1:.3f}  ROC-AUC={auc:.3f}")
        print(classification_report(y_test, preds, target_names=["No Disease", "Disease"]))

        cm = confusion_matrix(y_test, preds)
        results.append({
            "Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1, "ROC_AUC": auc
        })
        probs_dict[name] = probs
        fitted_models[name] = model

        # Save confusion matrix + feature importance for this model
        plot_confusion_matrix(cm, disease_name, name)

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    print(f"\n=== MODEL COMPARISON: {disease_name} ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(f"reports/{disease_name}_model_comparison.csv", index=False)

    plot_roc_curve(y_test, probs_dict, disease_name)

    best_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    plot_feature_importance(best_model, feature_names, disease_name, best_name)

    print(f"\n>>> BEST MODEL for {disease_name}: {best_name} "
          f"(F1={results_df.iloc[0]['F1']:.3f}, AUC={results_df.iloc[0]['ROC_AUC']:.3f})")

    return best_model, best_name, results_df


# ==========================================================================
# STEP 3: FULL PIPELINE FOR ONE DISEASE
# ==========================================================================
def run_pipeline(disease_key, csv_path, target_col, age_col=None, bmi_col=None):
    disease_name = disease_key.replace("_", " ").title()
    print(f"\n\n{'#'*70}\n# PIPELINE START: {disease_name}\n{'#'*70}")

    df = pd.read_csv(csv_path)

    # ---- EDA ----
    run_eda(df, target_col, disease_name)

    # ---- Cleaning ----
    df = clean_data(df)

    # ---- Feature engineering ----
    df, risk_stats = engineer_features(df, age_col=age_col, bmi_col=bmi_col)

    # ---- Encode engineered categorical columns ----
    categorical_cols = [c for c in ["AgeGroup", "BMICategory"] if c in df.columns]
    df, encoders = encode_categoricals(df, categorical_cols)

    # ---- Separate features/target ----
    y = df[target_col].values
    X_df = df.drop(columns=[target_col])
    feature_names = list(X_df.columns)
    X = X_df.values

    # ---- Train/test split + scaling ----
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    # ---- Balance classes (SMOTE) on TRAINING data only ----
    X_train_bal, y_train_bal = balance_classes(X_train, y_train)
    print(f"\nClass balance before SMOTE: {np.bincount(y_train)}")
    print(f"Class balance after  SMOTE: {np.bincount(y_train_bal)}")

    # ---- Train & compare models ----
    best_model, best_name, results_df = train_and_compare(
        X_train_bal, X_test, y_train_bal, y_test, feature_names, disease_name
    )

    # ---- Save artifacts ----
    joblib.dump(best_model, f"models/{disease_key}_model.pkl")
    joblib.dump(scaler, f"models/{disease_key}_scaler.pkl")
    joblib.dump(encoders, f"models/{disease_key}_encoder.pkl")

    # Save metadata needed by the Flask app (feature order, medians for
    # any fields the web form does not collect directly, best model name)
    meta = {
        "feature_names": feature_names,
        "target_col": target_col,
        "age_col": age_col,
        "bmi_col": bmi_col,
        "categorical_cols": categorical_cols,
        "best_model_name": best_name,
        "feature_medians": {c: float(X_df[c].median()) for c in feature_names},
        "risk_stats": risk_stats,
        "metrics": results_df.iloc[0].to_dict(),
    }
    with open(f"models/{disease_key}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved: models/{disease_key}_model.pkl, "
          f"models/{disease_key}_scaler.pkl, models/{disease_key}_encoder.pkl, "
          f"models/{disease_key}_meta.json")

    return results_df


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == "__main__":
    all_results = {}

    all_results["heart"] = run_pipeline(
        "heart", "dataset/heart.csv", target_col="target",
        age_col="age", bmi_col=None
    )

    all_results["diabetes"] = run_pipeline(
        "diabetes", "dataset/diabetes.csv", target_col="Outcome",
        age_col="Age", bmi_col="BMI"
    )

    all_results["breast_cancer"] = run_pipeline(
        "breast_cancer", "dataset/breast_cancer.csv", target_col="target",
        age_col=None, bmi_col=None
    )

    print(f"\n\n{'='*70}\nALL MODELS TRAINED SUCCESSFULLY\n{'='*70}")
    for disease, results in all_results.items():
        best = results.iloc[0]
        print(f"{disease:15s} -> Best: {best['Model']:20s} "
              f"F1={best['F1']:.3f}  AUC={best['ROC_AUC']:.3f}")
