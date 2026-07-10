"""
generate_datasets.py
--------------------
Generates three CSV datasets with the SAME COLUMN SCHEMA as the well-known
UCI datasets used for disease prediction:

    1. heart.csv         -> UCI Heart Disease (Cleveland) schema
    2. diabetes.csv       -> Pima Indians Diabetes schema
    3. breast_cancer.csv  -> UCI/Wisconsin Breast Cancer (Diagnostic) schema

WHY THIS FILE EXISTS:
This sandbox has no internet access, so the real UCI files can't be
downloaded here. This script creates statistically realistic, rule-based
synthetic data (correct value ranges, correlations, and class balance)
so the WHOLE pipeline (EDA -> preprocessing -> training -> Flask app)
can be built and tested end-to-end right now.

TO USE REAL DATA INSTEAD:
Simply download the real datasets and save them with the exact same
filenames/columns in this `dataset/` folder:
    - heart.csv          (Kaggle: "Heart Disease UCI")
    - diabetes.csv        (Kaggle: "Pima Indians Diabetes Database")
    - breast_cancer.csv   (sklearn.datasets.load_breast_cancer, or
                            Kaggle: "Breast Cancer Wisconsin (Diagnostic)")
Nothing else in the project needs to change.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

np.random.seed(42)


def generate_heart_disease(n=1000):
    """UCI Heart Disease schema (Cleveland subset column names)."""
    age = np.random.randint(29, 78, n)
    sex = np.random.randint(0, 2, n)                       # 1 = male, 0 = female
    cp = np.random.randint(0, 4, n)                        # chest pain type (0-3)
    trestbps = np.random.normal(131, 17, n).clip(94, 200)   # resting blood pressure
    chol = np.random.normal(246, 51, n).clip(126, 564)      # serum cholesterol
    fbs = np.random.binomial(1, 0.15, n)                    # fasting blood sugar > 120
    restecg = np.random.randint(0, 3, n)                    # resting ECG results
    thalach = np.random.normal(150, 22, n).clip(71, 202)    # max heart rate achieved
    exang = np.random.binomial(1, 0.33, n)                  # exercise induced angina
    oldpeak = np.random.exponential(1.0, n).clip(0, 6.2)    # ST depression
    slope = np.random.randint(0, 3, n)                      # slope of peak ST segment
    ca = np.random.randint(0, 4, n)                         # number of major vessels
    thal = np.random.choice([0, 1, 2, 3], n, p=[0.05, 0.55, 0.35, 0.05])

    # Rule-based risk score -> probability of disease (keeps data realistic,
    # not just random noise) + a little randomness so it isn't trivially separable
    risk = (
        0.03 * (age - 50) + 1.2 * sex + 0.9 * cp
        + 0.02 * (trestbps - 130) + 0.01 * (chol - 240)
        + 0.6 * fbs + 0.4 * restecg
        - 0.02 * (thalach - 150) + 1.0 * exang
        + 0.5 * oldpeak + 0.4 * slope + 0.6 * ca + 0.3 * thal
    )
    prob = 1 / (1 + np.exp(-(risk - risk.mean()) / risk.std()))
    target = np.random.binomial(1, prob)

    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps.round(0),
        "chol": chol.round(0), "fbs": fbs, "restecg": restecg,
        "thalach": thalach.round(0), "exang": exang, "oldpeak": oldpeak.round(1),
        "slope": slope, "ca": ca, "thal": thal, "target": target
    })
    return df


def generate_diabetes(n=1000):
    """Pima Indians Diabetes schema."""
    pregnancies = np.random.poisson(3, n).clip(0, 17)
    glucose = np.random.normal(120, 32, n).clip(44, 199)
    blood_pressure = np.random.normal(69, 19, n).clip(24, 122)
    skin_thickness = np.random.normal(20, 16, n).clip(0, 99)
    insulin = np.random.exponential(80, n).clip(0, 846)
    bmi = np.random.normal(32, 7.9, n).clip(18, 67)
    dpf = np.random.exponential(0.47, n).clip(0.08, 2.42)
    age = np.random.randint(21, 81, n)

    risk = (
        0.02 * glucose + 0.05 * bmi + 0.01 * blood_pressure
        + 0.6 * dpf + 0.03 * age + 0.1 * pregnancies
    )
    prob = 1 / (1 + np.exp(-(risk - risk.mean()) / risk.std()))
    outcome = np.random.binomial(1, prob)

    df = pd.DataFrame({
        "Pregnancies": pregnancies, "Glucose": glucose.round(0),
        "BloodPressure": blood_pressure.round(0),
        "SkinThickness": skin_thickness.round(0), "Insulin": insulin.round(0),
        "BMI": bmi.round(1), "DiabetesPedigreeFunction": dpf.round(3),
        "Age": age, "Outcome": outcome
    })
    return df


def generate_breast_cancer():
    """Uses the REAL Wisconsin Breast Cancer dataset bundled with scikit-learn
    (no synthetic data needed here — it ships locally with sklearn)."""
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    # sklearn target: 0 = malignant, 1 = benign -> flip so 1 = disease (malignant)
    # to keep the "1 = at risk" convention consistent across all 3 datasets
    df["target"] = 1 - df["target"]
    return df


if __name__ == "__main__":
    heart_df = generate_heart_disease()
    diabetes_df = generate_diabetes()
    bc_df = generate_breast_cancer()

    heart_df.to_csv("dataset/heart.csv", index=False)
    diabetes_df.to_csv("dataset/diabetes.csv", index=False)
    bc_df.to_csv("dataset/breast_cancer.csv", index=False)

    print("Generated dataset/heart.csv         ->", heart_df.shape)
    print("Generated dataset/diabetes.csv        ->", diabetes_df.shape)
    print("Generated dataset/breast_cancer.csv   ->", bc_df.shape)
