


import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

from utils.preprocessing import engineer_features

app = Flask(__name__)

DISEASES = ["heart", "diabetes", "breast_cancer"]
DISEASE_LABELS = {
    "heart": "Heart Disease",
    "diabetes": "Diabetes",
    "breast_cancer": "Breast Cancer",
}


ARTIFACTS = {}
for disease in DISEASES:
    with open(f"models/{disease}_meta.json") as f:
        meta = json.load(f)
    ARTIFACTS[disease] = {
        "model": joblib.load(f"models/{disease}_model.pkl"),
        "scaler": joblib.load(f"models/{disease}_scaler.pkl"),
        "encoders": joblib.load(f"models/{disease}_encoder.pkl"),
        "meta": meta,
    }
    print(f"Loaded artifacts for '{disease}' "
          f"(best model: {meta['best_model_name']})")



FORM_FIELDS = {
    "heart": [
        {"name": "age", "label": "Age", "type": "number", "min": 1, "max": 120, "default": 50},
        {"name": "sex", "label": "Sex", "type": "select",
         "options": [("1", "Male"), ("0", "Female")]},
        {"name": "cp", "label": "Chest Pain Type", "type": "select",
         "options": [("0", "Typical Angina"), ("1", "Atypical Angina"),
                     ("2", "Non-anginal Pain"), ("3", "Asymptomatic")]},
        {"name": "trestbps", "label": "Resting Blood Pressure (mm Hg)", "type": "number", "min": 80, "max": 220, "default": 120},
        {"name": "chol", "label": "Serum Cholesterol (mg/dl)", "type": "number", "min": 100, "max": 600, "default": 200},
        {"name": "fbs", "label": "Fasting Blood Sugar > 120 mg/dl", "type": "select",
         "options": [("1", "Yes"), ("0", "No")]},
        {"name": "restecg", "label": "Resting ECG Result", "type": "select",
         "options": [("0", "Normal"), ("1", "ST-T Abnormality"), ("2", "Left Ventricular Hypertrophy")]},
        {"name": "thalach", "label": "Max Heart Rate Achieved", "type": "number", "min": 60, "max": 220, "default": 150},
        {"name": "exang", "label": "Exercise Induced Angina", "type": "select",
         "options": [("1", "Yes"), ("0", "No")]},
        {"name": "oldpeak", "label": "ST Depression (oldpeak)", "type": "number", "step": "0.1", "min": 0, "max": 7, "default": 1.0},
        {"name": "slope", "label": "Slope of Peak Exercise ST Segment", "type": "select",
         "options": [("0", "Upsloping"), ("1", "Flat"), ("2", "Downsloping")]},
        {"name": "ca", "label": "Number of Major Vessels (0-3)", "type": "number", "min": 0, "max": 3, "default": 0},
        {"name": "thal", "label": "Thalassemia", "type": "select",
         "options": [("1", "Normal"), ("2", "Fixed Defect"), ("3", "Reversible Defect")]},
    ],
    "diabetes": [
        {"name": "Pregnancies", "label": "Number of Pregnancies", "type": "number", "min": 0, "max": 20, "default": 1},
        {"name": "Glucose", "label": "Glucose Level (mg/dL)", "type": "number", "min": 0, "max": 300, "default": 110},
        {"name": "BloodPressure", "label": "Blood Pressure (mm Hg)", "type": "number", "min": 0, "max": 200, "default": 70},
        {"name": "SkinThickness", "label": "Skin Thickness (mm)", "type": "number", "min": 0, "max": 100, "default": 20},
        {"name": "Insulin", "label": "Insulin Level (mu U/ml)", "type": "number", "min": 0, "max": 900, "default": 80},
        {"name": "BMI", "label": "BMI", "type": "number", "step": "0.1", "min": 10, "max": 70, "default": 25},
        {"name": "DiabetesPedigreeFunction", "label": "Diabetes Pedigree Function", "type": "number", "step": "0.01", "min": 0, "max": 3, "default": 0.5},
        {"name": "Age", "label": "Age", "type": "number", "min": 1, "max": 120, "default": 35},
    ],
    "breast_cancer": [
        {"name": "mean radius", "label": "Mean Radius", "type": "number", "step": "0.01", "default": 14},
        {"name": "mean texture", "label": "Mean Texture", "type": "number", "step": "0.01", "default": 19},
        {"name": "mean perimeter", "label": "Mean Perimeter", "type": "number", "step": "0.01", "default": 92},
        {"name": "mean area", "label": "Mean Area", "type": "number", "step": "0.1", "default": 655},
        {"name": "mean smoothness", "label": "Mean Smoothness", "type": "number", "step": "0.001", "default": 0.096},
        {"name": "mean compactness", "label": "Mean Compactness", "type": "number", "step": "0.001", "default": 0.104},
        {"name": "mean concavity", "label": "Mean Concavity", "type": "number", "step": "0.001", "default": 0.089},
        {"name": "mean concave points", "label": "Mean Concave Points", "type": "number", "step": "0.001", "default": 0.049},
        {"name": "mean symmetry", "label": "Mean Symmetry", "type": "number", "step": "0.001", "default": 0.181},
        {"name": "mean fractal dimension", "label": "Mean Fractal Dimension", "type": "number", "step": "0.001", "default": 0.063},
    ],
}



def predict_disease(disease_key: str, form_data: dict):
    artifacts = ARTIFACTS[disease_key]
    meta = artifacts["meta"]
    feature_names = meta["feature_names"]
    age_col = meta["age_col"]
    bmi_col = meta["bmi_col"]
    risk_stats = meta["risk_stats"]
    medians = meta["feature_medians"]

  
    engineered_cols = {"AgeGroup", "BMICategory", "RiskScore"}
    raw_columns = [c for c in feature_names if c not in engineered_cols]

    row = {}
    for col in raw_columns:
        if col in form_data and form_data[col] not in (None, ""):
            row[col] = float(form_data[col])
        else:
            row[col] = medians[col]
    df = pd.DataFrame([row])

    
    df, _ = engineer_features(df, age_col=age_col, bmi_col=bmi_col, risk_stats=risk_stats)

    # Encode categorical engineered columns with the SAVED encoders
    for col, encoder in artifacts["encoders"].items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col].astype(str))
            except ValueError:
               
                df[col] = 0

    # Ensure exact column order the model was trained on
    df = df[feature_names]

    # Scale + predict
    X_scaled = artifacts["scaler"].transform(df.values)
    proba = artifacts["model"].predict_proba(X_scaled)[0]
    prediction = int(np.argmax(proba))
    confidence = float(proba[prediction]) * 100

    return {
        "disease": DISEASE_LABELS[disease_key],
        "prediction": prediction,          # 0 = low risk, 1 = disease risk
        "confidence": round(confidence, 2),
        "model_used": meta["best_model_name"],
        "label": "Disease Risk Detected" if prediction == 1 else "Low Risk / No Disease",
    }



@app.route("/")
def index():
    return render_template("index.html", form_fields=FORM_FIELDS, diseases=DISEASE_LABELS)


@app.route("/predict/<disease_key>", methods=["POST"])
def predict(disease_key):
    if disease_key not in DISEASES:
        return jsonify({"error": "Unknown disease type"}), 400

    try:
        form_data = request.get_json(force=True)
        result = predict_disease(disease_key, form_data)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5001)
