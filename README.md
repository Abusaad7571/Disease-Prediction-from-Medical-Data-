# 🩺 VitalCheck — Disease Prediction System

An end-to-end **Machine Learning + Flask** web app that predicts the risk of
**Heart Disease**, **Diabetes**, and **Breast Cancer** from patient data, and
shows the result with a confidence percentage.

---

## 1. Project Structure

```
Disease_Prediction_System/
├── app.py                     # Flask web app (routes + prediction pipeline)
├── train_model.py             # EDA + preprocessing + model training/comparison
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── dataset/
│   ├── generate_datasets.py   # Creates the 3 CSVs (see Section 2 below)
│   ├── heart.csv
│   ├── diabetes.csv
│   └── breast_cancer.csv
├── models/                    # Saved after running train_model.py
│   ├── heart_model.pkl / heart_scaler.pkl / heart_encoder.pkl / heart_meta.json
│   ├── diabetes_model.pkl / diabetes_scaler.pkl / diabetes_encoder.pkl / diabetes_meta.json
│   └── breast_cancer_model.pkl / breast_cancer_scaler.pkl / breast_cancer_encoder.pkl / breast_cancer_meta.json
├── reports/                   # EDA plots + model comparison tables (auto-generated)
├── templates/
│   └── index.html             # Patient form + result UI
├── static/
│   └── style.css              # "Vitals monitor" clinical UI theme
└── utils/
    └── preprocessing.py       # Shared cleaning / feature engineering / scaling / SMOTE
```

---

## 2. About the Datasets (important — please read)

The task asks for the **UCI Heart Disease**, **Pima Diabetes**, and
**Wisconsin Breast Cancer** datasets. This project uses:

| Dataset | Source used here | Why |
|---|---|---|
| Breast Cancer | **Real data** — `sklearn.datasets.load_breast_cancer()` | Ships locally with scikit-learn, no download needed |
| Heart Disease | **Synthetic data**, same 14 columns as the real UCI Cleveland dataset (`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target`) | Generated with realistic ranges + rule-based labels so the pipeline works without internet access |
| Diabetes | **Synthetic data**, same 9 columns as the real Pima Indians dataset (`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`) | Same reason as above |

**To use the real UCI/Kaggle datasets instead:** just download them and save
them as `dataset/heart.csv` and `dataset/diabetes.csv` with the **exact same
column names** listed above. Nothing else in the code needs to change —
`train_model.py` will pick them up automatically. Good sources:
- Kaggle: "Heart Disease UCI"
- Kaggle: "Pima Indians Diabetes Database"

---

## 3. How to Run (step by step)

### Step 1 — Install dependencies
```bash
cd Disease_Prediction_System
pip install -r requirements.txt
```

### Step 2 — Generate the datasets (only needed once)
```bash
python dataset/generate_datasets.py
```
This creates `heart.csv`, `diabetes.csv`, and `breast_cancer.csv` inside `dataset/`.

### Step 3 — Train the models
```bash
python train_model.py
```
This will:
- Print EDA info (missing values, class balance, stats) to the terminal
- Save plots to `reports/` (class distribution, correlation heatmap, ROC curve, confusion matrix, feature importance)
- Train **Logistic Regression, SVM, Random Forest, XGBoost** for all 3 diseases
- Print a comparison table and pick the **best model per disease** (highest F1 score)
- Save `models/<disease>_model.pkl`, `<disease>_scaler.pkl`, `<disease>_encoder.pkl`, `<disease>_meta.json`

> Note: if `xgboost` or `imbalanced-learn` (SMOTE) aren't installed, the
> script automatically skips XGBoost / falls back to simple random
> oversampling — so it never crashes, it just uses what's available.

### Step 4 — Run the Flask app
```bash
python app.py
```
Then open **http://127.0.0.1:5000** in your browser.

---

## 4. How the ML Pipeline Works (exam-prep style notes)

### 4.1 Data Analysis
- `df.info()`, `df.describe()` → dataset shape, types, statistical summary
- `df.isnull().sum()` → missing value counts
- `df.drop_duplicates()` → duplicate removal
- `sns.countplot()` → class distribution (checks imbalance)
- `sns.heatmap(df.corr())` → correlation heatmap (spot multicollinearity)
- `feature_importances_` / `coef_` → which features matter most

### 4.2 Preprocessing (`utils/preprocessing.py`)
| Step | Function | What it does |
|---|---|---|
| Clean | `clean_data()` | Fills missing numeric values with **median**, categorical with **mode**; drops duplicate rows |
| Feature engineering | `engineer_features()` | Adds `AgeGroup` (Young/Adult/MiddleAged/Senior), `BMICategory` (Underweight/Normal/Overweight/Obese), and a composite `RiskScore` (mean of normalized numeric features) |
| Encoding | `encode_categoricals()` | `LabelEncoder` turns text categories into numbers |
| Balancing | `balance_classes()` | **SMOTE** (Synthetic Minority Oversampling) creates synthetic examples of the minority class so the model doesn't just learn to predict the majority class |
| Scaling | `split_and_scale()` | `StandardScaler` — makes all features mean=0, std=1 (important for Logistic Regression & SVM, which are distance/gradient based) |

**Key exam point:** the scaler is fit **only on the training set**, then
applied to the test set. Fitting on the full dataset before splitting would
leak test-set information into training — a common ML mistake.

### 4.3 Model Training & Comparison
Four classifiers are trained and compared per disease:

| Model | Intuition |
|---|---|
| Logistic Regression | Simple, interpretable, draws a linear decision boundary |
| SVM (RBF kernel) | Finds the maximum-margin boundary, works well in high dimensions |
| Random Forest | Ensemble of decision trees, handles non-linear patterns well |
| XGBoost | Gradient-boosted trees, usually the strongest tabular-data model |

The model with the **highest F1 score** on the held-out test set is selected
as the "best model" and saved for the Flask app to use. F1 is chosen over
plain accuracy because medical datasets are often imbalanced (few disease
cases vs many healthy cases) — accuracy alone can be misleading there.

### 4.4 Evaluation Metrics — quick definitions
- **Accuracy** = correct predictions / total predictions
- **Precision** = of all patients predicted "at risk," how many actually are
- **Recall** = of all patients who actually are at risk, how many we caught
- **F1 Score** = harmonic mean of Precision and Recall (balances both)
- **ROC-AUC** = how well the model separates the two classes across all thresholds (1.0 = perfect, 0.5 = random guessing)
- **Confusion Matrix** = table of True Positives / False Positives / True Negatives / False Negatives

---

## 5. How the Flask App Works

1. User picks a disease tab (Heart / Diabetes / Breast Cancer) and fills the form.
2. On submit, JavaScript sends the form values as JSON to `POST /predict/<disease>`.
3. `app.py` rebuilds the **exact same features** used in training:
   - Fills any field not on the form with that feature's training-set median (e.g. Breast Cancer only asks for the 10 "mean" measurements; the 20 "worst"/"error" columns are auto-filled)
   - Recomputes `AgeGroup` / `BMICategory` / `RiskScore` the same way training did
   - Encodes categoricals with the **saved** encoder, scales with the **saved** scaler
4. The saved model returns a probability → the app reports:
   - ✅ **Low Risk / No Disease** or ❌ **Disease Risk Detected**
   - A **confidence percentage**
5. The result renders in a gauge + verdict card on the right side of the page.

> ⚠️ **Disclaimer:** This is an educational project. It is **not** a
> substitute for professional medical advice, diagnosis, or treatment.

---

## 6. Frontend Design

The UI ("VitalCheck") uses a clinical **vitals-monitor** theme:
- A live animated ECG pulse line in the header (signature visual element)
- Deep teal / soft blue-gray palette instead of generic AI-page defaults
- A radial confidence **gauge** that animates in on prediction
- Fully responsive down to mobile (disease tabs become a horizontal scroll row)
- Respects `prefers-reduced-motion` for users who disable animations

---

## 7. Extending the Project
- Swap in real UCI/Kaggle CSVs (see Section 2) for production-quality data
- Add more diseases by adding a new `run_pipeline(...)` call in `train_model.py`, a form-field list in `app.py`, and a tab in `index.html`
- Add authentication + a patient history database for a real deployment
- Replace Flask's dev server (`app.run(debug=True)`) with a production WSGI server (e.g. Gunicorn) before deploying publicly
