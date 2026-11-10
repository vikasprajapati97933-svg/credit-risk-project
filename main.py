from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = BASE_DIR / "credit_risk_model.pkl"
THRESHOLD_PATH = BASE_DIR / "best_threshold.pkl"
LEGACY_THRESHOLD_PATH = BASE_DIR / "best_thershold.pkl"

ml_model = {}


def train_fallback_model():
    dataset_path = BASE_DIR / "credit_risk_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    target_col = "loan_status"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_columns = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    categorical_columns = [col for col in X.columns if col not in numeric_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, probabilities)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    best_threshold = float(thresholds[np.argmax(f1_scores)])

    joblib.dump(model, MODEL_PATH)
    joblib.dump(best_threshold, THRESHOLD_PATH)
    return model, best_threshold


def load_model_assets():
    candidate_thresholds = [THRESHOLD_PATH, LEGACY_THRESHOLD_PATH]
    threshold = None
    for threshold_file in candidate_thresholds:
        if not threshold_file.exists():
            continue
        try:
            threshold = float(joblib.load(threshold_file))
            if np.isfinite(threshold):
                break
        except Exception:
            continue

    candidate_models = [MODEL_PATH]
    model = None
    for model_file in candidate_models:
        if not model_file.exists():
            continue
        try:
            loaded_model = joblib.load(model_file)
            if hasattr(loaded_model, "predict_proba"):
                model = loaded_model
                break
        except Exception:
            continue

    if model is None:
        model, threshold = train_fallback_model()
    elif threshold is None:
        _, threshold = train_fallback_model()

    return model, threshold


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, threshold = load_model_assets()
    ml_model["model"] = model
    ml_model["threshold"] = threshold
    yield
    ml_model.clear()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoanApplication(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int


@app.get("/")
def read_root():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/predict")
def predict(data: LoanApplication):
    if not ml_model:
        model, threshold = load_model_assets()
        ml_model["model"] = model
        ml_model["threshold"] = threshold

    input_df = pd.DataFrame([data.model_dump()])
    probability = float(ml_model["model"].predict_proba(input_df)[:, 1][0])
    threshold = float(ml_model["threshold"])
    prediction = int(probability >= threshold)

    return {
        "default_probability": probability,
        "repayment_probability": 1.0 - probability,
        "default_prediction": prediction,
        "threshold": threshold,
        "Result": "High Risk" if prediction == 1 else "Low Risk",
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
