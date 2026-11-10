# Credit Risk Predictor

Credit Risk Assessment is a machine-learning web application that estimates the probability of loan default from an applicant's profile and requested loan details. It presents both default and repayment probabilities, then compares the estimated default risk with a model decision threshold to label the application **High Risk** or **Low Risk**.

> **Live demo:** [credit-risk-project-fdrt.onrender.com](https://credit-risk-project-fdrt.onrender.com)

This project is for demonstration and educational purposes. Its estimates are not financial advice or a lending decision, and should not be used as the sole basis for approving or denying credit.

## What it does

- Accepts applicant, credit-history, and loan information through a web form.
- Calculates the loan-to-income ratio from the requested amount and annual income.
- Returns estimated default and repayment probabilities from a scikit-learn model.
- Applies a saved classification threshold and displays the result alongside that threshold.
- Exposes a JSON prediction endpoint and interactive API documentation.
- Includes a Jupyter notebook for data exploration, model experiments, and evaluation.

## How it works

The FastAPI application loads `credit_risk_model.pkl` and a threshold from `best_threshold.pkl`. For compatibility with an earlier filename, it also checks `best_thershold.pkl`. If a usable model or threshold is unavailable, the application trains a fallback logistic-regression pipeline from the bundled dataset. The fallback pipeline imputes missing values, one-hot encodes categorical features, and selects a decision threshold using the F1 score on a held-out split.

The notebook explores the data and compares classification approaches, including logistic regression and XGBoost, with classification metrics and probability-threshold experiments. The deployed prediction behavior uses the model artifact that is loaded at startup; the notebook is an experimentation workflow rather than a guarantee that a particular experiment is the deployed model.

## Dataset

`credit_risk_dataset.csv` contains 32,581 loan records and 12 columns. The target is `loan_status` (0 or 1). Applicant and loan features include:

- `person_age`, `person_income`, `person_home_ownership`, `person_emp_length`
- `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`
- `cb_person_default_on_file`, `cb_person_cred_hist_length`

The target column is used for model training and is not an input to the prediction endpoint.

## Run locally

Requires Python 3.13.4 (see `runtime.txt`).

In PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) for the application or [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the interactive API documentation.

## Prediction API

Send a JSON `POST` request to `/predict`. All fields are required:

```json
{
  "person_age": 30,
  "person_income": 75000,
  "person_home_ownership": "RENT",
  "person_emp_length": 5,
  "loan_intent": "PERSONAL",
  "loan_grade": "C",
  "loan_amnt": 12000,
  "loan_int_rate": 11.5,
  "loan_percent_income": 0.16,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 5
}
```

`loan_percent_income` is a ratio, not a percentage: for example, `0.16` represents 16%. The web form calculates it automatically.

The response includes:

- `default_probability` and `repayment_probability`: estimated probabilities from 0 to 1.
- `default_prediction`: 1 when the default probability meets or exceeds the model threshold, otherwise 0.
- `threshold`: the decision threshold used for this prediction.
- `Result`: `"High Risk"` or `"Low Risk"`.

## Project files

| File | Purpose |
| --- | --- |
| `main.py` | FastAPI application, model loading, fallback training, and prediction endpoint |
| `static/` | Web application HTML, CSS, and JavaScript |
| `credit risk ml system.ipynb` | Data analysis, model experiments, and evaluation |
| `credit_risk_dataset.csv` | Dataset used for model training |
| `credit_risk_model.pkl` | Serialized prediction model |
| `best_threshold.pkl` | Saved classification threshold |
| `requirements.txt` | Python dependencies |
| `render.yaml` | Render deployment configuration |

## Deployment

The included `render.yaml` configures a Python web service on Render. It installs dependencies from `requirements.txt` and starts the app with Uvicorn.

