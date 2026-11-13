

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import joblib

# ==================================
# PROJECT PATH
# ==================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "classification_model.pkl"

# ==================================
# LOAD MODEL
# ==================================

if not MODEL_PATH.exists():
    raise FileNotFoundError("classification_model.pkl not found!")

model = joblib.load(MODEL_PATH)

# ==================================
# FASTAPI APPLICATION
# ==================================

app = FastAPI(
    title="Loan Classification Prediction",
    description="Loan Approval Prediction using Machine Learning",
    version="1.0"
)

# ==================================
# HTML TEMPLATES
# ==================================

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)

# ==================================
# INPUT FEATURES
# ==================================

class LoanInput(BaseModel):

    age: int
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: int
    loan_amount: int
    loan_term: int
    cibil_score: int
    residential_assets_value: int
    commercial_assets_value: int
    luxury_assets_value: int
    bank_asset_value: int
    employment_type: str
    years_employed: int
    loan_purpose: str
    property_area: str
    credit_history_years: int
    existing_loans: int
    previous_defaults: int
    monthly_expenses: int
    savings_balance: int
    monthly_emi: int
    debt_to_income_ratio: float
    interest_rate: float


# ==================================
# HOME PAGE
# ==================================

@app.get("/")
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="classification.html",
        context={}
    )


# ==================================
# PREDICTION API
# ==================================

@app.post("/predict")
def predict(data: LoanInput):

    try:

        # Convert input into DataFrame
        input_df = pd.DataFrame(
            [data.model_dump()]
        )

        print("Input DataFrame:")
        print(input_df)

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Get prediction result
        if prediction == 1:
            result = "Approved"
        else:
            result = "Rejected"

        print("Prediction:", prediction)
        print("Loan Status:", result)

        # Return output
        return {
            "prediction": int(prediction),
            "result": result,
            "message": "Prediction completed successfully"
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# ==================================
# HEALTH CHECK
# ==================================

@app.get("/health")
def health():

    return {
        "status": "Application is running",
        "model_loaded": model is not None
    }
