"""
risk_model.py
The non-visual part of the app: loading the saved model, converting answers into the codes the
model was trained on, and making a prediction. Kept separate from app.py so it can be tested
without Streamlit.
"""
from bisect import bisect_right
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).parent / "models"
DEFAULT_MODEL = "logreg_70k.joblib"

# About 14% of the surveyed adults had diabetes. The model learned from 50/50 data, so its raw
# probability overstates real risk; we rescale it to this prevalence (same formula as 03_evaluate).
REAL_PREVALENCE = 0.14

# BRFSS codes ---------------------------------------------------------------------------------
AGE_UPPER_BOUNDS = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]   # code 1 = 18-24 ... code 13 = 80+

EDUCATION = {
    1: "Never attended school, or kindergarten only",
    2: "Grades 1-8 (elementary)",
    3: "Grades 9-11 (some high school)",
    4: "Grade 12 or GED (high school graduate)",
    5: "College 1-3 years (some college or technical school)",
    6: "College 4+ years (college graduate)",
}
INCOME = {
    1: "Less than $10,000",
    2: "$10,000 to under $15,000",
    3: "$15,000 to under $20,000",
    4: "$20,000 to under $25,000",
    5: "$25,000 to under $35,000",
    6: "$35,000 to under $50,000",
    7: "$50,000 to under $75,000",
    8: "$75,000 or more",
}
# The BRFSS Income cutoffs are in nominal 2015 U.S. dollars. This optional PHP conversion uses
# the BSP daily reference rate and is only a currency conversion, not a local cost-of-living adjustment.
# Update from https://www.bsp.gov.ph/statistics/external/day99_data.aspx before future demonstrations.
INCOME_USD_CUTOFFS = [10_000, 15_000, 20_000, 25_000, 35_000, 50_000, 75_000]
PHP_PER_USD = 63.419
PHP_PER_USD_AS_OF = "23 Oct 2026"
GENERAL_HEALTH = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}

BMI_MIN, BMI_MAX = 12, 98      # range of BMI values the model saw in training


def age_to_code(age_years: int) -> int:
    """Exact age -> BRFSS age group code (1-13)."""
    return bisect_right(AGE_UPPER_BOUNDS, age_years) + 1


def age_group_label(code: int) -> str:
    if code == 13:
        return "80 or older"
    lo = 18 if code == 1 else AGE_UPPER_BOUNDS[code - 2]
    return f"{lo}-{AGE_UPPER_BOUNDS[code - 1] - 1}"


def income_code_from_usd(annual_income_usd: float) -> int:
    """Convert annual household income to the BRFSS income category (codes 1-8)."""
    return bisect_right(INCOME_USD_CUTOFFS, annual_income_usd) + 1


def compute_bmi(height_cm: float, weight_kg: float) -> float:
    return weight_kg / (height_cm / 100) ** 2


def bmi_to_code(bmi: float) -> int:
    """BMI as a whole number, kept inside the range seen in training."""
    return int(round(min(max(bmi, BMI_MIN), BMI_MAX)))


def bmi_category(bmi: float) -> str:
    if bmi < 18.5: return "Underweight"
    if bmi < 25: return "Healthy weight"
    if bmi < 30: return "Overweight"
    return "Obesity"


def feet_inches_to_cm(feet: float, inches: float) -> float:
    return (feet * 12 + inches) * 2.54


def pounds_to_kg(pounds: float) -> float:
    return pounds * 0.45359237


# Model ------------------------------------------------------------------------------------------
def list_models():
    """All saved model files, the default one first."""
    files = sorted(MODELS_DIR.glob("*.joblib"))
    return sorted(files, key=lambda p: (p.name != DEFAULT_MODEL, p.name))


def load_bundle(path) -> dict:
    """The saved dictionary: model, features, threshold, ..."""
    bundle = joblib.load(path)
    for key in ("model", "features", "threshold"):
        if key not in bundle:
            raise ValueError(f"'{Path(path).name}' is missing '{key}'. Was it saved by 02_train?")
    return bundle


def adjust_probability(p: float, prevalence: float = REAL_PREVALENCE) -> float:
    """Convert a probability from the balanced (50/50) scale to the real prevalence."""
    odds = p / (1 - p) * (prevalence / (1 - prevalence))
    return odds / (1 + odds)


def predict(bundle: dict, answers: dict) -> dict:
    """answers: {feature: coded value}. Returns the score, the adjusted probability and the flag."""
    row = pd.DataFrame([answers])[bundle["features"]]          # same columns, same order as training
    score = float(bundle["model"].predict_proba(row)[0, 1])
    return {
        "score": score,
        "adjusted": adjust_probability(score),
        "threshold": float(bundle["threshold"]),
        "flagged": score >= bundle["threshold"],
    }
