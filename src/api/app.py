import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from typing import Any
import os

# Local imports for core logic and schemas
from src.llm.feature_extractor import extract_features_from_text, StructuredFeatures
from src.llm.schemas import RawInput, PredictionResult
from src.utils.alerts import trigger_emergency_call

# --- Initialization & Constants ---
app = FastAPI(
    title="Blood Sugar Prediction API",
    description="Backend for LLM-powered blood glucose prediction and critical risk assessment.",
    version="1.0.0"
)

# Constants for Critical Alert Thresholds (representing the >75% risk level)
CRITICAL_HYPO_THRESHOLD = 50.0  # Below 50 mg/dL: requires immediate attention
CRITICAL_HYPER_THRESHOLD = 300.0 # Above 300 mg/dL: requires immediate attention

# Load environment variables (runs once at startup)
CARETAKER_NUMBER = os.getenv("TWILIO_CARETAKER_NUMBER") 

# 1. CONSOLIDATED Model and Preprocessor Loading
# This block runs once when the application starts
try:
    ML_MODEL = joblib.load("models/blood_glucose_model.pkl") 
    FEATURE_PREPROCESSOR = joblib.load("models/feature_preprocessor.pkl") 
    print("ML Model and Preprocessor loaded successfully.")
except FileNotFoundError as e:
    # If the model or preprocessor files are missing, the prediction endpoint will use a fallback
    print(f"WARNING: Required file not found: {e}. ML functionality disabled.")
    ML_MODEL = None
    FEATURE_PREPROCESSOR = None

# --- Helper Functions ---

def run_risk_classifier(predicted_glucose: float) -> tuple[str, str]:
    """
    Maps the predicted glucose value to a risk label and explanation.

    Risk thresholds (based on common medical guidelines):
    Hypo: < 70 mg/dL
    Normal: 70 - 180 mg/dL
    Hyper: > 180 mg/dL
    """
    if predicted_glucose < 70:
        return "Hypo", "Blood sugar is predicted to be low (Hypoglycemia). Seek immediate attention."
    elif predicted_glucose > 180:
        return "Hyper", "Blood sugar is predicted to be high (Hyperglycemia). Monitor closely."
    elif predicted_glucose >= 70 and predicted_glucose <= 100:
        return "Normal", "Prediction indicates normal blood sugar levels."
    else:
        return "Borderline", "Blood sugar is predicted to be elevated but not yet high."

def prepare_features_for_ml(features: StructuredFeatures, current_glucose: float) -> np.ndarray:
    """
    Converts the StructuredFeatures object into the final, encoded NumPy array 
    expected by the trained ML model.
    """
    
    # 1. Collect all features, including current glucose
    data = features.model_dump()
    # Rename key for consistency with typical training data
    data['fasting_glucose'] = current_glucose 
    
    # 2. Create a Pandas DataFrame (required for scikit-learn preprocessing)
    # The columns MUST match the order and names used during training!
    feature_df = pd.DataFrame([data])
    
    if FEATURE_PREPROCESSOR:
        # 3. Apply the saved preprocessor to handle one-hot encoding, scaling, etc.
        processed_features = FEATURE_PREPROCESSOR.transform(feature_df)
        return processed_features
    else:
        # Fallback for when the preprocessor is not available
        # This is a simple, unencoded vector and should only be used in the fallback prediction
        feature_vector = [
            data['fasting_glucose'],
            data['carbs'],
            data['protein'],
            data['fat'],
            data['GI'],
            data['minutes'], 
            data['numeric_intensity_factor']
        ]
        return np.array([feature_vector])


# --- API Endpoint (The Core Prediction Logic) ---

@app.post("/predict", response_model=PredictionResult)
def predict_glucose(input_data: RawInput):
    """
    Primary endpoint: Takes raw input, extracts features via LLM, predicts glucose 
    via ML model, classifies risk, and triggers emergency alerts if necessary.
    """
    
    # 1. LLM Feature Extraction (Phase 2)
    extracted_features = extract_features_from_text(input_data.raw_meal_exercise_text)

    # 2. Prepare Features for ML
    ml_input = prepare_features_for_ml(extracted_features, input_data.current_glucose_mgdl)

    # 3. ML Prediction (Phase 1)
    if ML_MODEL:
        # Predict the next_3hr_glucose using the loaded model
        prediction = ML_MODEL.predict(ml_input)[0] 
    else:
        # Fallback if model is not loaded
        prediction = input_data.current_glucose_mgdl + (extracted_features.carbs * 0.5) 
        
    predicted_glucose = float(prediction)
    
    # 4. Risk Classification (Phase 3)
    risk_label, explanation = run_risk_classifier(predicted_glucose)
    
    # 5. CRITICAL ALERT CHECK (New Logic)
    if (predicted_glucose < CRITICAL_HYPO_THRESHOLD or 
        predicted_glucose > CRITICAL_HYPER_THRESHOLD):
        
        if CARETAKER_NUMBER:
            # Trigger the call with the predicted value
            trigger_emergency_call(predicted_glucose, CARETAKER_NUMBER)
            explanation += " CRITICAL ALERT: Emergency call initiated to caretaker."
        else:
            print("ALERT: TWILIO_CARETAKER_NUMBER not set. Emergency call skipped.")

    # 6. Return the Result
    return PredictionResult(
        predicted_glucose_mgdl=round(predicted_glucose, 2),
        risk_label=risk_label,
        explanation=explanation
    )