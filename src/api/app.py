import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from typing import Dict, Any

# Assuming these are the locations for your components
from src.llm.feature_extractor import extract_features_from_text, StructuredFeatures
from src.llm.schemas import RawInput, PredictionResult

# --- Initialization ---
app = FastAPI(
    title="Blood Sugar Prediction API",
    description="Backend for LLM-powered blood glucose prediction and risk assessment.",
    version="1.0.0"
)

# 1. Load the Trained ML Model (Phase 1)
# NOTE: Replace 'model.pkl' with your actual saved model file path
try:
    ML_MODEL = joblib.load("models/blood_glucose_model.pkl") 
    print("ML Model loaded successfully.")
except FileNotFoundError:
    print("WARNING: ML Model not found. Using a dummy function for prediction.")
    ML_MODEL = None


# --- Helper Functions ---

def run_risk_classifier(predicted_glucose: float) -> tuple[str, str]:
    """Maps the predicted glucose value to a risk label and explanation (Phase 3 logic)."""
    if predicted_glucose < 70:
        return "Hypo", "Blood sugar is predicted to be low (Hypoglycemia). Seek immediate attention."
    elif predicted_glucose > 180:
        return "Hyper", "Blood sugar is predicted to be high (Hyperglycemia). Monitor closely."
    elif predicted_glucose >= 70 and predicted_glucose <= 100:
        return "Normal", "Prediction indicates normal blood sugar levels."
    else:
        return "Borderline", "Blood sugar is predicted to be elevated but not yet high."

# 1. Update the ML Model Loading to include a Preprocessor
try:
    ML_MODEL = joblib.load("models/blood_glucose_model.pkl") 
    # YOU MUST TRAIN AND SAVE THIS OBJECT DURING PHASE 1
    FEATURE_PREPROCESSOR = joblib.load("models/feature_preprocessor.pkl") 
    print("ML Model and Preprocessor loaded successfully.")
except FileNotFoundError:
    print("WARNING: ML Model/Preprocessor not found. Using dummy functions.")
    ML_MODEL = None
    FEATURE_PREPROCESSOR = None # Handle this gracefully

# 2. Update the prepare_features_for_ml function
def prepare_features_for_ml(features: StructuredFeatures, current_glucose: float) -> np.ndarray:
    """
    Converts the StructuredFeatures object into the final, encoded NumPy array 
    expected by the trained ML model.
    """
    
    # 1. Collect all features, including current glucose
    data = features.model_dump()
    # Rename 'current_glucose_mgdl' to 'fasting_glucose' to match the training schema
    data['fasting_glucose'] = current_glucose 
    
    # 2. Create a Pandas DataFrame (required for scikit-learn preprocessing)
    # The columns must match the order and names used during training!
    feature_df = pd.DataFrame([data])
    
    if FEATURE_PREPROCESSOR:
        # 3. Apply the saved preprocessor to handle one-hot encoding, scaling, etc.
        processed_features = FEATURE_PREPROCESSOR.transform(feature_df)
        return processed_features
    else:
        # Fallback (e.g., if preprocessor failed to load)
        # This is a very simple list that will likely fail if the model is complex
        feature_vector = [
            data['fasting_glucose'],
            data['carbs'],
            data['protein'],
            data['fat'],
            data['GI'],
            # The model will expect ENCODED versions of these next three:
            # data['exercise_name'], data['minutes'], data['intensity_level'], data['numeric_intensity_factor']
        ]
        return np.array([feature_vector]) # Returns an unencoded NumPy array


# --- API Endpoint (Phase 5: Real-Time Prediction Backend) ---

@app.post("/predict", response_model=PredictionResult)
def predict_glucose(input_data: RawInput):
    """
    Takes raw input, uses the LLM to extract features, predicts glucose 
    using the ML model, and classifies the risk.
    """
    
    # 1. LLM Feature Extraction (Phase 2)
    # The Pydantic RawInput ensures we have the correct fields.
    extracted_features = extract_features_from_text(input_data.raw_meal_exercise_text)

    # 2. Prepare Features for ML
    ml_input = prepare_features_for_ml(extracted_features, input_data.current_glucose_mgdl)

    # 3. ML Prediction (Phase 1)
    if ML_MODEL:
        # Predict the next_3hr_glucose
        prediction = ML_MODEL.predict(ml_input)[0] 
    else:
        # Fallback if model is not loaded (e.g., during testing)
        prediction = input_data.current_glucose_mgdl + (extracted_features.carbs * 0.5) 
        
    
    # 4. Risk Classification (Phase 3)
    risk_label, explanation = run_risk_classifier(prediction)

    # 5. Return the Result
    return PredictionResult(
        predicted_glucose_mgdl=round(float(prediction), 2),
        risk_label=risk_label,
        explanation=explanation
    )