import joblib
from fastapi import FastAPI
from typing import Dict, Any

# Assuming these are the locations for your components
from src.llm.feature_extractor import extract_features_from_text, StructuredFeatures
from src.schemas import RawInput, PredictionResult

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

def prepare_features_for_ml(features: StructuredFeatures, current_glucose: float) -> Any:
    """
    Converts the StructuredFeatures object into the format expected by the 
    trained scikit-learn/XGBoost model (e.g., a Pandas DataFrame or numpy array).
    """
    # NOTE: You must map all fields in the exact order your model was trained on.
    # This is a placeholder for your actual feature preparation logic.
    data = features.model_dump()
    data['fasting_glucose'] = current_glucose # Add the current reading as a feature
    
    # In a real system, you would handle one-hot encoding for 'exercise_name'
    # and 'intensity_level' to match the training feature set.
    
    # Example: returning a list of values for a simple model expectation
    feature_vector = [
        data['fasting_glucose'],
        data['carbs'],
        data['protein'],
        # ... include all other features in correct order
        data['numeric_intensity_factor']
    ]
    
    # Scikit-learn models typically expect an array-like structure (e.g., [[feature1, feature2, ...]])
    import numpy as np
    return np.array([feature_vector])


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