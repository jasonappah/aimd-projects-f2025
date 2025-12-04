import numpy as np
import pickle
import torch
from fastapi import FastAPI
import os

# Local imports for core logic and schemas
from src.llm.feature_extractor import extract_features_from_text, StructuredFeatures
from src.llm.schemas import RawInput, PredictionResult
from src.utils.alerts import trigger_emergency_call
from src.models.blood_glucose_model import BloodGlucoseModel
from src.utils.device import get_device

# --- Initialization & Constants ---
app = FastAPI(
    title="GlycoCare - Blood Sugar Prediction API",
    description="Backend for LLM-powered blood glucose prediction and critical risk assessment.",
    version="1.0.0"
)

# Constants for Critical Alert Thresholds (representing the >75% risk level)
CRITICAL_HYPO_THRESHOLD = 50.0  # Below 50 mg/dL: requires immediate attention
CRITICAL_HYPER_THRESHOLD = 300.0 # Above 300 mg/dL: requires immediate attention

# Load environment variables (runs once at startup)
CARETAKER_NUMBER = os.getenv("TWILIO_CARETAKER_NUMBER") 

# Get device for PyTorch model
DEVICE = get_device()

# 1. CONSOLIDATED Model and Preprocessor Loading
# This block runs once when the application starts
ML_MODEL = None
SCALER = None
ENCODERS = None

try:
    # Load PyTorch model checkpoint
    checkpoint_path = "checkpoints/final_model.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "checkpoints/best_model.pth"
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model_config = checkpoint['model_config']
    
    # Initialize model with saved config
    ML_MODEL = BloodGlucoseModel(
        input_size=model_config['input_size'],
        hidden_sizes=model_config['hidden_sizes'],
        dropout_rate=model_config['dropout_rate']
    ).to(DEVICE)
    
    # Load model weights
    ML_MODEL.load_state_dict(checkpoint['model_state_dict'])
    ML_MODEL.eval()  # Set to evaluation mode
    print(f"PyTorch model loaded successfully from {checkpoint_path}")
    
    # Load scaler
    scaler_path = "checkpoints/scaler.pkl"
    with open(scaler_path, 'rb') as f:
        SCALER = pickle.load(f)
    print("Scaler loaded successfully.")
    
    # Load encoders
    encoders_path = "checkpoints/encoders.pkl"
    with open(encoders_path, 'rb') as f:
        ENCODERS = pickle.load(f)
    print("Encoders loaded successfully.")
    
except FileNotFoundError as e:
    print(f"WARNING: Required file not found: {e}. ML functionality disabled.")
except Exception as e:
    print(f"WARNING: Error loading model/preprocessors: {e}. ML functionality disabled.")

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

def prepare_features_for_ml(features: StructuredFeatures, current_glucose: float) -> torch.Tensor:
    """
    Converts the StructuredFeatures object into the final, preprocessed PyTorch tensor 
    expected by the trained ML model.
    
    This function:
    1. Maps LLM-extracted features to training data format
    2. Adds default values for missing patient-specific features
    3. Applies categorical encoders (diabetes_type, exercise_name, intensity_level)
    4. Applies numerical scaler
    5. Converts to PyTorch tensor
    """
    
    if SCALER is None or ENCODERS is None:
        # Fallback: return a simple feature vector (will use fallback prediction)
        data = features.model_dump()
        feature_vector = [
            current_glucose,  # fasting_glucose
            data['carbs'],
            data['protein'],
            data['fat'],
            data['GI'],
            data['minutes'], 
            data['numeric_intensity_factor']
        ]
        return torch.tensor([feature_vector], dtype=torch.float32).to(DEVICE)
    
    # 1. Map LLM features to training data format
    # Training data columns (numerical): age, years_since_dx, baseline_a1c, fasting_glucose,
    #   meal_carbs, meal_protein, meal_fat, meal_gi, exercise_minutes, exercise_intensity_factor,
    #   time_since_insulin_hr, insulin_dose_units
    # Training data columns (categorical): diabetes_type, exercise_name, exercise_intensity_level
    
    # Use defaults for missing patient-specific features
    # These could be made configurable via API input in the future
    age = 50.0  # Default age
    years_since_dx = 5.0  # Default years since diagnosis
    baseline_a1c = 7.0  # Default A1C
    time_since_insulin_hr = 0.0  # Default: no recent insulin
    insulin_dose_units = 0.0  # Default: no insulin dose
    
    # Map LLM features
    fasting_glucose = current_glucose
    meal_carbs = features.carbs
    meal_protein = features.protein
    meal_fat = features.fat
    meal_gi = features.GI
    exercise_minutes = float(features.minutes)
    exercise_intensity_factor = features.numeric_intensity_factor
    
    # 2. Handle categorical features with encoders
    diabetes_encoder = ENCODERS['diabetes']
    exercise_name_encoder = ENCODERS['exercise_name']
    intensity_encoder = ENCODERS['intensity']
    
    # Encode categorical features
    # Default to first category if not found (or use a sensible default)
    diabetes_type = "Type 2"  # Default diabetes type
    
    exercise_name = features.exercise_name if features.exercise_name != "None" else "None"
    try:
        exercise_name_encoded = exercise_name_encoder.transform([exercise_name])[0]
    except (ValueError, KeyError):
        # If exercise not in encoder, use 0 (likely "None" or first category)
        exercise_name_encoded = 0
    
    intensity_level = features.intensity_level
    
    # 3. One-hot encode diabetes_type and exercise_intensity_level
    diabetes_columns = ENCODERS.get('diabetes_columns', [])
    intensity_columns = ENCODERS.get('intensity_columns', [])
    
    # Create one-hot vectors
    diabetes_onehot = np.zeros(len(diabetes_columns), dtype=np.float32)
    if diabetes_type in diabetes_encoder.classes_:
        col_name = f"diabetes_{diabetes_type}"
        if col_name in diabetes_columns:
            idx = diabetes_columns.index(col_name)
            diabetes_onehot[idx] = 1.0
    
    intensity_onehot = np.zeros(len(intensity_columns), dtype=np.float32)
    if intensity_level in intensity_encoder.classes_:
        col_name = f"intensity_{intensity_level}"
        if col_name in intensity_columns:
            idx = intensity_columns.index(col_name)
            intensity_onehot[idx] = 1.0
    
    # 4. Combine numerical features (in the order expected by scaler)
    # The scaler was fit on: age, years_since_dx, baseline_a1c, fasting_glucose,
    #   meal_carbs, meal_protein, meal_fat, meal_gi, exercise_minutes, 
    #   exercise_intensity_factor, time_since_insulin_hr, insulin_dose_units
    numerical_features = np.array([[
        age,
        years_since_dx,
        baseline_a1c,
        fasting_glucose,
        meal_carbs,
        meal_protein,
        meal_fat,
        meal_gi,
        exercise_minutes,
        exercise_intensity_factor,
        time_since_insulin_hr,
        insulin_dose_units
    ]], dtype=np.float32)
    
    # 5. Apply scaler to numerical features
    numerical_features_scaled = SCALER.transform(numerical_features)
    
    # 6. Combine all features in the order expected by the model:
    # [numerical_scaled, diabetes_onehot, intensity_onehot, exercise_name_encoded]
    combined_features = np.hstack([
        numerical_features_scaled,
        diabetes_onehot.reshape(1, -1),
        intensity_onehot.reshape(1, -1),
        np.array([[exercise_name_encoded]], dtype=np.float32)
    ])
    
    # 7. Convert to PyTorch tensor
    feature_tensor = torch.tensor(combined_features, dtype=torch.float32).to(DEVICE)
    
    return feature_tensor


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
        # Predict the next_3hr_glucose using the PyTorch model
        with torch.no_grad():
            prediction_tensor = ML_MODEL(ml_input)
            prediction = prediction_tensor.cpu().item()
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