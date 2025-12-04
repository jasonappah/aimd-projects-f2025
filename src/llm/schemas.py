from pydantic import BaseModel, Field

# Defines the exact structure and data types for the ML model's input
# This ensures a seamless connection between the LLM and the ML model.
class StructuredFeatures(BaseModel):
    """
    Schema for the structured features extracted by the LLM from raw meal/exercise text.
    Matches the requirements in plan.md: Section 4. LLM Feature Extraction Pipeline.
    """
    # Meal Features
    carbs: float = Field(description="Total carbohydrates in grams (g). Must be non-negative.")
    protein: float = Field(description="Total protein in grams (g). Must be non-negative.")
    fat: float = Field(description="Total fat in grams (g). Must be non-negative.")
    GI: float = Field(description="Glycemic Index (0 to 100). Estimate if not directly available.")

    # Exercise Features
    exercise_name: str = Field(description="Name of the exercise (e.g., 'Running', 'Weightlifting').")
    minutes: int = Field(description="Duration of the exercise in whole minutes. Must be non-negative.")
    intensity_level: str = Field(description="Categorical intensity: 'Low', 'Medium', or 'High'.")
    numeric_intensity_factor: float = Field(description="A numerical factor representing intensity (e.g., 0.5 to 1.5).")

# Example of how the data will look after extraction
# {
#     "carbs": 55.0,
#     "protein": 12.0,
#     "fat": 8.0,
#     "GI": 60.0,
#     "exercise_name": "Running",
#     "minutes": 30,
#     "intensity_level": "Medium",
#     "numeric_intensity_factor": 1.0
# }
    class RawInput(BaseModel):
        """
        Schema for the raw data coming from the Front-End (POST /predict).
        """
        current_glucose_mgdl: float
        raw_meal_exercise_text: str

    class PredictionResult(BaseModel):
        """
        Schema for the final response sent back to the Front-End.
        """
        predicted_glucose_mgdl: float
        risk_label: str # e.g., 'Normal', 'Hyper', 'Hypo'
        explanation: str # Textual explanation for the risk