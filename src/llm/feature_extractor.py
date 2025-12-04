import os
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel
from typing import Dict, Any

# Ensure you have set the GEMINI_API_KEY environment variable.
try:
    CLIENT = genai.Client()
except Exception as e:
    print(f"Warning: Could not initialize Gemini client. Is GEMINI_API_KEY set? Error: {e}")
    CLIENT = None 

# Import the schema defined in schemas.py
from .schemas import StructuredFeatures

def extract_features_from_text(raw_user_input: str) -> StructuredFeatures:
    """
    Converts raw user text describing a meal/exercise into structured features
    using the Gemini LLM with enforced Pydantic output schema.
    """
    if CLIENT is None:
        print("ERROR: Gemini client not initialized. Returning empty/default features.")
        # Return a safe default object to prevent app crash
        return StructuredFeatures(
            carbs=0.0, protein=0.0, fat=0.0, GI=0.0,
            exercise_name="None", minutes=0, intensity_level="Low", numeric_intensity_factor=0.0
        )

    # 1. Define the System Instruction
    system_instruction = (
        "You are an expert feature extraction system for a blood sugar prediction model. "
        "Your task is to analyze the user's raw text describing a meal and/or exercise "
        "and extract all features listed in the provided JSON schema. "
        "You MUST respond ONLY with a JSON object that strictly adheres to the schema. "
        "If a specific feature is not mentioned (e.g., no exercise), set its value to a "
        "reasonable default (0.0, 0, or 'None') as appropriate for its type."
    )
    
    # 2. Define the User Prompt
    user_prompt = f"Extract structured features from this input: {raw_user_input}"

    # 3. Configure the Generation Request
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=StructuredFeatures, 
    )

    try:
        # 4. Call the Gemini API
        response = CLIENT.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_prompt,
            config=config,
        )

        # 5. Validate and return
        validated_features = StructuredFeatures.model_validate_json(response.text)
        return validated_features

    except Exception as e:
        print(f"Gemini API Error: {e}")
        # Return default features on error
        return StructuredFeatures(
            carbs=0.0, protein=0.0, fat=0.0, GI=0.0,
            exercise_name="None", minutes=0, intensity_level="Low", numeric_intensity_factor=0.0
        )