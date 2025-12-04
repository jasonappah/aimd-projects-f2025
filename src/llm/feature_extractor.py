import os
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel
from typing import Dict, Any

# Ensure you have set the GEMINI_API_KEY environment variable.
# Example: client = genai.Client(api_key="YOUR_API_KEY")
# If GEMINI_API_KEY is set in your environment, the client initializes automatically.
try:
    CLIENT = genai.Client()
except Exception as e:
    print(f"Warning: Could not initialize Gemini client. Is GEMINI_API_KEY set? Error: {e}")
    CLIENT = None # Handle case where client setup fails initially

# --- 1. Import or Define the Schema ---
# Assuming 'StructuredFeatures' is defined in 'src/llm/schemas.py'
# You must import the exact class used for the ML model's feature set.
from .schemas import StructuredFeatures
# -------------------------------------


def extract_features_from_text(raw_user_input: str) -> StructuredFeatures:
    """
    Converts raw user text describing a meal/exercise into structured features
    using the Gemini LLM with enforced Pydantic output schema.

    Args:
        raw_user_input: The free-form text entered by the user.

    Returns:
        An instance of StructuredFeatures, validated by Pydantic.
    """
    if CLIENT is None:
        raise RuntimeError("Gemini client is not initialized. Check your API key setup.")

    # 1. Define the System Instruction (Context)
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
        # This is the key: tell Gemini to format its output according to the Pydantic schema
        response_mime_type="application/json",
        response_schema=StructuredFeatures, 
    )

    try:
        # 4. Call the Gemini API
        response = CLIENT.models.generate_content(
            model='gemini-2.5-flash',  # A fast model suitable for structured extraction
            contents=user_prompt,
            config=config,
        )

        # 5. The response.text will be a valid JSON string conforming to StructuredFeatures
        # Pydantic's parse_raw or model_validate_json can convert the JSON text back into
        # a validated Python object.
        validated_features = StructuredFeatures.model_validate_json(response.text)
        
        return validated_features

    except APIError as e:
        print(f"Gemini API Error during feature extraction: {e}")
        # In a real system, you'd handle this error (e.g., return a default
        # set of features or log the error and notify the user).
        raise
    except Exception as e:
        print(f"General error: {e}")
        raise


# Example usage (for testing)
if __name__ == "__main__":
    # Test 1: Meal entry
    text_meal = "A bowl of cheerios with milk, roughly 60g carbs, 10g protein, and 5g fat."
    features_meal = extract_features_from_text(text_meal)
    print("\n--- Extracted Meal Features ---")
    print(features_meal.model_dump_json(indent=2))

    # Test 2: Combined entry
    text_combined = "Had a quick sandwich (30g carbs) and then did a 30-minute intense run."
    features_combined = extract_features_from_text(text_combined)
    print("\n--- Extracted Combined Features ---")
    print(features_combined.model_dump_json(indent=2))