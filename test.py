import os
from dotenv import load_dotenv
from src.llm.feature_extractor import extract_features_from_text
from src.utils.alerts import trigger_emergency_call

# 1. Load Environment Variables
load_dotenv()

def test_llm_connection():
    print("\n--- Testing LLM Feature Extraction ---")
    test_text = "I ate a large banana and ran for 30 minutes."
    print(f"Input Text: '{test_text}'")
    
    try:
        features = extract_features_from_text(test_text)
        print("SUCCESS: Features Extracted:")
        print(features.model_dump_json(indent=2))
        return True
    except Exception as e:
        print(f"FAILURE: LLM Extraction failed. Error: {e}")
        return False

def test_twilio_connection():
    print("\n--- Testing Twilio Alert System ---")
    caretaker_number = os.getenv("TWILIO_CARETAKER_NUMBER")
    
    if not caretaker_number:
        print("FAILURE: TWILIO_CARETAKER_NUMBER is not set in .env")
        return False
        
    print(f"Simulating critical glucose event (45 mg/dL). Calling {caretaker_number}...")
    
    try:
        # Trigger call with a mock critical value
        trigger_emergency_call(45.0, caretaker_number, user_name="Test User")
        print("SUCCESS: Twilio call initiated (Check your phone!)")
        return True
    except Exception as e:
        print(f"FAILURE: Twilio call failed. Error: {e}")
        return False

if __name__ == "__main__":
    print("Starting Integration Tests...")
    
    llm_status = test_llm_connection()
    
    # Only proceed to Twilio test if you want to actually make a call
    confirm = input("\nDo you want to proceed with the Twilio phone call test? (y/n): ")
    if confirm.lower() == 'y':
        twilio_status = test_twilio_connection()
    else:
        print("Skipping Twilio test.")
        
    print("\nTests Completed.")