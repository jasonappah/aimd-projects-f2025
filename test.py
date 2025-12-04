import sys
import os
import traceback  # Import traceback to print full error details
from dotenv import load_dotenv

# 1. Load Environment Variables FIRST
# CRITICAL FIX: This must run BEFORE importing src modules so the Gemini Client 
# can find the API Key when it initializes at import time.
load_dotenv()

# --- CRITICAL FIX: Add project root to Python Path ---
# This allows 'from src.llm...' imports to work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.llm.feature_extractor import extract_features_from_text
    from src.utils.alerts import trigger_emergency_call
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import project modules. {e}")
    print("Ensure you are running this script from the project root directory.")
    sys.exit(1)

def test_llm_connection():
    print("\n--- Testing LLM Feature Extraction ---")
    test_text = "I ate a large banana and ran for 30 minutes."
    print(f"Input Text: '{test_text}'")
    
    # Check API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("FAILURE: GEMINI_API_KEY not found in .env")
        return False
    
    try:
        features = extract_features_from_text(test_text)
        print("SUCCESS: Features Extracted:")
        print(features.model_dump_json(indent=2))
        return True
    except Exception as e:
        print(f"FAILURE: LLM Extraction failed.")
        print(f"Error Message: {e}")
        print("-" * 20)
        traceback.print_exc()  # Print the full stack trace
        print("-" * 20)
        return False

def test_twilio_connection():
    print("\n--- Testing Twilio Alert System ---")
    caretaker_number = os.getenv("TWILIO_CARETAKER_NUMBER")
    
    # Debugging credentials
    acc_sid = os.getenv("ACC_SID_TWILIO")
    auth_token = os.getenv("AUTH_TOKEN_TWILIO")
    
    if not all([acc_sid, auth_token, caretaker_number]):
        print("FAILURE: Missing Twilio credentials in .env")
        print(f"  ACC_SID: {'Found' if acc_sid else 'Missing'}")
        print(f"  AUTH_TOKEN: {'Found' if auth_token else 'Missing'}")
        print(f"  CARETAKER_NUMBER: {'Found' if caretaker_number else 'Missing'}")
        return False
        
    print(f"Simulating critical glucose event (45 mg/dL). Calling {caretaker_number}...")
    
    try:
        # Trigger call with a mock critical value
        trigger_emergency_call(45.0, caretaker_number, user_name="Integration Test User")
        print("SUCCESS: Twilio call logic executed (Check if phone rings!)")
        return True
    except Exception as e:
        print(f"FAILURE: Twilio call failed. Error: {e}")
        return False

if __name__ == "__main__":
    print("Starting Quick Integration Tests...")
    
    # 1. Test LLM
    llm_status = test_llm_connection()
    
    if not llm_status:
        print("\nSkipping Twilio test due to LLM failure (or check credentials).")
    else:
        # 2. Test Twilio (Ask first to avoid spam calling)
        confirm = input("\nLLM Test Passed. Proceed with Twilio phone call test? (y/n): ")
        if confirm.lower() == 'y':
            test_twilio_connection()
        else:
            print("Skipping Twilio test.")
        
    print("\nTests Completed.")