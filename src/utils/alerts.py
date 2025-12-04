from twilio.rest import Client
from dotenv import load_dotenv
import os

# Load environment variables (ensure this runs once at application start)
# This is usually needed if running locally; FastAPI deployment environments handle this differently.
load_dotenv() 

def trigger_emergency_call(predicted_glucose: float, caretaker_number: str, user_name: str = "the patient"):
    """
    Triggers a phone call to the caretaker via Twilio if the risk is critical.
    
    Args:
        predicted_glucose: The predicted glucose value (used in the message).
        caretaker_number: The phone number of the caretaker (e.g., '+1...').
        user_name: Identifier for the patient.
    """
    # Use environment variable names from your original snippet
    account_sid = os.getenv("ACC_SID_TWILIO") 
    auth_token = os.getenv("AUTH_TOKEN_TWILIO")
    twilio_phone_number = os.getenv("TWILIO_NUMBER") 
    
    # Safety check: ensure all Twilio credentials are set
    if not all([account_sid, auth_token, twilio_phone_number]):
        print("ALERT: Twilio credentials missing (ACC_SID_TWILIO, AUTH_TOKEN_TWILIO, or TWILIO_NUMBER). Emergency call NOT initiated.")
        return

    # TwiML message requesting an immediate prick test and action
    message = (
        f"This is a critical blood sugar alert for {user_name}. "
        f"The system predicts an extreme glucose reading of {predicted_glucose:.1f} mg/dL in three hours. "
        "Please perform an immediate finger prick test "
        "and take necessary corrective action. The prediction is dangerously low or high."
    )
    twiml_response = f"<Response><Say>{message}</Say></Response>"

    try:
        client = Client(account_sid, auth_token)

        call = client.calls.create(
            to=caretaker_number,
            from_=twilio_phone_number,
            twiml=twiml_response
        )

        print(f"CRITICAL ALERT: Call initiated to caretaker {caretaker_number}. Call SID: {call.sid}")
        
    except Exception as e:
        print(f"ERROR: Failed to initiate Twilio call to {caretaker_number}. Error: {e}")