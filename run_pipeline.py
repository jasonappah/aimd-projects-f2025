import os
import subprocess
import sys
import time

# --- Configuration ---
# Uses the current python executable to ensure we use the same environment
PYTHON_EXEC = sys.executable 
# Path to your data file (relative to project root)
DATA_PATH = "data/glucose_timeseries_5000_24h.csv" 

# Define the experiments to run based on your existing configs
EXPERIMENTS = [
    {
        "name": "1. Baseline Models (XGBoost/Logistic Regression)",
        "script": "scripts/train_baselines.py",
        "args": ["--config", "configs/baseline_config.yaml"]
    },
    {
        "name": "2. LSTM (Long Short-Term Memory)",
        "script": "scripts/train_deep.py",
        "args": ["--config", "configs/lstm_config.yaml"]
    },
    {
        "name": "3. TCN (Temporal Convolutional Network)",
        "script": "scripts/train_deep.py",
        "args": ["--config", "configs/tcn_config.yaml"]
    },
    {
        "name": "4. Transformer (Time-Series Transformer)",
        "script": "scripts/train_deep.py",
        "args": ["--config", "configs/transformer_config.yaml"]
    }
]

def check_data():
    """Verifies that the data file exists before starting."""
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Data file not found at: {DATA_PATH}")
        print("Please ensure 'glucose_timeseries_5000_24h.csv' is inside the 'data' folder.")
        return False
    return True

def run_command(command, step_name):
    """Runs a shell command and prints status."""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"CMD: {' '.join(command)}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    try:
        # CRITICAL: Add current directory to PYTHONPATH so 'src' can be imported by the scripts
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
        
        # Run the script
        result = subprocess.run(command, env=env, check=True)
        
        duration = time.time() - start_time
        print(f"\n[SUCCESS] {step_name} completed in {duration:.2f} seconds.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {step_name} failed with exit code {e.returncode}.")
        return False

def main():
    if not check_data():
        sys.exit(1)

    print("Starting Hypoglycemia Prediction Pipeline...")
    
    # 1. Run Training Loops
    results = {}
    for exp in EXPERIMENTS:
        cmd = [PYTHON_EXEC, exp["script"]] + exp["args"]
        success = run_command(cmd, exp["name"])
        results[exp["name"]] = "Success" if success else "Failed"
        
        # Stop the pipeline if a step fails
        if not success:
            print(f"Stopping pipeline due to failure in {exp['name']}")
            break

    # 2. Run Evaluation / Comparison
    if all(r == "Success" for r in results.values()):
        print("\nAll training jobs finished. Running comparison...")
        compare_cmd = [PYTHON_EXEC, "scripts/compare_models.py"]
        run_command(compare_cmd, "Model Comparison & Evaluation")

    # 3. Final Summary
    print("\n--- Pipeline Summary ---")
    for name, status in results.items():
        print(f"{name}: {status}")

if __name__ == "__main__":
    main()