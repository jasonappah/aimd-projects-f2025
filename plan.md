
Project Framework: Blood Sugar Prediction System
==============================================

1. High-Level Architecture
--------------------------
User enters glucose + meal/exercise text → Backend stores input → LLM extracts structured features →
ML model predicts next-3hr glucose → Risk layer classifies → UI displays risk and explanation.

2. Data & Schema
----------------
Training data fields include patient profile, fasting glucose, meal nutrients, exercise details (with names),
insulin usage, and next_3hr_glucose. Real system stores raw text + LLM-extracted features + prediction.

3. ML Training Pipeline
-----------------------
- Load synthetic dataset
- Train/validation split
- Train RandomForest/XGBoost
- Save model + feature schema
- Map predicted glucose to risk: hypo (<70), borderline, normal, hyper (>180)

4. LLM Feature Extraction Pipeline
----------------------------------
LLM converts raw meal/exercise text into:
carbs, protein, fat, GI, exercise_name, minutes, intensity_level, numeric intensity factor.

5. Real-Time Prediction Backend
-------------------------------
POST /predict takes raw input → LLM → structured features → ML → risk classifier → returns prediction + label.

6. Front-End Flow
-----------------
Screens: Login/Profile, Daily Entry, Prediction Result, History/Trends.

7. Roadmap
----------
Phase 1: Offline ML
Phase 2: LLM feature extraction
Phase 3: Backend API
Phase 4: Front-end
Phase 5: Feedback loop + retraining

