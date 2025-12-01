Quick summary (what to do first)
Strong baselines (ensure you beat them): logistic regression, XGBoost/LightGBM (tabular/time-window features).


Time-series models: TCN/LSTM/GRU and a Transformer-based time-series model.


Train for event detection: optimize for hypo_next_3_hours with event-aware metrics (more below).


Calibration & uncertainty: calibrate outputs and quantify uncertainty (reduces false negatives).


Simulation of prick policy: show tradeoff curves (pricks saved vs hypos missed).


Hardening & reproducibility: seeds, CI, clear evaluation, hyperparam search (Optuna), and crisp ablations.



Dataset & preprocessing (high impact)
Windowing: convert hourly series into sliding windows (e.g., last 1, 3, 6 hours) that predict hypo_next_3_hours. Use overlapping windows for more samples.


Target definition: keep hypo_next_3_hours as primary target. Also create time_to_hypo and end-of-window glucose as secondary targets for multitask learning.


Imputation: keep missingness as feature. Use simple forward-fill + indicator for “was missing” or learnable imputation (GRU-D, BRITS).


Feature engineering (huge payoff): rolling means/std (1h, 3h, 6h), last delta, insulin-to-carb ratio, time-since-last-meal, time-of-day sin/cos, sleep_flag, pricks_per_day, cumulative insulin in last 3h, rolling min.


Categorical encoding: one-hot or embedding for activity_event.


Normalize per-person: either global scaler or person-wise mean/std (if multiple days per person).


Class balancing: hypos are rare — use class weighting, focal loss, or oversample positive windows (SMOTE for tabular windows).



Baselines to implement (quick, essential)
Logistic regression + engineered features — fast and interpretable.


XGBoost / LightGBM on the same features — typically strong for tabular time-window data.


A simple MLP on the same features.


Beating these is necessary before complex deep models.

Deep model candidates (implement & compare)
TCN (Temporal Convolutional Network) — excellent for stable, efficient time-series modeling.


LSTM/GRU with attention — solid for shorter sequences (1–12 hours).


Transformer / TST / Informer style — try for longer context windows; limited MPS memory could be an issue.


Hybrid: CNN/TCN front-end + Transformer/attention head.


Multitask network: predict hypo_next_3_hours (classification) + glucose_next_hour (regression). Multitask often improves robustness.



Losses & training objectives
Primary: binary cross-entropy for hypo (use class weights or focal loss).


Auxiliary: MSE for glucose regression, or ordinal loss if you bucket glucose.


Event-weighting: if you care about early detection, weight samples occurring shortly before hypoglycemia more.


Metric-driven stopping: early-stop based on PR-AUC or F1 at chosen operating point, not just loss.



Evaluation metrics (choose event-aware metrics)
Primary: Precision-Recall AUC (PR-AUC) for hypo_next_3_hours (better for imbalanced data).


Secondary: ROC-AUC, F1 at operating threshold.


Operational metrics:


False negatives per 10k hours (or per patient-day).


False alarms per day (prick/re-check rate).


Time-to-detection distribution (how early before a hypo you alert).


Prick reduction vs missed hypo curve (plot tradeoff).


Calibration: Brier score + calibration curves. Poorly calibrated models are unsafe for medical alerts.



Uncertainty & safety (critical for winning judged projects)
Calibrate probabilities: Platt scaling or isotonic on validation.


Uncertainty estimates: MC dropout, deep ensembles (3–5 models), or evidential deep learning. Use uncertainty to suppress low-confidence alerts and to request a prick instead.


Conformal prediction: gives valid prediction sets under exchangeability — good for conservative clinical claims.



Training recipe & optimizations
Framework: PyTorch (good). Use torch.compile() if available for speedups.


MPS notes: PyTorch MPS backend on Macs is usable but missing some ops and can be slower for some workloads; test carefully. If possible, validate on NVIDIA GPU (cloud) for final runs.


Mixed precision: use AMP (torch.cuda.amp) for GPU; MPS also benefits from float32 but AMP support varies.


Batch size: tune — larger batch for stable gradients; use gradient accumulation if memory-limited.


LR schedule: OneCycle or CosineAnnealing + warmup.


Optimizers: AdamW, AdamP, or Lion (if available). Use weight decay.


Regularization: dropout, label smoothing, weight decay, early stopping.


Grad clipping: to prevent spikes for RNNs.


Hyperparam tuning: Optuna or Ray Tune — search LR, weight decay, depth, hidden size, window length. Prioritize learning rate, window length, and model capacity.


Checkpointing: save best by val PR-AUC and keep last N checkpoints.



Interpretability & explainability (visible wins in judges' eyes)
SHAP or Integrated Gradients for feature importance (tabular & NN).


Per-event explanations: show why model flagged a specific hour (high insulin, recent drop, exercise).


Rule-based overlay: combine model score with simple safety rules (e.g., if glucose < 65, always alert) — simple rules + ML is safer.



Deployment & latency (project polish)
Real-time inference: window -> model -> calibrate -> threshold -> notify. Keep model footprint low (distill if needed).


Edge vs cloud: small TCN/MLP can run on-device; bigger transformers run on cloud. Provide both options in repo.


Alert policy: Add hysteresis (require two consecutive high-risk windows or risk > threshold for X minutes) to reduce false alarms.


SendGrid integration: implement test mode and show logs and rate limits. For demo, simulate alerts and response time.



Experimental plan (prioritized)
Baselines: logistic + LightGBM with window features. (1–2 days)


TCN + sliding windows multitask (3–4 days).


Uncertainty: deep ensemble of 3 TCNs and calibration (2 days).


Ablations: remove features (insulin, carbs, activity) to show importance (1–2 days).


Policy simulation: show prick reduction vs missed hypos curves (2 days).


Final: Combine best model + safety rules + explainability + live demo + reproducible notebook (2–3 days).



How to demonstrate you “win” (presentation + deliverables)
Clear metric target: e.g., “Reduce daily pricks by X% while keeping missed hypos < Y per 10k hours.” Show a plot.


Baseline vs final model: table with PR-AUC, FN rate, pricks/day, mean time-to-detection.


Ablation study: show that features/uncertainty/ensembling improved core metrics.


Calibration & safety: calibration plots and an example patient timeline showing correct early alerts and lack of false alarms.


Code + notebook: single notebook that reproduces training of the baseline, evaluation, and policy simulation.


Live demo: small web app or script that feeds a patient’s last 6 hours and shows predicted risk + explanation + whether to prick.



MPS-specific tips (if you’ll train on a Mac)
Test functional parity: ensure all ops you use run on MPS. Some PyTorch ops or third-party libs (Apex/amp custom ops) may not.


Use CPU dataloader + MPS device transfers; set num_workers > 0 for dataloader but test for stability.


If training is slow or unstable, use small experiments on MPS and do heavy hyperparam sweeps on a GPU cloud instance.


Use torch.backends.mps settings and latest PyTorch for improved support.



Quick checklist you can follow now
Implement baseline logistic + LightGBM on window features.


Create sliding-window dataset and train/val/test splits by person (no leakage).


Train TCN + LSTM; compare PR-AUC and FN rate.


Implement ensemble & calibration; compute calibrated probabilities.


Build prick-policy simulator and report reduction vs missed hypos.


Add SHAP explanations and per-case visualization.


Package reproducible notebook and short video gif of an alert demo.
High-value feature ideas (with how-to & why they’re unique)
Personalized Snack/Insulin Recommendation (Actionable)


What: Given current glucose, recent carbs/insulin, and planned activity, recommend a snack size or insulin correction to prevent hypo/hyper in the next 3 hours.


Implementation: Multitask model (classification for snack/no-snack + regression for grams/units) or policy-learning (RL with safety constraints). Constrain recommendations by simple rules (e.g., never recommend >1 unit correction without human consent).


Data needed: carbs, insulin, glucose history, response curves (you can simulate individualized sensitivity from synthetic data).


Models: XGBoost/NN for initial prototype; RL or contextual bandits for adaptive policies.


Judges like: direct utility + safety rules.


Context-Aware Alerting with Uncertainty-Based Hysteresis


What: Alerts are triggered only when risk + model confidence exceed thresholds; low-confidence cases suggest “please test” instead of alerting.


Implementation: Deep ensembles + calibrated probabilities; decision logic that requires either high risk or repeated moderate risk across windows.


Data needed: same time-series + logs of corrective actions.


Benefit: fewer false alarms, safer product.


Activity Scheduler / Hypo-Avoidance Planner


What: Recommend times to exercise or eat to minimize hypo risk over the day (e.g., “If you run at 6pm, take 8g carbs beforehand”).


Implementation: Simulation-based optimization using the person’s learned glucose dynamics (digital twin) + simple optimization (grid search / model predictive control).


Novelty: Turns a model into a planner — judges love prescriptive outcomes.


Root-Cause & Counterfactual Explanations


What: When a hypo is predicted, output “Why?” and “What could have been done?” (e.g., missed insulin, late snack, unexpected exercise).


Implementation: SHAP + counterfactual generator (find minimal change to features to flip prediction).


Impact: Clinicians/participants get trust and actionable post-hoc guidance.


Adaptive Prick Policy Simulator & Optimizer


What: Learn a policy that tells when to prick to maximally reduce tests while keeping missed hypos below a threshold. Optimize for user burden + safety.


Implementation: Formulate as constrained optimization (minimize pricks subject to FN rate constraint) or contextual bandit and simulate on historical data.


Why unique: Directly ties ML value to real user burden reduction.


Multi-risk Dashboard: Short-term + Long-term Risk


What: Show short-term hypo risk and longer-term glycemic variability risk (e.g., time-in-range, predicted A1c proxy).


Implementation: Two heads on model: near-term classifier + aggregator for long-term metrics.


Benefit: Broader clinical relevance — not just emergency detection.


Wearable & Multimodal Fusion (HR, HRV, Skin Temp, Motion)


What: Fuse heart rate, HRV, accelerometer, sleep, and device-derived stress estimates to catch non-glycemic signals preceding dips (autonomic signs of hypo).


Implementation: Multimodal transformer/TCN, late fusion with gating.


Data: Requires wearable streams; you can simulate HR responses in synthetic data for demo.


Judges: real-world realism + cross-device capability.


Meal Photo → Carb Estimation Integration


What: Let user snap a meal, estimate carbs with a CV model, feed to planner/predictor.


Implementation: Off-the-shelf image model fine-tuned on food calorie datasets + simple carb-to-insulin pipeline.


Novelty: Highly consumer-facing and practical.


Federated / Privacy-Preserving Learning


What: Train models across users without centralizing their raw data.


Implementation: Simulate federated averaging; show privacy-preserving evaluation (differential privacy or secure aggregation).


Judges: addresses real deployment privacy concerns.


Clinician Triage Dashboard + Batch Review Tools


What: Aggregate patient alerts, show ranked list by risk + explainability, allow clinician to bulk-dismiss or message.


Implementation: Web dashboard, ranking model, SHAP summaries, exportable reports.


Competitive edge: product-ready, clinical workflow integration.


Behavioral Nudges & Gamification


What: Reward users for maintaining time-in-range, provide streaks, gentle nudges when risk is trending up.


Implementation: Lightweight rule-based system + UX design; A/B test simulated user retention.


Judges: shows adoption thinking.


Anomaly Detection & New-onset Detection


What: Alert on “unusual” glucose patterns that don’t match historical behavior (illness, meds change).


Implementation: Unsupervised model (autoencoder, isolation forest) on features; provide clinician flagging.


Benefit: Useful beyond hypo detection.


Digital Twin + What-if Simulator


What: Let clinician/user simulate “If I skip insulin, what happens?” or “If I run 30 min now, risk change?”


Implementation: Fit a parametric or learned dynamic model per-person, run fast rollouts.


Wow factor: build a miniature simulator in the demo.


Cross-patient Cohort Discovery & Population Insights


What: Cluster users by insulin sensitivity, carb response, activity patterns; surface cohorts for targeted interventions.


Implementation: Unsupervised clustering on response parameters; show cohort-level plots and suggested policies per cohort.


Research value: shows generalizability and deeper insight.


How to prioritize (fast wins vs big wins)
Fast wins (1–2 days): uncertainty-based alerting, adaptive prick policy simulator (on synthetic data), root-cause SHAP explanations.


Mid-term (3–7 days): personalized snack/insulin recommender, planner, and cohort discovery.


Big/novel (2+ weeks): multimodal wearable fusion, federated training demo, digital twin+simulator.


Data & modeling tradeoffs
If you only have hourly glucose, start with planner + snack recommender using person-specific parameters estimated from the hourly series.


For multimodal or meal photo features, simulate the missing modalities for judges and clearly label them synthetic — but also show how to plug in real device streams.


Safety-first: always pair recommendations with conservative rules and show those rules in your evaluation.


Evaluation & UX metrics to show judges
Clinical-style metrics: missed hypoglycemia per 10k hours, false alarms per patient-day.


Utility metrics: % reduction in pricks, average carbs recommended per avoidable hypo, time-to-detection improvement.


User metrics: reduction in alert fatigue (alerts/day), acceptability (simulated).


Ablations: show how each feature (wearables, planner, uncertainty) improves the tradeoff curve.


Demo & storytelling ideas (slide + live demo)
Live patient timeline: show 6-hour window, predicted risk, counterfactual (“if you had taken 2u insulin less, outcome would be…”), recommended snack, and whether prick is advised.


Policy simulation slide: pricks saved vs missed hypos for baseline vs your model vs model+planner.


One-click clinician report: summary for a patient-week with cohort assignment and recommendations.


Quick technical suggestions to build the features
Expose new features as engineered columns in your dataset (e.g., recommended_snack_g, recommended_insulin_u, uncertainty, digital_twin_params, cohort_label).


Use a modular codebase: data/ pipeline → models/ → policies/ → ui/ so you can mix-and-match features.


Provide safety overrides: rule-based cutoffs always trump the model for critical decisions.



