# Student Risk Prediction Project

This project builds a student-risk classification model using the Kaggle student dataset.

## What it does
- Explores the dataset and identifies attendance-performance patterns.
- Cleans the data and prepares model-ready features.
- Defines a custom multi-class risk label: Low, Medium, High.
- Trains a softmax logistic-regression model from scratch.
- Evaluates using accuracy, confusion matrix, precision, recall, and F1-score.
- Uses a stratified train/test split and fits feature scaling on training rows only.
- Produces a model-prediction report with student-specific recommendations.
- Saves evaluation metrics as JSON and an attendance/score visualization as PNG.

## Risk criteria
Since the dataset has no observed intervention outcome or risk label, the project
defines a transparent proxy target using the average of `Total_Score`,
`Midterm_Score`, and `Final_Score`:
- **High:** attendance below 60% or average score below 55.
- **Medium:** otherwise, attendance below 80% or average score below 75.
- **Low:** attendance at least 80% and average score at least 75.

The model is evaluated against these rule-generated labels. Therefore, the reported
metrics measure how well logistic regression reproduces the chosen criteria; they
do **not** demonstrate that the model predicts future academic failure or that an
intervention will improve outcomes. A validated model for those claims would need
historical outcomes (for example, course completion or intervention records).

## Run the project
```bash
python main.py
```

## Outputs
- `outputs/attendance_vs_risk.png`
- `outputs/student_risk_report.csv`
- `outputs/evaluation_metrics.json`

The program prints the dataset/risk summary, attendance-to-score correlation,
test accuracy, macro and per-risk precision/recall/F1, confusion matrix, and one
sample prediction to the terminal.

## Dataset handling
The script expects the CSV at `data/student_dataset/dcs_student_data.csv`.
The dataset CSV from the Kaggle source is included at that path.

## Dataset
The dataset is downloaded from the Kaggle source included in the task brief:
https://www.kaggle.com/datasets/ganeshkumarofficial/student-dataset
