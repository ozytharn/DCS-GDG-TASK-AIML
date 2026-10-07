"""Student risk detection pipeline for the AI/ML recruitment task.

The dataset is a synthetic student dataset downloaded from Kaggle. The task asks to:
- explore the data,
- preprocess it,
- build a model that identifies students at risk,
- evaluate the model, and
- generate a final recommendation report.

This module intentionally avoids external ML libraries so it runs reliably on the
current environment even when scipy/sklearn import is blocked by local policy.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_PATH = Path(__file__).resolve().parent / "data" / "student_dataset" / "dcs_student_data.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the Kaggle student dataset and perform the first cleaning pass."""
    df = pd.read_csv(path)

    # Convert relevant fields to numeric for analysis and model training.
    numeric_columns = [
        "Age",
        "Attendance (%)",
        "Midterm_Score",
        "Final_Score",
        "Assignments_Avg",
        "Quizzes_Avg",
        "Participation_Score",
        "Projects_Score",
        "Total_Score",
        "math_score",
        "reading_score",
        "writing_score",
        "science_score",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Replace impossible values (negative or above-range scores) with the median.
    invalid_ranges = {
        "Age": (15, 60),
        "Attendance (%)": (0, 100),
        "Midterm_Score": (0, 100),
        "Final_Score": (0, 100),
        "Assignments_Avg": (0, 100),
        "Quizzes_Avg": (0, 100),
        "Participation_Score": (0, 10),
        "Projects_Score": (0, 100),
        "Total_Score": (0, 100),
        "math_score": (0, 100),
        "reading_score": (0, 100),
        "writing_score": (0, 100),
        "science_score": (0, 100),
    }

    for column, (lower, upper) in invalid_ranges.items():
        if column in df.columns:
            df.loc[(df[column] < lower) | (df[column] > upper), column] = np.nan
            df[column] = df[column].fillna(df[column].median())

    # Make category strings consistent for downstream encoding.
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip()
    if "Department" in df.columns:
        df["Department"] = df["Department"].astype(str).str.strip()
    if "Grade" in df.columns:
        df["Grade"] = df["Grade"].astype(str).str.strip()

    return df


def derive_risk_label(df: pd.DataFrame) -> pd.Series:
    """Create a multi-class target using attendance and academic performance.

    Using custom thresholds is useful here because the dataset does not contain a direct
    "risk" column. The labels are defined as:
    - High Risk: poor attendance or weak overall performance.
    - Medium Risk: borderline attendance or mixed performance.
    - Low Risk: healthy attendance and strong academic standing.
    """
    average_score = (df["Total_Score"] + df["Midterm_Score"] + df["Final_Score"]) / 3
    attendance = df["Attendance (%)"]

    risk = np.select(
        [
            (attendance < 60) | (average_score < 55),
            (attendance < 80) | (average_score < 75),
        ],
        ["High", "Medium"],
        default="Low",
    )
    return pd.Series(risk, index=df.index, name="Risk")


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare a model-ready matrix and target labels from the dataset."""
    df = df.copy()
    df["Risk"] = derive_risk_label(df)

    # The task emphasizes the relationship between attendance and academic marks, so
    # the most important features are the performance signals and preparation indicators.
    feature_columns = [
        "Age",
        "Attendance (%)",
        "Midterm_Score",
        "Final_Score",
        "Assignments_Avg",
        "Quizzes_Avg",
        "Participation_Score",
        "Projects_Score",
        "Total_Score",
        "Gender",
        "Department",
        "test_preparation_course",
        "math_score",
        "reading_score",
        "writing_score",
        "science_score",
    ]

    X = df[feature_columns].copy()
    y = df["Risk"].copy()

    # One-hot encoding for categories while keeping the numeric signal intact.
    categorical = ["Gender", "Department"]
    X_encoded = pd.get_dummies(X, columns=categorical, dtype=float)
    return X_encoded, y


def standardize(
    X: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features using supplied training statistics, or compute them."""
    values = np.asarray(X, dtype=float)
    if mean is None:
        mean = values.mean(axis=0)
    if std is None:
        std = values.std(axis=0)
    std[std == 0] = 1.0
    return (values - mean) / std, mean, std


def stratified_train_test_split(y: pd.Series, test_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Create a stable class-preserving train/test split without scikit-learn."""
    rng = np.random.default_rng(42)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for label in sorted(y.unique()):
        label_indices = np.where(y.to_numpy() == label)[0]
        rng.shuffle(label_indices)
        split_point = max(1, int(len(label_indices) * (1 - test_fraction)))
        train_indices.extend(label_indices[:split_point].tolist())
        test_indices.extend(label_indices[split_point:].tolist())

    return np.array(train_indices), np.array(test_indices)


class SoftmaxRegression:
    """Multi-class logistic regression implemented from scratch for reliable training."""

    def __init__(self, learning_rate: float = 0.1, epochs: int = 600, random_state: int = 42) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.weights_ = None
        self.bias_ = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftmaxRegression":
        rng = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y)
        class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
        y_indices = np.array([class_to_index[value] for value in y], dtype=int)

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self.weights_ = rng.normal(0, 0.01, size=(n_features, n_classes))
        self.bias_ = np.zeros(n_classes, dtype=float)

        for _ in range(self.epochs):
            logits = X @ self.weights_ + self.bias_
            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            one_hot = np.zeros_like(probabilities)
            one_hot[np.arange(n_samples), y_indices] = 1.0

            loss_gradient = (probabilities - one_hot) / n_samples
            self.weights_ -= self.learning_rate * (X.T @ loss_gradient)
            self.bias_ -= self.learning_rate * loss_gradient.sum(axis=0)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.weights_ + self.bias_
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> np.ndarray:
    """Build a confusion matrix for the multi-class classifier."""
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        actual_index = labels.index(actual)
        predicted_index = labels.index(predicted)
        matrix[actual_index, predicted_index] += 1
    return matrix


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Compute per-class precision, recall, and F1-score."""
    precision = {}
    recall = {}
    f1 = {}

    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))

        precision_value = tp / (tp + fp) if (tp + fp) else 0.0
        recall_value = tp / (tp + fn) if (tp + fn) else 0.0
        f1_value = 2 * precision_value * recall_value / (precision_value + recall_value) if (precision_value + recall_value) else 0.0

        precision[label] = precision_value
        recall[label] = recall_value
        f1[label] = f1_value

    return precision, recall, f1


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, object]:
    """Return standard metrics for the student-risk model."""
    accuracy = float(np.mean(y_true == y_pred))
    cm = compute_confusion_matrix(y_true, y_pred, labels)
    precision, recall, f1 = precision_recall_f1(y_true, y_pred, labels)

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "labels": labels,
    }


def plot_attendance_vs_risk(df: pd.DataFrame, output_path: Path) -> None:
    """Create a chart showing the relationship between attendance and academic risk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    risk_order = ["Low", "Medium", "High"]
    colors = {"Low": "#2ca02c", "Medium": "#ffb000", "High": "#d62728"}

    plt.figure(figsize=(10, 6))
    for risk in risk_order:
        subset = df[df["Risk"] == risk]
        plt.scatter(
            subset["Attendance (%)"],
            subset["Total_Score"],
            s=25,
            c=colors[risk],
            alpha=0.8,
            label=risk,
        )

    plt.title("Attendance vs Overall Score by Risk Level")
    plt.xlabel("Attendance (%)")
    plt.ylabel("Total Score")
    plt.legend(title="Risk")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_recommendation(student: pd.Series) -> str:
    """Create a readable recommendation for a specific student using the model rules."""
    _ = float(student["Attendance (%)"])
    _ = float(student["Total_Score"])
    risk = str(student["Risk"]).upper()

    if risk == "HIGH":
        return "Improve attendance, schedule regular study blocks, and focus on the next assessments to prevent academic decline."
    if risk == "MEDIUM":
        return "Maintain current effort while improving attendance and strengthening weaker subjects to avoid a future drop in performance."
    return "Continue the current pattern and keep using the preparation course to sustain a strong academic trajectory."


def generate_report(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Create a final report table matching the task's required format."""
    report = df[["Student_ID", "Attendance (%)", "Total_Score", "Risk"]].copy()
    report["Student"] = report["Student_ID"]
    report["Attendance"] = report["Attendance (%)"].map(lambda value: f"{value:.0f}%")
    report["Risk"] = report["Risk"].str.upper()
    report["Recommendation"] = report.apply(build_recommendation, axis=1)
    final = report[["Student", "Attendance", "Risk", "Recommendation"]].copy()
    final.to_csv(output_path, index=False)
    return final


def save_metrics(metrics: dict[str, object], output_path: Path) -> None:
    """Persist the evaluation metrics in a readable, machine-friendly JSON file."""
    serializable = {
        "accuracy": metrics["accuracy"],
        "labels": metrics["labels"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "confusion_matrix_label_order": metrics["labels"],
    }
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def run_pipeline() -> dict[str, object]:
    """Run the full workflow for exploration, preprocessing, training, evaluation, and reporting."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATASET_PATH)
    df["Risk"] = derive_risk_label(df)

    # Data exploration insight block.
    correlation = df["Attendance (%)"].corr(df["Total_Score"])
    risk_summary = df.groupby("Risk").agg(
        students=("Student_ID", "count"),
        mean_attendance=("Attendance (%)", "mean"),
        mean_total_score=("Total_Score", "mean"),
    )

    X, y = prepare_features(df)
    X_converted = X.to_numpy(dtype=float)
    train_idx, test_idx = stratified_train_test_split(y)
    X_train_norm, train_mean, train_std = standardize(X_converted[train_idx])
    X_norm = (X_converted - train_mean) / train_std

    # Keep this evaluation model separate from the final model used to generate
    # the all-student report, so the reported test metrics remain holdout metrics.
    model = SoftmaxRegression(learning_rate=0.1, epochs=500, random_state=42)
    model.fit(X_train_norm, y.to_numpy()[train_idx])

    y_train_pred = model.predict(X_train_norm)
    y_test_pred = model.predict(X_norm[test_idx])
    y_true_train = y.to_numpy()[train_idx]
    y_true_test = y.to_numpy()[test_idx]

    train_metrics = evaluate_model(y_true_train, y_train_pred, ["Low", "Medium", "High"])
    test_metrics = evaluate_model(y_true_test, y_test_pred, ["Low", "Medium", "High"])

    # Refit on all labelled rows only after holdout evaluation, then predict every row
    # for the deliverable report. The target itself is the custom rule, not observed
    # future intervention outcomes; see README for this important limitation.
    final_model = SoftmaxRegression(learning_rate=0.1, epochs=500, random_state=42)
    final_model.fit(X_norm, y.to_numpy())
    report_df = df.copy()
    report_df["Risk"] = final_model.predict(X_norm)

    plot_attendance_vs_risk(df, OUTPUT_DIR / "attendance_vs_risk.png")
    report = generate_report(report_df, OUTPUT_DIR / "student_risk_report.csv")
    metrics_path = OUTPUT_DIR / "evaluation_metrics.json"
    save_metrics(test_metrics, metrics_path)
    top_student = report_df[report_df["Risk"] == "High"].sort_values("Attendance (%)").iloc[0]

    summary = {
        "df": df,
        "risk_summary": risk_summary,
        "correlation": correlation,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "report": report,
        "top_student": top_student,
        "metrics_path": metrics_path,
    }
    return summary


def print_summary(summary: dict[str, object]) -> None:
    """Print the findings and evaluation in the format requested by the task."""
    print("Dataset shape:", summary["df"].shape)
    print("\nRisk summary:")
    print(summary["risk_summary"].to_string())
    print(f"\nAttendance vs total score correlation: {summary['correlation']:.4f}")
    print("\nTest set accuracy:", round(summary["test_metrics"]["accuracy"], 4))
    for metric in ("precision", "recall", "f1"):
        values = summary["test_metrics"][metric]
        macro_average = sum(values.values()) / len(values)
        print(f"Test macro {metric}: {macro_average:.4f}")
        print(f"Test {metric} by risk:", {risk: round(score, 4) for risk, score in values.items()})
    print("Test confusion matrix:\n", summary["test_metrics"]["confusion_matrix"])
    print("\nSample prediction:")
    student = summary["top_student"]
    print(f"Student: {student['Student_ID']}")
    print(f"Attendance: {student['Attendance (%)']:.0f}%")
    print(f"Marks: {student['Total_Score']:.0f}")
    print(f"Risk: {student['Risk'].upper()}")
    print(f"Recommendation: {build_recommendation(student)}")
    print("\nReport saved to:", OUTPUT_DIR / "student_risk_report.csv")
    print("Evaluation metrics saved to:", summary["metrics_path"])
    print("Visualization saved to:", OUTPUT_DIR / "attendance_vs_risk.png")


if __name__ == "__main__":
    print_summary(run_pipeline())
