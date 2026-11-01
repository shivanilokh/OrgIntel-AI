
import pandas as pd

# Load actual relationships
truth = pd.read_csv(
    "data/ground_truth/reporting_relationships.csv"
)

# Load predictions
pred = pd.read_csv(
    "data/processed/inferred_relationships.csv"
)

# Check required prediction columns
required = [
    "employee_id",
    "candidate_manager_id",
    "confidence",
    "margin"
]

for col in required:
    if col not in pred.columns:
        raise ValueError(
            f"Missing column: {col}. "
            "Update inference_engine.py first."
        )

# Keep employees who have an actual manager
actual = truth[
    truth["manager_id"].notna()
].copy()

# Evaluate only employees processed by inference engine
actual = actual[
    actual["employee_id"].isin(pred["employee_id"])
].copy()

# Merge actual and predicted records
data = actual.merge(
    pred,
    on="employee_id",
    how="left"
)

thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

results = []

MIN_MARGIN = 0.05

total_actual = len(data)

for threshold in thresholds:

    accepted = (
        data["candidate_manager_id"].notna()
        & (data["confidence"] >= threshold)
        & (data["margin"] >= MIN_MARGIN)
    )

    correct = (
        data["candidate_manager_id"]
        == data["manager_id"]
    ) & accepted

    true_positive = int(correct.sum())

    false_positive = int(
        (accepted & ~correct).sum()
    )

    precision = (
        true_positive / int(accepted.sum())
        if accepted.sum() > 0
        else 0
    )

    recall = (
        true_positive / total_actual
        if total_actual > 0
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )

    coverage = (
        int(accepted.sum()) / total_actual
        if total_actual > 0
        else 0
    )

    results.append({
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "coverage": round(coverage, 4),
        "accepted_predictions": int(accepted.sum()),
        "correct_predictions": true_positive,
        "wrong_predictions": false_positive
    })

result_df = pd.DataFrame(results)

result_df.to_csv(
    "evaluation/threshold_comparison.csv",
    index=False
)

print("\n--- THRESHOLD COMPARISON ---")
print("Employees evaluated:", total_actual)
print(result_df.to_string(index=False))

print("\nThreshold analysis saved successfully!")
