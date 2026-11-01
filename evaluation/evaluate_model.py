
import pandas as pd

# Load actual and predicted relationships
truth = pd.read_csv(
    "data/ground_truth/reporting_relationships.csv"
)

pred = pd.read_csv(
    "data/processed/inferred_relationships.csv"
)

# Exclude employees with no actual manager
actual = truth[truth["manager_id"].notna()].copy()

# Match actual and predicted records
data = actual.merge(
    pred,
    on="employee_id",
    how="left"
)

# Accepted predictions
accepted = data[
    data["predicted_manager_id"].notna()
]

# Correct and incorrect predictions
correct = (
    accepted["manager_id"]
    == accepted["predicted_manager_id"]
)

true_positive = int(correct.sum())
false_positive = len(accepted) - true_positive
false_negative = len(data) - true_positive

# Metrics
precision = (
    true_positive / (true_positive + false_positive)
    if true_positive + false_positive > 0 else 0
)

recall = (
    true_positive / (true_positive + false_negative)
    if true_positive + false_negative > 0 else 0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall > 0 else 0
)

coverage = (
    len(accepted) / len(data)
    if len(data) > 0 else 0
)

print("\n========== MODEL EVALUATION ==========")

print("Total actual relationships:", len(data))
print("Accepted predictions:", len(accepted))
print("Correct predictions:", true_positive)
print("Incorrect predictions:", false_positive)
print("Unresolved relationships:", len(data) - len(accepted))

print("\nPrecision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("Coverage:", round(coverage, 4))

# Save final results
results = pd.DataFrame([{
    "total_actual_relationships": len(data),
    "accepted_predictions": len(accepted),
    "correct_predictions": true_positive,
    "incorrect_predictions": false_positive,
    "unresolved_relationships": len(data) - len(accepted),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1, 4),
    "coverage": round(coverage, 4)
}])

results.to_csv(
    "evaluation/evaluation_results.csv",
    index=False
)

print("\nEvaluation results saved successfully!")
