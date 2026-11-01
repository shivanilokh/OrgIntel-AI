
import pandas as pd
import os

MIN_CONFIDENCE = 0.80
MIN_MARGIN = 0.05

# 1. Load complete employee dataset
employees = pd.read_csv("data/raw/employees.csv")

print("Employees loaded:", len(employees))
print("Companies loaded:", employees["company_id"].nunique())
print("Total employees:", len(employees))

print("\nAvailable columns:")
print(employees.columns.tolist())


# 2. Normalize job titles
TITLE_MAPPING = {
    "CEO": "Chief Executive Officer",
    "CTO": "Chief Technology Officer",
    "CFO": "Chief Financial Officer",
    "COO": "Chief Operating Officer",
    "CRO": "Chief Revenue Officer",
    "CMO": "Chief Marketing Officer",
    "CHRO": "Chief Human Resources Officer",
    "SWE": "Software Engineer",
    "Sr Data Analyst": "Senior Data Analyst",
    "Sr. Data Analyst": "Senior Data Analyst",
    "VP Engg": "VP Engineering",
    "Head - Data": "Head of Data & Analytics"
}


def normalize_title(title):
    if pd.isna(title):
        return "Unknown"

    title = str(title).strip()
    return TITLE_MAPPING.get(title, title)


employees["normalized_title"] = (
    employees["job_title"].apply(normalize_title)
)

print("\nNormalized titles:")
print(
    employees[
        ["job_title", "normalized_title"]
    ].head(20)
)


# 3. Calculate seniority level
def get_seniority(title):
    title = str(title).lower()

    if "chief executive officer" in title or title == "ceo":
        return 10
    if "chief " in title:
        return 9
    if "vice president" in title or "vp " in title:
        return 8
    if "director" in title or "head of" in title:
        return 7
    if "manager" in title:
        return 6
    if "team lead" in title or "lead" in title:
        return 5
    if "principal" in title or "staff " in title:
        return 4
    if "senior" in title or "sr " in title:
        return 3
    if title == "unknown":
        return 0

    return 2


employees["seniority_level"] = (
    employees["normalized_title"].apply(get_seniority)
)

print("\nSeniority levels:")
print(
    employees[
        ["normalized_title", "seniority_level"]
    ].head(20)
)


# 4. Identify managers
def is_manager(title):
    title = str(title).lower()

    manager_keywords = [
        "chief", "ceo", "cto", "cfo", "coo",
        "cro", "cmo", "chro", "vice president",
        "vp ", "director", "head of",
        "manager", "team lead"
    ]

    return any(word in title for word in manager_keywords)


# 5. Calculate candidate score
def calculate_score(employee, candidate):
    score = 0.0

    if (
        pd.notna(employee["function_group"])
        and employee["function_group"]
        == candidate["function_group"]
    ):
        score += 0.70

    if (
        pd.notna(employee["department"])
        and employee["department"]
        == candidate["department"]
    ):
        score += 0.15

    gap = (
        candidate["seniority_level"]
        - employee["seniority_level"]
    )

    if 1 <= gap <= 3:
        score += 0.10
    elif gap > 3:
        score += 0.03

    if is_manager(candidate["normalized_title"]):
        score += 0.05

    return round(score, 3)


# 6. Infer reporting relationships
predictions = []

# Process each company separately
for company_id, company_employees in employees.groupby("company_id"):

    print("Processing company:", company_id)

    for _, employee in company_employees.iterrows():

        if employee["normalized_title"] == "Chief Executive Officer":
            continue

        # Handle missing job titles
        if employee["normalized_title"] == "Unknown":
            predictions.append({
                "employee_id": employee["employee_id"],
                "candidate_manager_id": None,
                "predicted_manager_id": None,
                "confidence": 0,
                "margin": 0,
                "status": "Unresolved"
            })
            continue

        # Find senior employees in the same company
        candidates = company_employees.loc[
            (company_employees["employee_id"] != employee["employee_id"]) &
            (company_employees["seniority_level"] > employee["seniority_level"])
        ].copy()

        if candidates.empty:
            predictions.append({
                "employee_id": employee["employee_id"],
                "candidate_manager_id": None,
                "predicted_manager_id": None,
                "confidence": 0,
                "margin": 0,
                "status": "Unresolved"
            })
            continue

        # Special case: Executives report to CEO
        if employee["department"] == "Executive":

            ceo_candidates = candidates[
                candidates["normalized_title"]
                == "Chief Executive Officer"
            ]

            if not ceo_candidates.empty:
                ceo = ceo_candidates.iloc[0]

                predictions.append({
                    "employee_id": employee["employee_id"],
                    "candidate_manager_id": ceo["employee_id"],
                    "predicted_manager_id": ceo["employee_id"],
                    "confidence": 0.98,
                    "margin": 1.0,
                    "status": "Inferred"
                })

            else:
                predictions.append({
                    "employee_id": employee["employee_id"],
                    "candidate_manager_id": None,
                    "predicted_manager_id": None,
                    "confidence": 0,
                    "margin": 0,
                    "status": "Unresolved"
                })

            continue

        # Keep only manager candidates
        candidates = candidates[
            candidates["normalized_title"].apply(is_manager)
        ]

        if candidates.empty:
            predictions.append({
                "employee_id": employee["employee_id"],
                "candidate_manager_id": None,
                "predicted_manager_id": None,
                "confidence": 0,
                "margin": 0,
                "status": "Unresolved"
            })
            continue

        # Calculate candidate scores
        candidates["score"] = candidates.apply(
            lambda candidate: calculate_score(
                employee, candidate
            ),
            axis=1
        )

        # Prefer candidates from the same team
        if pd.notna(employee["team"]):

            team_candidates = candidates[
                candidates["team"] == employee["team"]
            ]

            if not team_candidates.empty:
                candidates = team_candidates.copy()

        # Sort candidates by score
        candidates = candidates.sort_values(
            "score",
            ascending=False
        )

        best = candidates.iloc[0]
        best_score = best["score"]

        second_score = (
            candidates.iloc[1]["score"]
            if len(candidates) > 1
            else 0
        )

        margin = best_score - second_score

        # Accept or reject prediction
        is_accepted = (
            best_score >= MIN_CONFIDENCE
            and margin >= MIN_MARGIN
        )

        predictions.append({
            "employee_id": employee["employee_id"],
            "candidate_manager_id": best["employee_id"],
            "predicted_manager_id": (
                best["employee_id"] if is_accepted else None
            ),
            "confidence": best_score,
            "margin": margin,
            "status": (
                "Inferred" if is_accepted else "Unresolved"
            )
        })


# 7. Save predictions
predictions_df = pd.DataFrame(predictions)

os.makedirs("data/processed", exist_ok=True)

predictions_df.to_csv(
    "data/processed/inferred_relationships.csv",
    index=False
)

print("\n========== INFERENCE COMPLETED ==========")

print("\nPrediction summary:")
print(predictions_df["status"].value_counts())

print("\nTotal predictions:", len(predictions_df))

print("\nOutput saved to:")
print("data/processed/inferred_relationships.csv")

print("\nSample predictions:")
print(predictions_df.head(15))
