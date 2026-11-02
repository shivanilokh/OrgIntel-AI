
import pandas as pd
import os

# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

MIN_CONFIDENCE = 0.80
MIN_MARGIN = 0.05

INPUT_FILE = "data/raw/employees.csv"
OUTPUT_FILE = "data/processed/inferred_relationships.csv"


# --------------------------------------------------
# 2. LOAD EMPLOYEE DATA
# --------------------------------------------------

employees = pd.read_csv(INPUT_FILE)

print("========== DATASET LOADED ==========")
print("Total employees:", len(employees))
print("Total companies:", employees["company_id"].nunique())


required_columns = [
    "employee_id",
    "company_id",
    "job_title",
    "department",
    "function_group",
    "team"
]

missing_columns = [
    col for col in required_columns
    if col not in employees.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# --------------------------------------------------
# 3. NORMALIZE JOB TITLES
# --------------------------------------------------

TITLE_MAPPING = {
    "ceo": "Chief Executive Officer",
    "cto": "Chief Technology Officer",
    "cfo": "Chief Financial Officer",
    "coo": "Chief Operating Officer",
    "cro": "Chief Revenue Officer",
    "cmo": "Chief Marketing Officer",
    "chro": "Chief Human Resources Officer",
    "swe": "Software Engineer",
    "sr data analyst": "Senior Data Analyst",
    "sr. data analyst": "Senior Data Analyst",
    "vp engg": "VP Engineering",
    "head - data": "Head of Data & Analytics"
}


def normalize_title(title):

    if pd.isna(title):
        return "Unknown"

    title = str(title).strip()

    return TITLE_MAPPING.get(
        title.lower(),
        title
    )


employees["normalized_title"] = (
    employees["job_title"].apply(normalize_title)
)


# --------------------------------------------------
# 4. CALCULATE SENIORITY
# --------------------------------------------------

def get_seniority(title):

    title = str(title).lower().strip()

    if title == "unknown":
        return 0

    if "chief executive officer" in title or title == "ceo":
        return 10

    if "chief " in title:
        return 9

    if "vice president" in title or title.startswith("vp "):
        return 8

    if "director" in title or "head of" in title:
        return 7

    if "manager" in title:
        return 6

    if "team lead" in title or "lead" in title:
        return 5

    if "principal" in title or "staff " in title:
        return 4

    if "senior" in title or title.startswith("sr "):
        return 3

    return 2


employees["seniority_level"] = (
    employees["normalized_title"].apply(get_seniority)
)


# --------------------------------------------------
# 5. IDENTIFY MANAGERS
# --------------------------------------------------

def is_manager(title):

    title = str(title).lower()

    manager_keywords = [
        "chief",
        "ceo",
        "cto",
        "cfo",
        "coo",
        "cro",
        "cmo",
        "chro",
        "vice president",
        "vp ",
        "director",
        "head of",
        "manager",
        "team lead"
    ]

    return any(
        keyword in title
        for keyword in manager_keywords
    )


# --------------------------------------------------
# 6. HELPER FUNCTIONS
# --------------------------------------------------

def is_missing(value):

    return pd.isna(value)


def same_value(value1, value2):

    if is_missing(value1) or is_missing(value2):
        return False

    return str(value1).strip().lower() == str(value2).strip().lower()


def calculate_score(employee, candidate):

    score = 0.0

    # Same functional group
    if same_value(
        employee["function_group"],
        candidate["function_group"]
    ):
        score += 0.70

    # Same department
    if same_value(
        employee["department"],
        candidate["department"]
    ):
        score += 0.15

    # Seniority difference
    gap = (
        candidate["seniority_level"]
        - employee["seniority_level"]
    )

    if 1 <= gap <= 3:
        score += 0.10

    elif gap > 3:
        score += 0.03

    # Manager title
    if is_manager(candidate["normalized_title"]):
        score += 0.05

    return round(score, 3)


def unresolved_record(employee):

    return {
        "employee_id": employee["employee_id"],
        "candidate_manager_id": None,
        "predicted_manager_id": None,
        "confidence": 0.0,
        "margin": 0.0,
        "status": "Unresolved"
    }


# --------------------------------------------------
# 7. INFER REPORTING RELATIONSHIPS
# --------------------------------------------------

predictions = []

print("\n========== INFERENCE STARTED ==========")

# Missing company IDs cannot safely be matched
missing_company = employees[
    employees["company_id"].isna()
]

for _, employee in missing_company.iterrows():

    if employee["normalized_title"] != "Chief Executive Officer":
        predictions.append(unresolved_record(employee))


# Process each company separately
valid_employees = employees[
    employees["company_id"].notna()
]

company_groups = valid_employees.groupby("company_id")

total_companies = len(company_groups)

for company_number, (company_id, company_data) in enumerate(
    company_groups,
    start=1
):

    company_records = company_data.to_dict("records")

    # Preselect managers once per company
    manager_pool = [
        person
        for person in company_records
        if is_manager(person["normalized_title"])
    ]

    # Identify CEO candidates
    ceo_candidates = [
        person
        for person in company_records
        if person["normalized_title"] == "Chief Executive Officer"
    ]

    print(
        f"Processing company {company_number}/{total_companies}: "
        f"{company_id}"
    )

    for employee in company_records:

        title = employee["normalized_title"]

        # CEO is the root of the reporting structure
        if title == "Chief Executive Officer":
            continue

        # Unknown titles cannot be confidently assigned
        if title == "Unknown":
            predictions.append(
                unresolved_record(employee)
            )
            continue

        # Executive employees report to the CEO
        if same_value(employee["department"], "Executive"):

            if ceo_candidates:

                ceo = ceo_candidates[0]

                predictions.append({
                    "employee_id": employee["employee_id"],
                    "candidate_manager_id": ceo["employee_id"],
                    "predicted_manager_id": ceo["employee_id"],
                    "confidence": 0.98,
                    "margin": 1.0,
                    "status": "Inferred"
                })

            else:

                predictions.append(
                    unresolved_record(employee)
                )

            continue

        # Find managers in the same company
        # who are more senior than the employee
        candidates = [
            candidate
            for candidate in manager_pool
            if (
                candidate["employee_id"] != employee["employee_id"]
                and candidate["seniority_level"] > employee["seniority_level"]
            )
        ]

        if not candidates:

            predictions.append(
                unresolved_record(employee)
            )
            continue

        # Calculate candidate scores
        scored_candidates = []

        for candidate in candidates:

            score = calculate_score(
                employee,
                candidate
            )

            scored_candidates.append({
                "employee_id": candidate["employee_id"],
                "score": score,
                "team": candidate["team"]
            })

        # Prefer same-team managers when available
        same_team_candidates = [
            candidate
            for candidate in scored_candidates
            if same_value(
                employee["team"],
                candidate["team"]
            )
        ]

        if same_team_candidates:
            scored_candidates = same_team_candidates

        # Sort by score, highest first
        scored_candidates.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        best = scored_candidates[0]

        best_score = best["score"]

        second_score = (
            scored_candidates[1]["score"]
            if len(scored_candidates) > 1
            else 0.0
        )

        margin = round(
            best_score - second_score,
            3
        )

        # Accept only sufficiently strong predictions
        is_accepted = (
            best_score >= MIN_CONFIDENCE
            and margin >= MIN_MARGIN
        )

        predictions.append({

            "employee_id": employee["employee_id"],

            "candidate_manager_id": best["employee_id"],

            "predicted_manager_id": (
                best["employee_id"]
                if is_accepted
                else None
            ),

            "confidence": best_score,

            "margin": margin,

            "status": (
                "Inferred"
                if is_accepted
                else "Unresolved"
            )
        })


# --------------------------------------------------
# 8. SAVE PREDICTIONS
# --------------------------------------------------

predictions_df = pd.DataFrame(predictions)

os.makedirs(
    "data/processed",
    exist_ok=True
)

predictions_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# 9. VALIDATE PREDICTED RELATIONSHIPS
# --------------------------------------------------

employee_lookup = employees.set_index(
    "employee_id"
)

inferred = predictions_df[
    predictions_df["status"] == "Inferred"
].copy()

self_reporting = 0
cross_company_links = 0
invalid_seniority = 0
missing_managers = 0

for _, row in inferred.iterrows():

    employee_id = row["employee_id"]
    manager_id = row["predicted_manager_id"]

    if employee_id == manager_id:
        self_reporting += 1
        continue

    if (
        manager_id not in employee_lookup.index
        or employee_id not in employee_lookup.index
    ):
        missing_managers += 1
        continue

    employee_data = employee_lookup.loc[employee_id]
    manager_data = employee_lookup.loc[manager_id]

    if employee_data["company_id"] != manager_data["company_id"]:
        cross_company_links += 1

    if (
        manager_data["seniority_level"]
        <= employee_data["seniority_level"]
    ):
        invalid_seniority += 1


# --------------------------------------------------
# 10. FINAL SUMMARY
# --------------------------------------------------

print("\n========== INFERENCE COMPLETED ==========")

print("Total employees loaded:", len(employees))

print("Total companies processed:", total_companies)

print("\nPrediction summary:")

print(
    predictions_df["status"].value_counts()
)

print("\nTotal predictions:", len(predictions_df))

print("\nValidation summary:")

print("Self-reporting relationships:", self_reporting)

print("Cross-company relationships:", cross_company_links)

print("Invalid seniority relationships:", invalid_seniority)

print("Missing managers:", missing_managers)

print("\nOutput saved to:")
print(OUTPUT_FILE)

print("\nSample predictions:")

print(
    predictions_df.head(15).to_string(index=False)
)
