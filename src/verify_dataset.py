
import pandas as pd

companies = pd.read_csv("data/raw/companies.csv")
employees = pd.read_csv("data/raw/employees.csv")
truth = pd.read_csv(
    "data/ground_truth/reporting_relationships.csv"
)

print("\n--- DATASET SUMMARY ---")

print("Companies:", companies["company_id"].nunique())
print("Employees:", employees["employee_id"].nunique())

print("\n--- EMPLOYEES PER COMPANY ---")

counts = employees.groupby("company_id").size()

print("Minimum employees:", counts.min())
print("Maximum employees:", counts.max())

print("\n--- MISSING VALUES ---")

print(employees.isnull().sum())

print("\n--- GROUND TRUTH ---")

print("Total relationships:", len(truth))
print("Missing managers:", truth["manager_id"].isnull().sum())

assert companies["company_id"].nunique() == 500
assert counts.min() >= 100
assert employees["employee_id"].is_unique

print("\nALL DATASET CHECKS PASSED!")
