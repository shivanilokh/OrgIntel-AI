
import pandas as pd
import networkx as nx

# Load employees
employees = pd.read_csv(
    "data/raw/employees.csv"
)

# Load inferred relationships
pred = pd.read_csv(
    "data/processed/inferred_relationships.csv"
)

# Only accepted reporting lines
edges = pred[
    pred["predicted_manager_id"].notna()
].copy()

employee_company = employees.set_index(
    "employee_id"
)["company_id"].to_dict()

# Graph creation
G = nx.DiGraph()

G.add_nodes_from(employees["employee_id"])

self_reporting = 0
missing_managers = 0
cross_company = 0

for _, row in edges.iterrows():

    employee_id = row["employee_id"]
    manager_id = row["predicted_manager_id"]

    if employee_id == manager_id:
        self_reporting += 1
        continue

    if manager_id not in employee_company:
        missing_managers += 1
        continue

    if (
        employee_company.get(employee_id)
        != employee_company.get(manager_id)
    ):
        cross_company += 1
        continue

    G.add_edge(employee_id, manager_id)

# Check duplicate reporting assignments
duplicate_managers = int(
    edges["employee_id"].duplicated().sum()
)

# Check cycles
has_cycle = not nx.is_directed_acyclic_graph(G)

cycle_count = len(list(nx.simple_cycles(G)))

# Final report
report = f"""
========== GRAPH VALIDATION ==========

Total employees: {len(employees)}

Accepted reporting lines: {len(edges)}

Self reporting errors: {self_reporting}

Missing manager errors: {missing_managers}

Cross company errors: {cross_company}

Duplicate employee assignments: {duplicate_managers}

Circular relationships detected: {cycle_count}

Graph contains cycles: {has_cycle}

Graph is valid:
{not (
    self_reporting or
    missing_managers or
    cross_company or
    duplicate_managers or
    has_cycle
)}
"""

print(report)

with open(
    "evaluation/graph_validation.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(report)

print("Graph validation report saved!")
