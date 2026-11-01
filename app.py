import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="OrgIntel AI", page_icon="🏢", layout="wide")
BASE_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_data():
    companies = pd.read_csv(BASE_DIR / "data/raw/companies.csv")
    employees = pd.read_csv(BASE_DIR / "data/raw/employees.csv")
    predictions = pd.read_csv(BASE_DIR / "data/processed/inferred_relationships.csv")
    return companies, employees, predictions

companies, employees, predictions = load_data()

st.title("🏢 OrgIntel AI")
st.subheader("AI-Powered Organizational Intelligence Platform")
st.write("Explore organizational reporting structures using employee data and confidence-based inference.")
st.divider()

st.sidebar.title("Company Explorer")
company_options = sorted(employees["company_id"].dropna().unique())
selected_company = st.sidebar.selectbox("Select Company", company_options)

company_employees = employees[employees["company_id"] == selected_company].copy()
company_predictions = predictions[predictions["employee_id"].isin(company_employees["employee_id"])].copy()

company_data = company_employees.merge(company_predictions[["employee_id","predicted_manager_id","confidence","status"]], on="employee_id", how="left")
company_data["status"] = company_data["status"].fillna("Unresolved")
company_data["department"] = company_data["department"].fillna("Unknown")
company_data["name"] = company_data["name"].fillna("Unknown Employee")
company_data["job_title"] = company_data["job_title"].fillna("Unknown Title")

total_employees = len(company_data)
inferred_count = int((company_data["status"] == "Inferred").sum())
unresolved_count = int((company_data["status"] == "Unresolved").sum())

st.subheader("Company Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Employees", total_employees)
col2.metric("Inferred Relationships", inferred_count)
col3.metric("Unresolved Relationships", unresolved_count)
st.divider()

st.subheader("Organizational Chart Explorer")
department_options = sorted(company_data["department"].astype(str).unique())
selected_department = st.selectbox("Select Department", ["All Departments"] + department_options)

if selected_department == "All Departments":
    department_employees = company_data.copy()
else:
    department_employees = company_data[company_data["department"].astype(str) == selected_department].copy()

st.caption(f"Selected company: {selected_company} | Department: {selected_department}")

# --- NEW COLORED CHART LIKE YOUR IMAGE ---
inferred_employees = department_employees[(department_employees["status"] == "Inferred") & (department_employees["predicted_manager_id"].notna())].copy()

if inferred_employees.empty:
    st.warning("No inferred reporting relationships found for this department.")
else:
    employee_lookup = company_data.set_index("employee_id", drop=False)
    included_ids = set(inferred_employees["employee_id"].tolist())
    pending_ids = list(included_ids)
    while pending_ids:
        cur = pending_ids.pop()
        if cur not in employee_lookup.index: continue
        mid = employee_lookup.loc[cur, "predicted_manager_id"]
        if pd.notna(mid) and mid in employee_lookup.index and mid not in included_ids:
            included_ids.add(mid)
            pending_ids.append(mid)

    children = {}
    edges = []
    for _, r in inferred_employees.iterrows():
        if r["predicted_manager_id"] in included_ids:
            edges.append((r["predicted_manager_id"], r["employee_id"]))
            children.setdefault(r["predicted_manager_id"], []).append(r["employee_id"])
    
    roots = [eid for eid in included_ids if eid not in {e for m,e in edges}]
    
    levels = {}
    queue = [(root, 0) for root in roots]
    visited = set()
    node_level = {}
    while queue:
        node, lvl = queue.pop(0)
        if node in visited: continue
        visited.add(node)
        node_level[node] = lvl
        levels.setdefault(lvl, []).append(node)
        for ch in children.get(node, []):
            if ch not in visited: queue.append((ch, lvl+1))

    def get_style(level, title):
        t = str(title).lower()
        if level==0 or "chief" in t or "ceo" in t: return "#FDF2FF", "#C084FC", "#6B21A8"
        if level==1 or "director" in t: return "#E6FFF4", "#10B981", "#065F46"
        if level==2 or ("manager" in t and "lead" not in t): return "#FFF1E6", "#FB923C", "#9A3412"
        if "lead" in t: return "#EDE9FE", "#8B5CF6", "#4C1D95"
        return "#FFE4E6", "#FB7185", "#9F1239"

    html = """
    <style>
    .org-wrap{font-family:Arial; background:white; padding:20px; border-radius:12px; border:1px solid #E5E7EB;}
    .org-level{display:flex; justify-content:center; gap:24px; margin:18px 0; flex-wrap:wrap;}
    .org-card{border-radius:10px; padding:12px 16px; min-width:160px; max-width:200px; text-align:left; box-shadow:0 1px 3px rgba(0,0,0,0.08); border:2px solid;}
    .org-name{font-weight:700; font-size:13px; color:#111827; margin-bottom:2px;}
    .org-title{font-size:11.5px; color:#4B5563;}
    .org-line{height:24px; width:2px; background:#111827; margin:0 auto;}
    </style>
    <div class="org-wrap">
    """

    sorted_levels = sorted(levels.keys())
    for lvl in sorted_levels:
        html += '<div class="org-level">'
        for eid in levels[lvl]:
            row = employee_lookup.loc[eid]
            bg, border, _ = get_style(node_level.get(eid,5), row["job_title"])
            html += f'<div class="org-card" style="background:{bg}; border-color:{border};"><div class="org-name">{row["name"]}</div><div class="org-title">{row["job_title"]}</div></div>'
        html += '</div>'
        if lvl != sorted_levels[-1]:
            html += '<div style="display:flex; justify-content:center;"><div class="org-line"></div></div>'

    html += "</div>"
    
    st.markdown(html, unsafe_allow_html=True)
    st.caption("Color coding: Purple=CEO, Green=Directors, Orange=Managers, Violet=Leads, Pink=Executives. Matches your reference image.")

st.divider()
st.subheader("Employee Directory")
search = st.text_input("Search employee by name")
employee_view = department_employees.copy()
if search:
    employee_view = employee_view[employee_view["name"].astype(str).str.contains(search, case=False, na=False)]
st.dataframe(employee_view[["employee_id","name","job_title","department","predicted_manager_id","confidence","status"]], use_container_width=True, hide_index=True)

st.divider()
st.subheader("Unresolved Relationships")
unresolved_data = company_data[company_data["status"] == "Unresolved"].copy()
st.write("These employees do not have an accepted manager prediction.")
st.metric("Unresolved Employees", len(unresolved_data))
st.dataframe(unresolved_data[["name","job_title","department","confidence","status"]], use_container_width=True, hide_index=True)
st.divider()
st.caption("OrgIntel AI | Organizational Intelligence and Confidence-Based Reporting Inference")