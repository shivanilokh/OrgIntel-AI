
import pandas as pd
import random
import uuid
from faker import Faker

fake = Faker("en_IN")
random.seed(42)
Faker.seed(42)

# 1. Company industries
industries = [
    "Technology", "Banking", "Healthcare",
    "E-commerce", "Manufacturing", "Telecommunications",
    "Consulting", "Retail", "Education", "Insurance"
]

# 2. Department structure
departments = {
    "Engineering": ("Technology", "CTO"),
    "Data & Analytics": ("Technology", "CTO"),
    "Cyber Security": ("Technology", "CTO"),
    "Product": ("Technology", "CTO"),
    "Sales": ("Sales", "CRO"),
    "Marketing": ("Marketing", "CMO"),
    "Finance": ("Finance", "CFO"),
    "Human Resources": ("HR", "CHRO"),
    "Operations": ("Operations", "COO"),
    "Customer Support": ("Operations", "COO")
}

employee_titles = {
    "Engineering": ["Software Engineer", "Backend Developer",
                    "Frontend Developer", "QA Engineer",
                    "DevOps Engineer"],
    "Data & Analytics": ["Data Analyst", "Data Scientist",
                         "Data Engineer", "BI Analyst"],
    "Cyber Security": ["Security Analyst", "Security Engineer",
                       "SOC Analyst"],
    "Product": ["Product Analyst", "Product Designer",
                "Product Specialist"],
    "Sales": ["Sales Executive", "Account Executive",
              "Business Development Representative"],
    "Marketing": ["Marketing Specialist", "SEO Analyst",
                  "Content Strategist"],
    "Finance": ["Financial Analyst", "Accountant",
                "Finance Associate"],
    "Human Resources": ["HR Executive", "Recruiter",
                        "HR Analyst"],
    "Operations": ["Operations Analyst", "Operations Executive",
                   "Process Associate"],
    "Customer Support": ["Customer Support Executive",
                         "Support Analyst", "Customer Success Associate"]
}

companies = []
employees = []
ground_truth = []

# 3. Generate 500 companies
for c in range(1, 501):

    company_id = f"COMP{c:04d}"
    company_name = fake.company()

    companies.append({
        "company_id": company_id,
        "company_name": company_name,
        "industry": random.choice(industries),
        "company_size": random.choice(
            ["Small", "Medium", "Large", "Enterprise"]
        ),
        "country": random.choice(
            ["India", "USA", "UK", "Singapore", "Australia"]
        )
    })

    company_people = []

    # Helper function
    def add_employee(title, department, function_group,
                     manager_id=None, team=None):

        employee_id = str(uuid.uuid4())

        employee = {
            "employee_id": employee_id,
            "company_id": company_id,
            "name": fake.name(),
            "job_title": title,
            "department": department,
            "function_group": function_group,
            "team": team,
            "email": fake.email(),
            "location": fake.city()
        }

        company_people.append(employee)

        ground_truth.append({
            "employee_id": employee_id,
            "company_id": company_id,
            "manager_id": manager_id
        })

        return employee_id

    # 4. CEO
    ceo_id = add_employee(
        "Chief Executive Officer",
        "Executive",
        "Executive"
    )

    # 5. Executive leadership
    executives = [
        ("Chief Technology Officer", "Technology", "CTO"),
        ("Chief Revenue Officer", "Sales", "CRO"),
        ("Chief Marketing Officer", "Marketing", "CMO"),
        ("Chief Financial Officer", "Finance", "CFO"),
        ("Chief Human Resources Officer", "HR", "CHRO"),
        ("Chief Operating Officer", "Operations", "COO")
    ]

    executive_ids = {}

    for title, function, executive_key in executives:
        executive_ids[executive_key] = add_employee(
            title, "Executive", function, ceo_id
        )

    # 6. Generate department heads, managers and employees
    for department, (function, executive_key) in departments.items():

        head_id = add_employee(
            f"Head of {department}",
            department,
            function,
            executive_ids[executive_key]
        )

        for manager_number in range(2):

            team_name = f"{department} Team {manager_number + 1}"

            manager_id = add_employee(
                random.choice([
                    f"Senior Manager - {department}",
                    f"Director of {department}",
                    f"Team Lead - {department}"
                ]),
                department,
                function,
                head_id,
                team_name
            )

            # 7. Employees reporting to their manager
            for _ in range(random.randint(7, 12)):

                title = random.choice(employee_titles[department])

                if random.random() < 0.25:
                    title = "Senior " + title

                add_employee(
                    title,
                    department,
                    function,
                    manager_id,
                    team_name
                )

    employees.extend(company_people)

# 8. Create original clean dataset
clean_df = pd.DataFrame(employees)

# 9. Create messy observed dataset
messy_df = clean_df.copy()

abbreviations = {
    "Chief Executive Officer": "CEO",
    "Chief Technology Officer": "CTO",
    "Chief Financial Officer": "CFO",
    "Chief Operating Officer": "COO",
    "Chief Revenue Officer": "CRO",
    "Chief Marketing Officer": "CMO",
    "Chief Human Resources Officer": "CHRO",
    "Data Analyst": "Sr Data Analyst",
    "Software Engineer": "SWE",
    "Head of Data & Analytics": "Head - Data"
}

for index in messy_df.index:

    title = messy_df.loc[index, "job_title"]

    if title in abbreviations and random.random() < 0.45:
        messy_df.loc[index, "job_title"] = abbreviations[title]

    if random.random() < 0.05:
        messy_df.loc[index, "department"] = None

    if random.random() < 0.02:
        messy_df.loc[index, "job_title"] = None

    if random.random() < 0.03:
        messy_df.loc[index, "location"] = None

# 10. Save files
pd.DataFrame(companies).to_csv(
    "data/raw/companies.csv", index=False
)

messy_df.to_csv(
    "data/raw/employees.csv", index=False
)

pd.DataFrame(ground_truth).to_csv(
    "data/ground_truth/reporting_relationships.csv",
    index=False
)

print("DATASET GENERATION COMPLETED!")
print("Total companies:", len(companies))
print("Total employees:", len(messy_df))
print("Ground truth relationships:", len(ground_truth))
print("Companies file saved!")
print("Employees file saved!")
print("Ground truth file saved!")

