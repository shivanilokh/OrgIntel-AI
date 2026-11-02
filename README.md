
## Day 3 - Model Evaluation

The inference engine uses a rule-based scoring
approach to identify potential reporting managers.

### Confidence Scoring
- Functional group matching
- Department matching
- Seniority difference
- Manager title detection
- Candidate ambiguity using score margin

### Threshold Tuning

Multiple confidence thresholds are evaluated
to balance precision, recall and coverage.

### Evaluation Metrics

- Precision
- Recall
- F1 Score
- Prediction Coverage

### Graph Validation

The inferred organizational graph is checked for:

- Self-reporting relationships
- Missing managers
- Cross-company reporting links
- Duplicate employee assignments
- Circular reporting relationships

### Ground Truth Evaluation

Predicted reporting relationships are compared
against the synthetic ground-truth dataset.

Low-confidence relationships are kept unresolved
rather than forcing an uncertain manager assignment.


## Project Overview

OrgIntel AI is an AI-powered organizational intelligence platform designed to infer reporting relationships between employees using job titles, departments, functional groups, seniority levels, and confidence-based scoring.

The platform converts flat employee datasets into organizational reporting structures and visualizes inferred relationships through an interactive Streamlit dashboard.

## Problem Statement

Organizations often have employee information without clearly defined reporting relationships. This project aims to infer potential reporting managers from available employee attributes while avoiding unreliable assumptions.

The system assigns confidence scores to inferred relationships and keeps uncertain relationships unresolved.

## Dataset Generation

The project uses a synthetic dataset generated using Python.

- Minimum 500 companies.
- At least 100 employees per company.
- Multiple industries and company sizes.
- Different departments, functions, and seniority levels.
- Inconsistent job titles and abbreviations.
- Missing or incomplete employee information.

The dataset includes company and employee information along with ground-truth reporting relationships for evaluation.

## Project Structure

```text
OrgIntel-AI/
│
├── app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── ground_truth/
│
├── src/
│   ├── generate_dataset.py
│   ├── verify_dataset.py
│   └── inference_engine.py
│
├── evaluation/
│   ├── evaluate_model.py
│   ├── threshold_analysis.py
│   └── evaluation_results.csv
│
├── notebooks/
├── requirements.txt
└── README.md
```

## Inference Methodology

The inference engine uses a rule-based scoring approach to identify potential reporting managers.

Candidate managers are evaluated using:

- Functional group similarity.
- Department similarity.
- Seniority differences.
- Managerial title indicators.
- Confidence score and candidate score margin.

The system selects a manager only when the confidence and ambiguity requirements are satisfied.

## Confidence Scoring

The inference engine combines multiple organizational signals to calculate candidate scores.

The current scoring logic considers functional group matching, department matching, seniority differences, and manager title indicators.

A minimum confidence threshold and score margin are used to avoid accepting ambiguous reporting relationships.

## Handling Unresolved Relationships

The system does not force reporting relationships when evidence is insufficient.

Employees without sufficiently confident manager predictions are classified as Unresolved.

This approach prioritizes reliable reporting relationships over speculative connections.

## Model Evaluation

The predicted reporting relationships are compared against the synthetic ground-truth dataset.

Evaluation includes precision, recall, F1-score, prediction coverage, and threshold comparison.

The final evaluation results will be recorded in the evaluation folder after full-dataset inference and validation.

## Key Design Decisions

- Synthetic data generation provides known ground-truth relationships.
- Rule-based scoring makes inference logic transparent.
- Confidence thresholds reduce uncertain predictions.
- Unresolved cases are retained instead of forcing incorrect reporting lines.
- Organizational charts visualize accepted inferred relationships.

## How to Run the Application

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Generate the dataset:

```bash
python src/generate_dataset.py
```

Run the dataset verification:

```bash
python src/verify_dataset.py
```

Run the inference engine:

```bash
python src/inference_engine.py
```

Launch the Streamlit application:

```bash
streamlit run app.py
```

## Live Application

**Hosted Application:** To be added after deployment.

**GitHub Repository:** Add your repository URL here.