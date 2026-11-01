
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
