package policies.explainability

default allow = false

allow {
    input.explainability_score >= input.thresholds.explainability
}