package policies.explainability

default allow = false

allow {
    input.explainability_score >= input.thresholds.explainability
    input.attention_score >= input.thresholds.attention
    input.interpretability_score >= input.thresholds.interpretability
}