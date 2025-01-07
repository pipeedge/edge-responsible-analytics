package policies.reliability

default allow = false

allow {
    input.reliability_score >= input.thresholds.reliability
    input.prediction_stability >= input.thresholds.prediction_stability
}
