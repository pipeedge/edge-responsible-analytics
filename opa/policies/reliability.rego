package policies.reliability

default allow = false

allow {
    input.reliability.metrics.reliability_score >= input.reliability.threshold.reliability
}

allow {
    input.reliability.metrics.prediction_stability >= input.reliability.threshold.prediction_stability
}