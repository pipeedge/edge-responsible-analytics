package policies.reliability

default allow = false

allow {
    input.reliability_score >= input.thresholds.reliability
}