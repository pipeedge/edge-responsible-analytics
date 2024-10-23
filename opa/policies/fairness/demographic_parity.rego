package policies.fairness.demographic_parity

default allow = false

allow {
    input.fairness.metrics.demographic_parity_difference <= input.fairness.threshold.demographic_parity_difference
    input.fairness.metrics.accuracy >= input.fairness.threshold.accuracy
}
