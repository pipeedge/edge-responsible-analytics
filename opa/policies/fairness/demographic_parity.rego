package policies.fairness.demographic_parity

default allow = false

allow {
    input.fairness.metrics.demographic_parity <= input.fairness.threshold.demographic_parity
}