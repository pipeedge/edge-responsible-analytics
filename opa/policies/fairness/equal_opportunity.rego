package policies.fairness.equal_opportunity

default allow = false

allow {
    input.fairness.metrics.equal_opportunity <= input.fairness.threshold.equal_opportunity
}