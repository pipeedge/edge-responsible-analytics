package policies.fairness

import data.policies.fairness.demographic_parity
import data.policies.fairness.equal_opportunity

default allow = false
default failed_policies = []

allow {
    demographic_parity.allow
    equal_opportunity.allow
}

failed_policies = [policy | not data.policies.fairness[policy].allow]