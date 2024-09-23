package policies.fairness

import data.policies.fairness.demographic_parity
import data.policies.fairness.equal_opportunity

default allow = false
default failed_policies = []

# Allow if all individual fairness policies allow
allow {
    demographic_parity.allow
    equal_opportunity.allow
}

# List of failed policies
failed_policies := [policy | 
    policies := {"demographic_parity": demographic_parity.allow, 
                 "equal_opportunity": equal_opportunity.allow}
    policy := k; v := policies[k]; not v
]
