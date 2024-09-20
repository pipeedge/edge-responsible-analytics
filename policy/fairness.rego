package model

default allow = false

# Define the policy for model evaluation
allow {
    input.fairness >= 0.8
}