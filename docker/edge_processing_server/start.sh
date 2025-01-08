#!/bin/bash

# Start OPA in the background
opa run --server --set=decision_logs.console=true /app/opa/policies &

# Start the aggregator
python3 aggregator.py