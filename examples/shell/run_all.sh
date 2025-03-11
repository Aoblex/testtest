#!/bin/bash

# Run all training scripts
for script in examples/run_*.sh; do
    if [ "$script" != "examples/run_all.sh" ]; then
        echo "Running $script..."
        bash "$script"
    fi
done 