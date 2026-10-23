#!/bin/bash

# Script for running max_cut_qaoa for all methods from MIN_QUBITS to MAX_QUBITS

MIN_QUBITS=4
MAX_QUBITS=6

METHODS=("polypus_cunqa" "polypus_local")

echo "Running max_cut_qaoa experiments from $MIN_QUBITS to $MAX_QUBITS qubits"
echo "Methods: ${METHODS[@]}"
echo ""

for method in "${METHODS[@]}"; do
    echo "=========================================="
    echo "Running experiments for method: $method"
    echo "=========================================="
    
    for qubits in $(seq $MIN_QUBITS $MAX_QUBITS); do
        echo "Running max_cut_qaoa with $qubits qubits using $method method..."
        python examples/max_cut_qaoa.py --qubits $qubits --method $method --cores_per_qpu 9
        echo ""
        sleep 2
    done
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
