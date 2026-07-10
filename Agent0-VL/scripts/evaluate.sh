#!/bin/bash
# Agent0-VL Evaluation Script
# Evaluate trained model on various benchmarks

set -e

# Default configuration
MODEL_PATH="${MODEL_PATH:-./checkpoints/agent0/iteration_3/final}"
OUTPUT_DIR="${OUTPUT_DIR:-./evaluation_results}"
BENCHMARKS="${BENCHMARKS:-mathverse,mathvista,chartqa}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ENABLE_TOOLS="${ENABLE_TOOLS:-true}"
ENABLE_VERIFICATION="${ENABLE_VERIFICATION:-true}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --benchmarks) BENCHMARKS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --enable_tools) ENABLE_TOOLS="$2"; shift 2 ;;
        --enable_verification) ENABLE_VERIFICATION="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=================================="
echo "Agent0-VL Evaluation"
echo "=================================="
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Benchmarks: ${BENCHMARKS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Tools: ${ENABLE_TOOLS}"
echo "Verification: ${ENABLE_VERIFICATION}"
echo "=================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run evaluation
python scripts/evaluate.py \
    --model_path "${MODEL_PATH}" \
    --benchmarks "${BENCHMARKS}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --enable_tools "${ENABLE_TOOLS}" \
    --enable_verification "${ENABLE_VERIFICATION}" \
    --save_predictions true

echo "=================================="
echo "Evaluation Complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=================================="
