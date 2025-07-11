#!/bin/bash
set -e

# Define evaluation settings
TAG=$1

echo "Starting evaluation $TAG ..."

OUTPUT_DIR="./eval/$TAG"
MERGED_MODELS_DIR="models/$TAG"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir "$OUTPUT_DIR" # Create output directory if it doesn't exist
mkdir "$LOG_DIR" -p

# Define tasks and models as lists
tasks=(
    "MathVista_MINI"
    "MathVerse_MINI_Vision_Only"
    "MathVerse_MINI_Vision_Dominant"
    "MathVerse_MINI_Vision_Intensive"
    "MathVerse_MINI_Text_Lite"
    "MathVerse_MINI_Text_Dominant"
    "MathVerse_MINI_Text_Intensive"
    "MathVision_MINI"
    "MMStar"
    "DynaMath"
)
    # "MM-Math"

models=(
    "llava_next_merge_7b"
)

merges=(
    "${MERGED_MODELS_DIR}/merged_model_0.9.pth"
)

echo "Starting evaluation of VLM models..."


for model in "${models[@]}"; do
    # First evaluate base models without merging
    echo "Evaluating base model: ${model}"
    python VLMEvalKit/run.py \
        --data "${tasks[@]}" \
        --model "$model" \
        --verbose \
        --work-dir "${OUTPUT_DIR}/base_models" > "${LOG_DIR}/${model}_base.log"
    
    # Then evaluate with merged weights
    for merge in "${merges[@]}"; do
        echo "Evaluating merged model: ${model} with weights: ${merge}"
        python VLMEvalKit/run.py \
            --data "${tasks[@]}" \
            --model "$model" \
            --verbose \
            --merge_model "$merge" \
            --work-dir "${OUTPUT_DIR}/merged_models" > "${LOG_DIR}/${model}_merged.log"
    done
done

echo "All evaluation tasks completed successfully!" 