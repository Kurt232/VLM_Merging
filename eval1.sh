#!/bin/bash
set -e

# Define evaluation settings
OUTPUT_DIR="./eval1"
# MERGED_MODELS_DIR="models/linear_merge"
MERGED_MODELS_DIR="models/llava_dart_uniform"

# Define tasks and models as lists
tasks=(
    # "MathVista_MINI"
    # "MathVerse_MINI_Vision_Only"
    # "MathVerse_MINI_Vision_Dominant"
    # "MathVerse_MINI_Vision_Intensive"
    # "MathVerse_MINI_Text_Lite"
    # "MathVerse_MINI_Text_Dominant"
    # "MathVerse_MINI_Text_Intensive"
    # "MathVision_MINI"
    # "MMStar"
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
    CUDA_VISIBLE_DEVICES="0" python VLMEvalKit/run.py \
        --data "${tasks[@]}" \
        --model "$model" \
        --verbose \
        --work-dir "${OUTPUT_DIR}/base_models" > test.log 2>&1 &
    
    # Then evaluate with merged weights
    for merge in "${merges[@]}"; do
        echo "Evaluating merged model: ${model} with weights: ${merge}"
        CUDA_VISIBLE_DEVICES="1" python VLMEvalKit/run.py \
            --data "${tasks[@]}" \
            --model "$model" \
            --verbose \
            --merge_model "$merge" \
            --work-dir "${OUTPUT_DIR}/merged_models"
    done
    wait
done

echo "All evaluation tasks completed successfully!" 