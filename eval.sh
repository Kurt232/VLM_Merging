#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES='2'

# Count available GPUs
export GPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Detected ${GPU} GPUs for distributed training"

# Define evaluation settings
OUTPUT_DIR="./eval"
# MERGED_MODELS_DIR="models/linear_merge"
MERGED_MODELS_DIR="models/llava_dart_uniform"

# Define tasks and models as lists
tasks=(
    "MathVista_MINI"
    "MathVerse_MINI_Vision_Only"
    "MathVerse_MINI_Vision_Dominant"
    "MathVerse_MINI_Vision_Intensive"
    "MathVerse_MINI_Text_Lite"
    "MathVerse_MINI_Text_Dominant"
    "MathVerse_MINI_Text_Intensive"
)

    # "MathVision_MINI"
    # "MM-Math"
    # "DynaMath"
    # "MMStar"

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
    torchrun --nproc-per-node=${GPU} --master-port=12345 VLMEvalKit/run.py \
        --data "${tasks[@]}" \
        --model "$model" \
        --verbose \
        --work-dir "${OUTPUT_DIR}/base_models"
    
    # Then evaluate with merged weights
    for merge in "${merges[@]}"; do
        echo "Evaluating merged model: ${model} with weights: ${merge}"
        torchrun --nproc-per-node=${GPU} --master-port=12345 VLMEvalKit/run.py \
            --data "${tasks[@]}" \
            --model "$model" \
            --verbose \
            --merge_model "$merge" \
            --work-dir "${OUTPUT_DIR}/merged_models"
    done
done

echo "All evaluation tasks completed successfully!" 