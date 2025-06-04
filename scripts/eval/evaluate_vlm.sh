#!/bin/bash

# Configure cache directories and environment
export TRANSFORMERS_CACHE='/path/to/model_hub'
export HF_DATASETS_CACHE='/path/to/model_hub'
export HF_HOME='/path/to/model_hub'
export PYTORCH_KERNEL_CACHE_PATH='/path/to/cache'
export CUDA_VISIBLE_DEVICES='0,1'  # Specify which GPUs to use

# Optional: Set proxy and API keys if needed
# export https_proxy="your_proxy_here"
# export OPENAI_API_KEY='your_openai_key_here'

# Set CUDA environment if needed
export CUDA_HOME="/path/to/cuda"
export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

# Change to the project directory
cd /path/to/VLMEvalKit

# Source conda to make conda activate available in this script
eval "$(conda shell.bash hook)"
conda activate your_environment

# Count available GPUs
export GPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Detected ${GPU} GPUs for distributed training"

# Define evaluation settings
OUTPUT_DIR="/path/to/evaluation_results"
MERGED_MODELS_DIR="/path/to/merged_models"

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
    "MM-Math"
    "DynaMath"
    "MMStar"
)

models=(
    "llava_next_merge_7b"
    "Qwen2-VL-7B-Instruct"
    "idefics2_8b"
)

# Define merged model weights to evaluate
merges=(
    "${MERGED_MODELS_DIR}/merged_model_0.7.pth"
    "${MERGED_MODELS_DIR}/merged_model_0.9.pth"
)

echo "Starting evaluation of VLM models..."

# Run combinations of tasks and models
for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        # First evaluate base models without merging
        echo "Evaluating base model: ${model} on task: ${task}"
        torchrun --nproc-per-node=${GPU} --master-port=12345 run.py \
            --data "$task" \
            --model "$model" \
            --verbose \
            --work-dir "${OUTPUT_DIR}/base_models"
        
        # Then evaluate with merged weights
        for merge in "${merges[@]}"; do
            echo "Evaluating merged model: ${model} with weights: ${merge} on task: ${task}"
            torchrun --nproc-per-node=${GPU} --master-port=12345 run.py \
                --data "$task" \
                --model "$model" \
                --verbose \
                --merge_model "$merge" \
                --work-dir "${OUTPUT_DIR}/merged_models"
        done
    done
done

echo "All evaluation tasks completed successfully!" 