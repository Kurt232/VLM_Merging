#!/bin/bash

# Configure cache directories
export HF_HOME='/path/to/model_hub'
export PYTORCH_KERNEL_CACHE_PATH='/path/to/cache'
export CUDA_VISIBLE_DEVICES='0'  # Specify which GPU to use

# Optional: Set Hugging Face token if needed
export HF_TOKEN="your_hf_token_here"

# Change to the project directory
cd /path/to/VLMmerging/

# Source conda to make conda activate available in this script
eval "$(conda shell.bash hook)"
conda activate your_environment

# Output directory for merged models
OUTPUT_DIR='/path/to/merged_models'

# Example 1: Basic linear interpolation merge
echo "Merging models with linear interpolation..."
python merge.py --model1_path path/to/model1 --model2_path path/to/model2 \
    --basemodel_path path/to/base_model \
    --output_dir "${OUTPUT_DIR}/linear_merge" --alpha 0.9 --mode 'base'

# Example 2: TIES merging strategy
echo "Merging models with TIES strategy..."
python merge.py --model1_path path/to/model1 --model2_path path/to/model2 \
    --basemodel_path path/to/base_model \
    --output_dir "${OUTPUT_DIR}/ties_merge" --alpha 1 --mode 'ties' --density 0.2 --alpha2 0.2

# Example 3: Layer swapping strategy
echo "Merging models with layer swapping strategy..."
python merge.py --model1_path path/to/model1 --model2_path path/to/model2 \
    --basemodel_path path/to/base_model \
    --output_dir "${OUTPUT_DIR}/layerswap_merge" --alpha 0.9 --mode 'layerswap' --base_layer_num 5

echo "All merging operations completed successfully!"