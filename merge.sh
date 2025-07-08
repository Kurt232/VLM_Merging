CUDA_VISIBLE_DEVICES=2

OUTPUT_DIR="./models"
python merge.py --model1_path llava-hf/llama3-llava-next-8b-hf --model2_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --output_dir "${OUTPUT_DIR}/linear_merge" --alpha 0.9 --mode 'base'