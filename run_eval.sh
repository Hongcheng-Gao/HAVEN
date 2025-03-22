#!/bin/bash

MODEL_LIST=(
    "VideoChatGPT"
    "Video-LLaMA-2"
    "VideoChat2"
    "VideoLLaVA"
    "LLaMA-VID"
    "PLLaVA"
    "LLaVA-NeXT-Video"
    "ShareGPT4Video"
    "LLaVA"
    "Qwen2.5-VL"
    "Video-LLaMA-2-7B"
    "Video-LLaMA-2-13B"
    "LLaMA-VID-13B"
    "PLLaVA-13B"
    "LLaVA-NeXT-Video-34B"
    "Valley"
)
VIDEO_PATH="path/to/video/files"
OUTPUT_PATH="path/to/final/output"
DATASET_PATH="Data/test_data.json"
API_KEY="your_api_key_here"
GPT_URL="your_gpt_url_here"

for MODEL_NAME in "${MODEL_LIST[@]}"; do
    ANSWER_FILE="${OUTPUT_PATH}/${MODEL_NAME}_test_result.json"

    echo "Running Infer.py for model: $MODEL_NAME..."
    python Infer.py --model_name "$MODEL_NAME" --video_path "$VIDEO_PATH" --output_file "$ANSWER_FILE" --dataset_path "$DATASET_PATH"

    echo "Running Judge.py for model: $MODEL_NAME..."
    python Judge.py --model_name "$MODEL_NAME" --answer_file "$ANSWER_FILE" --output_file "$OUTPUT_PATH/${MODEL_NAME}_eval_result.json" --api_key "$API_KEY" --GPT_url "$GPT_URL"

    echo "Completed processing for model: $MODEL_NAME"
done

echo "All models have been tested and evaluated!"
