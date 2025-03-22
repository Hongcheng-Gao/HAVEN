#!/bin/bash

MODEL_NAME="name of your model"
VIDEO_PATH="path/to/video/files"
OUTPUT_PATH="path/to/final output"
DATASET_PATH="Data/test_data.json"
ANSWER_FILE="path/to/answer.json"
API_KEY="your_api_key_here"
GPT_URL="your_gpt_url_here"

echo "Running Test.py..."
python Test.py --model_name "$MODEL_NAME" --video_path "$VIDEO_PATH" --output_file "$ANSWER_FILE" --dataset_path "$DATASET_PATH"

echo "Running Eval.py..."
python Eval.py --model_name "$MODEL_NAME" --answer_file "$ANSWER_FILE" --output_file "$OUTPUT_PATH" --api_key "$API_KEY" --GPT_url "$GPT_URL"

echo "Testing and evaluation completed!"
