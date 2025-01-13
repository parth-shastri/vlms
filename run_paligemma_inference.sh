MODEL_ID="google/paligemma-3b-pt-224"
PROMPT="caption en"
IMAGE_FILE_PATH="./test_images/cyber-kitty.jpg"
MAX_TOKENS=256
TEMPERATURE=0.8
TOP_P=0.95
DO_SAMPLE="True"

python -m paligemma.inference \
         --model_id "$MODEL_ID" \
         --prompt "$PROMPT" \
         --image_file_path "$IMAGE_FILE_PATH" \
         --max_tokens_to_generate "$MAX_TOKENS" \
         --temperature "$TEMPERATURE" \
         --top_p "$TOP_P" \
         --do_sample