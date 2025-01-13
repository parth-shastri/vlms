import torch
import argparse
# from transformers import AutoProcessor
from paligemma.inference_utils import test_inference
from paligemma.model_utils import load_hf_model
from paligemma.processing_paligemma import PaliGemmaProcessor


def get_args():
    parser = argparse.ArgumentParser(description='Run inference with PaLI-GEMMA model')
    parser.add_argument('--model_id', type=str, required=True,
                        help='HuggingFace model ID')
    parser.add_argument('--prompt', type=str, required=True,
                        help='Text prompt for inference')
    parser.add_argument('--image_file_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--max_tokens_to_generate', type=int, default=100,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.9,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.8,
                        help='Top p sampling parameter')
    parser.add_argument('--do_sample', action='store_true',
                        help='Whether to use sampling for generation')
    parser.add_argument('--only_cpu', action='store_true',
                        help='Force CPU usage only')
    return parser.parse_args()


def main(
    model_id: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.9,
    top_p: float = 0.8,
    do_sample: bool = False,
    only_cpu: bool = False,
):
    device = "cpu"

    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

    print("Device in Use: ", device)
    print("Loading model ...")

    model_path = model_id.split("/")[-1]
    model, tokenizer = load_hf_model(model_path, model_id, device)
    model = model.eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    # try out the normal processor
    # processor = AutoProcessor.from_pretrained(model_id)
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print("Running inference ...")
    with torch.no_grad():
        test_inference(
            model,
            processor,
            prompt,
            image_file_path,
            device,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
        )


if __name__ == "__main__":
    args = get_args()
    main(
        model_id=args.model_id,
        prompt=args.prompt,
        image_file_path=args.image_file_path,
        max_tokens_to_generate=args.max_tokens_to_generate,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        only_cpu=args.only_cpu,
    )
