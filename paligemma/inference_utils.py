import torch
from PIL import Image
from paligemma.processing_paligemma import PaliGemmaProcessor
from paligemma.modelling_paligemma import PaliGemmaForConditionalGeneration
from paligemma.cache_utils import KVCache


def move_inputs_to_device(model_inputs, device):
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


def get_model_inputs(processor: PaliGemmaProcessor, prompt: str, image_file_path: str, device: str):
    image = Image.open(image_file_path)
    image, text = [image], [prompt]
    model_inputs = processor(text=text, images=image, return_tensors="pt")
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs


def _sample_top_p(probs: torch.Tensor, top_p: float):
    # Perform top_p sampling of the probs
    sorted_probs, indices_probs = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    mask = cum_probs - sorted_probs > top_p

    sorted_probs[mask] = 0.0
    sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))

    next_token = torch.multinomial(sorted_probs, num_samples=1)

    next_token = torch.gather(indices_probs, -1, next_token)
    return next_token


def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    prompt: str,
    image_file_path: str,
    device: str,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
):
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    # debug print statement.
    print("[DEBUG]: Inference inputs: \n")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Pixel Values shape: {pixel_values.shape}")

    # init the KV cache
    kv_cache = KVCache()

    # Generate tokens until you see the stop token
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    for _ in range(max_tokens_to_generate):
        # forward pass
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache
        )
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]
        # Sample the next token
        if do_sample:
            # Apply temperature
            next_token_probs = torch.softmax(next_token_logits / (temperature + 1e-8), dim=-1)
            next_token = _sample_top_p(next_token_probs, top_p)
            # next_token = torch.multinomial(next_token_probs, num_samples=1)

        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        assert next_token.size() == (1, 1)
        # squeeze the next token
        next_token = next_token.squeeze(0)
        generated_tokens.append(next_token)

        # stop token
        if next_token.item() == stop_token:
            break

        input_ids = next_token.unsqueeze(-1)
        # input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )
        # set pixels to None
        pixel_values = None

        # decode the next token
        print(processor.tokenizer.decode(next_token, skip_special_tokens=True), end="")

    generated_tokens = torch.cat(generated_tokens, dim=-1)

    # print(f"[DEBUG]: Generated token ids: \n{generated_tokens}")

    # print("The whole output: ")
    # # Decode the generated tokens
    # decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    # print(prompt + " " + decoded)
