import json
import os
import glob
from transformers import AutoTokenizer
from transformers import PaliGemmaForConditionalGeneration as hf_paligemma
from paligemma.modelling_paligemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from safetensors import safe_open


def load_hf_model(model_path: str, model_id: str, device: str) -> tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    # Load the model weights from the HuggingFace repository if no safetensors file are found in the model_path or the modelpath doesn't exist
    # fetch the safetensors file to get the weight files.
    os.makedirs(model_path, exist_ok=True)

    safetensors_files = glob.glob(
        os.path.join(
            model_path,
            "--".join(["models"] + model_id.split("/")),
            "**",
            "model-*.safetensors",
        ),
        recursive=True,
    )
    if len(safetensors_files) == 0:
        model = hf_paligemma.from_pretrained(model_id, cache_dir=model_path)
        del model
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=model_path, padding_side='right')
    assert tokenizer.padding_side == "right"

    # load the model weights to tensors dict
    # tensors = {}
    # for safetensors_file in safetensors_files:
    #     with safe_open(safetensors_file, framework="pt", device='cpu') as f:
    #         for key in f.keys():
    #             tensors[key] = f.get_tensor(key)

    # Load weights with proper mapping
    state_dict = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                # Handle language model prefix mapping
                if key.startswith("language_model.model"):
                    new_key = key
                elif key.startswith("vision_tower"):
                    new_key = key
                elif key.startswith("multi_modal_projector"):
                    new_key = key
                else:
                    new_key = key
                state_dict[new_key] = f.get_tensor(key)

    # load the model configuration
    config_path = glob.glob(os.path.join(model_path, "--".join(["models"] + model_id.split("/")), "**", "config.json"), recursive=True)[0]
    with open(config_path, "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(config).to(device)

    # load the model state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print(f"Missing Keys required by the model: \n{missing_keys}")
    print(f"Unexpected keys recieved to the model: \n{unexpected_keys}")

    # tie the embedding weights
    model.tie_weights()

    # Load the HF model directly (Check the inference code.)
    # model = hf_paligemma.from_pretrained(model_id, cache_dir=model_path).to(device)

    return model, tokenizer
