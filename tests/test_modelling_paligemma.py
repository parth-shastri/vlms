import pytest
import torch

# test_modelling_gemma.py
from paligemma.modelling_paligemma import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaConfig,
)
from paligemma.modelling_siglip import SiglipVisionConfig
from paligemma.modelling_gemma import GemmaConfig
from paligemma.cache_utils import KVCache


@pytest.fixture
def model_configs():
    vision_config = SiglipVisionConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        image_size=224,
        patch_size=14,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    )

    text_config = GemmaConfig(
        vocab_size=257152,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=12,
        num_attention_heads=16,
        num_key_value_heads=16,
        max_position_embeddings=8192,
    )

    config = PaliGemmaConfig(
        vision_config=vision_config,
        text_config=text_config,
        projection_dim=2048,
        image_token_index=256000,
    )
    return config


@pytest.fixture
def model(model_configs):
    return PaliGemmaForConditionalGeneration(model_configs)


# def test_basic_forward_pass(model):
#     batch_size = 2
#     seq_length = 20
#     img_size = 224
#     patch_size = 14

#     num_patches = (img_size // patch_size) ** 2

#     # Create sample inputs
#     input_ids = torch.randint(0, 257152, (batch_size, seq_length))
#     pixel_values = torch.randn(batch_size, 3, img_size, img_size)

#     # Add image token at a specific position
#     image_ids = torch.stack(
#         [torch.tensor([model.config.image_token_index] * num_patches)] * batch_size,
#         dim=0,
#     )
#     input_ids = torch.cat([image_ids, input_ids], dim=-1)
#     attention_mask = torch.ones_like(input_ids)

#     outputs = model(
#         input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask
#     )

#     assert "logits" in outputs
#     assert outputs["logits"].shape == (
#         batch_size,
#         input_ids.shape[-1],
#         model.config.vocab_size,
#     )
#     assert outputs["logits"].dtype == torch.float32


def test_forward_with_kv_cache(model):
    batch_size = 1
    seq_length = 10
    img_size = 224
    patch_size = 14

    num_patches = (img_size // patch_size) ** 2

    input_ids = torch.randint(0, 257152, (batch_size, seq_length))
    pixel_values = torch.randn(batch_size, 3, img_size, img_size)
    attention_mask = torch.ones(batch_size, num_patches + seq_length)
    kv_cache = KVCache()

    # Add num_patches image tokens at the start of the input_ids
    image_ids = torch.tensor([model.config.image_token_index] * num_patches).unsqueeze(
        0
    )
    input_ids = torch.cat([image_ids, input_ids], dim=1)

    # put the model in eval mode
    model = model.eval()

    for _ in range(5):  # generation steps
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        # logits and kvcache
        logits = outputs["logits"]
        kv_cache = outputs["kv_cache"]

        print(kv_cache.num_items())

        next_token_logit = logits[:, -1, :]
        # perform a simple sampling (greedy)
        next_token = torch.argmax(next_token_logit, dim=-1, keepdim=True)

        input_ids = next_token
        # update the attention mask
        attention_mask = torch.cat(
            [attention_mask, torch.ones(batch_size, 1, device=input_ids.device)], dim=-1
        )
        # Update the pixel values to None.
        pixel_values = None

    assert "kv_cache" in outputs
    assert isinstance(outputs["kv_cache"], KVCache)
    assert "logits" in outputs


# def test_forward_training_mode(model):
#     batch_size = 2
#     seq_length = 15
#     img_size = 224
#     patch_size = 14

#     num_patches = (img_size // patch_size) ** 2

#     input_ids = torch.randint(0, 257152, (batch_size, seq_length))
#     pixel_values = torch.randn(batch_size, 3, img_size, img_size)

#     # Add image token at a specific position
#     image_ids = torch.stack(
#         [torch.tensor([model.config.image_token_index] * num_patches)] * batch_size,
#         dim=0,
#     )
#     input_ids = torch.cat([image_ids, input_ids], dim=-1)
#     attention_mask = torch.ones_like(input_ids)
#     labels = torch.randint(0, 257152, (batch_size, input_ids.shape[-1]))
#     token_type_ids = torch.ones_like(input_ids)
#     # this should set the self.training attr to True
#     model.train()

#     outputs = model(
#         input_ids=input_ids,
#         pixel_values=pixel_values,
#         attention_mask=attention_mask,
#         labels=labels,
#         token_type_ids=token_type_ids,
#     )

#     assert "logits" in outputs
#     assert outputs["logits"].shape == (
#         batch_size,
#         input_ids.shape[-1],
#         model.config.vocab_size,
#     )


# def test_invalid_image_token_count(model):
#     batch_size = 1
#     seq_length = 10
#     img_size = 224

#     input_ids = torch.randint(0, 257152, (batch_size, seq_length))
#     pixel_values = torch.randn(batch_size, 3, img_size, img_size)
#     attention_mask = torch.ones_like(input_ids)

#     # Add multiple image tokens which doesn't match the image features
#     input_ids[:, 2] = model.config.image_token_index
#     input_ids[:, 3] = model.config.image_token_index

#     with pytest.raises(ValueError, match="Number of images doesn't match"):
#         model(
#             input_ids=input_ids,
#             pixel_values=pixel_values,
#             attention_mask=attention_mask,
#         )


# def test_device_consistency(model):
#     if not torch.cuda.is_available():
#         pytest.skip("CUDA not available")

#     batch_size = 1
#     seq_length = 10
#     img_size = 224
#     patch_size = 14

#     num_patches = (img_size // patch_size) ** 2

#     model = model.cuda()
#     input_ids = torch.randint(0, 257152, (batch_size, seq_length)).cuda()

#     image_ids = torch.stack(
#         [
#             torch.tensor(
#                 [model.config.image_token_index] * num_patches, device=input_ids.device
#             )
#         ]
#         * batch_size,
#         dim=0,
#     )
#     input_ids = torch.cat([image_ids, input_ids], dim=-1)

#     # Random pixel values
#     pixel_values = torch.randn(batch_size, 3, img_size, img_size).cuda()
#     attention_mask = torch.ones_like(input_ids)

#     outputs = model(
#         input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask
#     )

#     assert outputs["logits"].device.type == "cuda"
