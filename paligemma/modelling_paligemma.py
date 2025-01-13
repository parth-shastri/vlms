import torch
from torch import nn
from paligemma.modelling_siglip import SiglipVisionConfig, SiglipVisionModel
from paligemma.modelling_gemma import GemmaConfig, GemmaForCausalLM
from paligemma.cache_utils import KVCache


class PaliGemmaConfig:

    def __init__(
        self,
        vision_config: SiglipVisionConfig | dict,
        text_config: GemmaConfig | dict,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        self.vision_config = vision_config
        self.text_config = text_config
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.is_encoder_decoder = False

        self.vision_config = (
            vision_config
            if isinstance(vision_config, SiglipVisionConfig)
            else SiglipVisionConfig(**vision_config)
        )
        self.text_config = (
            text_config
            if isinstance(text_config, GemmaConfig)
            else GemmaConfig(**text_config)
        )
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (
            self.vision_config.image_size // self.vision_config.patch_size
        ) ** 2
        self.vision_config.projection_dim = projection_dim


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(
            config.vision_config.hidden_size,
            config.vision_config.projection_dim,
            bias=True,
        )

    def forward(self, image_features):
        # image_features: [B_S, num_patches, embed_dim] -> [B_S, num_patches, projection_dim]
        hidden_states = self.linear(image_features)
        return hidden_states


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _prepare_4d_causal_mask(
        self,
        input_embeds,
        attention_mask,
        token_type_ids,
        is_training: bool = False,
        kv_cache: KVCache | None = None,
        cache_position: torch.Tensor | None = None,
    ):
        # Calculate the seq_len, the len(query) during inference
        dtype = input_embeds.dtype
        min_dtype = torch.finfo(dtype).min
        seq_len = input_embeds.shape[1]
        past_kv_length = kv_cache.num_items() if kv_cache is not None else 0

        if cache_position is None:
            cache_position = torch.arange(
                past_kv_length, past_kv_length + seq_len, device=input_embeds.device
            )

        if kv_cache is None or kv_cache.num_items() == 0:
            target_length = seq_len

        else:
            target_length = (
                past_kv_length + seq_len
            )  # equivalent to cache_position[0] + seq_len
        # print(seq_len, target_length)
        # prepare the causal mask now
        causal_mask = torch.full(
            (seq_len, target_length), fill_value=min_dtype, device=input_embeds.device
        )
        # this mask is filled with -inf
        if seq_len != 1:
            if is_training:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            else:
                # no causal mask
                causal_mask[:, :seq_len] = 0.0
        # mask according to cache position
        causal_mask *= torch.arange(
            target_length, device=cache_position.device
        ) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(
            input_embeds.shape[0], 1, -1, -1
        )
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
                :, None, None, :
            ].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[
                :, :, :, :mask_length
            ].masked_fill(padding_mask, min_dtype)
            # if we are training we need the image + prefix fully masked
            if is_training:
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(
                    token_type_ids[:, None, None, :].to(causal_mask.device) == 0,
                    0,
                )

        return causal_mask

    def _merge_input_ids_with_image_features(
        self, image_features, input_embeds, input_ids
    ):
        # create the mask for the image tokens in the input ids
        special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
        special_image_mask = (
            special_image_mask.expand_as(input_embeds).to(input_embeds.device)
        )
        if input_embeds[special_image_mask].numel() != image_features.numel():
            image_tokens_in_text = torch.sum(input_ids == self.config.image_token_index)
            raise ValueError(
                "Number of images doesn't match the number of image tokens in the text. "
                f"Got {image_tokens_in_text} image_tokens in text but {image_features.shape[0] * image_features.shape[1]} found."
            )
        image_features = image_features.to(input_embeds.device, input_embeds.dtype)
        input_embeds = input_embeds.masked_scatter(special_image_mask, image_features)
        return input_embeds

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor = None,
        kv_cache: KVCache | None = None,
        **kwargs
    ) -> tuple:

        # make sure the input is not padded
        assert torch.all(attention_mask == 1), "The input_ids should not be padded"

        # check if is training
        is_training = token_type_ids is not None and labels is not None

        # 1. Extract input embeddings
        # shape -> (B_S, seq_len, embed_dim)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 2. Merge the text & images if the pixel values are not None
        if pixel_values is not None:
            # shape -> (B_S, num_patches, embed_dim)      (seq_len := num_patches + text_len)
            selected_image_feature = self.vision_tower(
                pixel_values.to(input_embeds.device)
            )
            image_features = self.multi_modal_projector(selected_image_feature)
            image_features = image_features / (self.config.text_config.hidden_size**0.5)
            # Merge the embeddings of the text & the image tokens
            input_embeds = self._merge_input_ids_with_image_features(
                image_features,
                input_embeds,
                input_ids,
            )

        # cache position
        past_kv_length = kv_cache.num_items() if kv_cache is not None else 0
        cache_position = torch.arange(
            past_kv_length,
            past_kv_length + input_embeds.shape[1],
            device=input_embeds.device,
        )

        # position ids
        if kv_cache is not None:
            position_ids = cache_position.unsqueeze(0) + 1  # position ids start from 1
        else:
            position_ids = (
                torch.arange(
                    0, input_embeds.shape[1], device=input_embeds.device
                ).unsqueeze(0) + 1
            )

        causal_mask = self._prepare_4d_causal_mask(
            input_embeds,
            attention_mask,
            token_type_ids,
            is_training,
            kv_cache=kv_cache,
            cache_position=cache_position,
        )
        # print(f"Input shape: {input_ids.shape}")
        # print(f"Position IDs: {position_ids}")
        # print(f"Attention mask: {causal_mask}")
        # 3. Run the language model
        outputs = self.language_model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache,
            **kwargs
        )
        # TODO: add calculation of loss here.
        if pixel_values is not None and kwargs.get("output_hidden_states", False):
            outputs.update({"image_hidden_states": image_features})
        return outputs
