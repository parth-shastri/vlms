"""
Modelling Gemma: Google's opensource decoder-only language model
                 Uses 'sentencepiece' tokenizer
"""

from typing import Callable
import torch
from torch import nn
from paligemma.cache_utils import KVCache

# from torch.nn import CrossEntropyLoss


class GemmaConfig:
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        max_position_embeddings=8192,
        rms_norm_eps=1e-7,
        rope_theta=10000,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())

        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # calculate theta according to the RoPE paper
        # theta_i = base ^ (-2i / dim) where i = 0, 1, 2, ..., dim
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [B_S, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        # expand inv_freq
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        # position_ids : [B_S, seq_len] -> [B_S, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type

        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            # Autocast enables automatic type exchange between fp16 to fp32
            # freqs: [B_S, seq_len, head_size]
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            # concat the freqs
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        # upcast to the input datatype after done computation.
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    # Build the [-x2, x1, -x4, x3, ...] for the sin part
    x1 = x[..., : x.shape[-1] // 2]  # First half of the last dimension
    x2 = x[..., x.shape[-1] // 2:]  # second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q, k, cos, sin, unsqueeze_dim=1):
    # the unsqueeze dim parameter is supposed to unsqueeze the dimension
    # ... corresponding to the num_heads.
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # apply the rotary embedding
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling: float,
    dropout: float,
    **kwargs,
):
    # key_states.shape: [B_S, n_heads, seq_len, hidden_dim]:: [0, 1, 2, 3]
    key_states = repeat_kv(key, n_rep=module.num_key_value_groups)
    value_states = repeat_kv(value, n_rep=module.num_key_value_groups)

    # attention_weights.shape: [B_S, n_heads, seq_len, seq_len]
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        # no difference as the attention mask has the same shape as the attn_weights.
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x):
        # x: [B_S, seq_len, hidden_size]
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.scaling = self.head_dim**-0.5

        self.is_causal = True

        assert (
            self.hidden_size % self.num_heads == 0
        ), "Hidden size must be divisible by the number of heads"
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        **kwargs,
    ):
        # hidden_states: [B_S, seq_len, hidden_size]
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # project the hidden states to q, k, v
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(
                key_states, value_states, self.layer_idx
            )

        attention_interface: Callable = eager_attention_forward
        # attn_weights, attn_output: [B_S, num_heads, seq_len, seq_len], [B_S, seq_len, num_heads, head_dim]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            scaling=self.scaling,
            dropout=self.attention_dropout if self.training else 0.0,
            **kwargs,
        )

        # attn_output: [B_S, seq_len, num_heads, head_dim]
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        kv_cache: KVCache | None = None,
        **kwargs,
    ):
        output_attentions = kwargs.get("output_attentions", False)
        residual = hidden_states
        # hidden_size: [B_S, seq_len, hidden_size]
        hidden_states = self.input_layernorm(hidden_states)

        # [B_S, seq_len, hidden_size]
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        # [B_S, seq_len, hidden_size]
        hidden_states = residual + hidden_states

        # [B_S, seq_len, hidden_size]
        residual = hidden_states
        # [B_S, seq_len, hidden_size]
        hidden_states = self.post_attention_layernorm(hidden_states)
        # [B_S, seq_len, hidden_size]
        hidden_states = self.mlp(hidden_states)
        # [B_S, seq_len, hidden_size]
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.padding_idx = config.pad_token_id
        # get the head_dim to calcuate the inv_freq values.
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        attention_mask: torch.Tensor | None,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor | None,
        kv_cache: KVCache | None = None,
        **kwargs,
    ):
        # input_embeds: [B_S, seq_len, hidden_size]

        # [B_S, seq_len, hidden_size]
        hidden_states = input_embeds

        normalizer = torch.tensor(
            self.config.hidden_size**0.5, dtype=hidden_states.dtype
        )
        hidden_states = hidden_states * normalizer

        all_hidden_states = () if kwargs.get("output_hidden_states", False) else None
        all_attn_weights = () if kwargs.get("output_attentions", False) else None

        # decoder layers
        for decoder_layer in self.layers:
            if all_hidden_states is not None:
                all_hidden_states += (hidden_states,)
            # [B_S, seq_len, hidden_size]
            layer_output = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
                **kwargs,
            )
            hidden_states = layer_output[0]

            if all_attn_weights is not None:
                all_attn_weights += (layer_output[1],)

        # [B_S, seq_len, hidden_size]
        hidden_states = self.norm(hidden_states)

        if all_hidden_states is not None:
            all_hidden_states += (hidden_states,)

        # [B_S, seq_len, vocab_size]
        return (hidden_states, all_hidden_states, all_attn_weights)


class GemmaForCausalLM(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        input_embeds: torch.Tensor | None,
        kv_cache: KVCache | None = None,
        **kwargs,
    ):
        # forward
        # input_embeds: [B_S, seq_len, hidden_size]
        # outputs: [B_S, seq_len, hidden_size]
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache,
            **kwargs,
        )

        # last hidden states
        hidden_states = outputs[0]
        # TODO: Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {"logits": logits}

        if kwargs.pop("output_hidden_states", False):
            return_data.update({"hidden_states": outputs[1]})

        if kwargs.pop("output_attentions", False):
            return_data.update({"attentions": outputs[2]})

        if kv_cache is not None:
            # return updated cache
            return_data["kv_cache"] = kv_cache

        return return_data
