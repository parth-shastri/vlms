import torch


class KVCache:
    def __init__(self):
        self.k_cache: list[torch.Tensor] = []
        self.v_cache: list[torch.Tensor] = []
        # TODO: one more method that we can try out is prefill the k_cache, v_cache
        #       ... with empty tensors to make compatible with torch compile.

    def num_items(self):
        if len(self.k_cache) == 0:
            return 0
        else:
            return self.k_cache[0].shape[-2]

    def update(
        self, key_states, value_states, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.k_cache) <= layer_idx:
            # if there is no entry in the key cache
            self.k_cache.append(key_states)
            self.v_cache.append(value_states)

        else:
            # concat the existing key states in the cache with the new ones
            # this concat will happen on the seq len dimension
            self.k_cache[layer_idx] = torch.cat(
                [self.k_cache[layer_idx], key_states], dim=-2
            )
            self.v_cache[layer_idx] = torch.cat(
                [self.v_cache[layer_idx], value_states], dim=-2
            )

        return self.k_cache[layer_idx], self.v_cache[layer_idx]
