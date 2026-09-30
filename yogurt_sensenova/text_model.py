import torch
from torch import nn

from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
from comfy.ldm.sensenova import model as native


class TextAttention(native.Attention):
    def __init__(self, device, dtype, operations):
        nn.Module.__init__(self)
        self.comfy_attention = ComfyAttention()
        for name, heads in (("q_proj", native.NUM_HEADS), ("k_proj", native.NUM_KV_HEADS), ("v_proj", native.NUM_KV_HEADS)):
            setattr(self, name, operations.Linear(native.HIDDEN_SIZE, heads * native.HEAD_DIM, bias=False, device=device, dtype=dtype))
        self.o_proj = operations.Linear(native.HIDDEN_SIZE, native.HIDDEN_SIZE, bias=False, device=device, dtype=dtype)
        for name in ("q_norm", "q_norm_hw", "k_norm", "k_norm_hw"):
            setattr(self, name, operations.RMSNorm(native.HEAD_DIM // 2, eps=1e-6, device=device, dtype=dtype))

    def forward_cached(self, hidden, rope, past_key, past_value, transformer_options):
        query, key, value = self._project(hidden, rope, False)
        key = torch.cat((past_key, key), dim=2)
        value = torch.cat((past_value, value), dim=2)
        output = optimized_attention(
            AttentionTensorContainer(query), AttentionTensorContainer(key), AttentionTensorContainer(value),
            native.NUM_HEADS, skip_reshape=True, enable_gqa=True,
            transformer_options=transformer_options, preferred_attention=self.comfy_attention,
        )
        return self.o_proj(output), key, value


class TextLayer(native.DecoderLayer):
    def __init__(self, device, dtype, operations):
        nn.Module.__init__(self)
        self.self_attn = TextAttention(device, dtype, operations)
        self.mlp = native.MLP(device=device, dtype=dtype, operations=operations)
        self.input_layernorm = operations.RMSNorm(native.HIDDEN_SIZE, eps=1e-6, device=device, dtype=dtype)
        self.post_attention_layernorm = operations.RMSNorm(native.HIDDEN_SIZE, eps=1e-6, device=device, dtype=dtype)

    def forward_cached(self, hidden, rope, key, value, transformer_options):
        attention, key, value = self.self_attn.forward_cached(self.input_layernorm(hidden), rope, key, value, transformer_options)
        hidden = hidden + attention
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden, key, value


class TextBackbone(nn.Module):
    def __init__(self, device, dtype, operations):
        super().__init__()
        self.embed_tokens = operations.Embedding(native.VOCAB_SIZE, native.HIDDEN_SIZE, padding_idx=151643, device=device, dtype=dtype)
        self.layers = nn.ModuleList(TextLayer(device, dtype, operations) for _ in range(native.NUM_LAYERS))
        self.norm = operations.RMSNorm(native.HIDDEN_SIZE, eps=1e-6, device=device, dtype=dtype)


class TextLanguageModel(nn.Module):
    def __init__(self, device, dtype, operations):
        super().__init__()
        self.model = TextBackbone(device, dtype, operations)
        self.lm_head = operations.Linear(native.HIDDEN_SIZE, native.VOCAB_SIZE, bias=False, device=device, dtype=dtype)


class SenseNovaTextModel(nn.Module):
    def __init__(self, device, dtype, operations):
        super().__init__()
        self.dtype = dtype
        self.vision_model = native.VisionModel(device=device, dtype=dtype, operations=operations)
        self.language_model = TextLanguageModel(device, dtype, operations)

    def prefill(self, input_ids, images=None, indexes=None, mask=None, transformer_options=None):
        hidden = self.language_model.model.embed_tokens(input_ids, out_dtype=self.dtype)
        if images:
            embeddings = [self.vision_model(native._pad_to_merged_patch_size(image)) for image in images]
            hidden = hidden.clone()
            hidden[input_ids == 151669] = torch.cat(embeddings, dim=1).reshape(-1, native.HIDDEN_SIZE)
        if indexes is None:
            positions = torch.arange(input_ids.shape[1], device=hidden.device)
            indexes = torch.stack((positions, torch.zeros_like(positions), torch.zeros_like(positions)))
            mask = torch.full((input_ids.shape[1], input_ids.shape[1]), float("-inf"), device=hidden.device, dtype=hidden.dtype).triu(1)
            next_time = torch.full((input_ids.shape[0],), input_ids.shape[1], device=hidden.device, dtype=torch.long)
        else:
            indexes = indexes.transpose(0, 1)
            next_time = indexes[0].amax(dim=-1) + 1
        rope = native._prepare_mrope(indexes, hidden.device, hidden.dtype)
        keys, values = [], []
        options = dict(transformer_options or {})
        for index, layer in enumerate(self.language_model.model.layers):
            options["block_index"] = index
            hidden, key, value = layer.forward_prefix(hidden, rope, mask, options)
            keys.append(key)
            values.append(value)
        logits = self.language_model.lm_head(self.language_model.model.norm(hidden[:, -1:]))[:, 0].float()
        return logits, keys, values, next_time

    def decode_token(self, input_ids, keys, values, position, transformer_options=None):
        hidden = self.language_model.model.embed_tokens(input_ids, out_dtype=self.dtype)
        time = torch.full(input_ids.shape, position, dtype=torch.long, device=hidden.device)
        indexes = torch.stack((time, torch.zeros_like(time), torch.zeros_like(time)))
        rope = native._prepare_mrope(indexes, hidden.device, hidden.dtype)
        next_keys, next_values = [], []
        options = dict(transformer_options or {})
        for index, layer in enumerate(self.language_model.model.layers):
            options["block_index"] = index
            hidden, key, value = layer.forward_cached(hidden, rope, keys[index], values[index], options)
            next_keys.append(key)
            next_values.append(value)
        logits = self.language_model.lm_head(self.language_model.model.norm(hidden))[:, -1].float()
        return logits, next_keys, next_values
