from dataclasses import dataclass, replace
from pathlib import Path

import torch

import comfy.lora
import comfy.model_detection
import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
import comfy.sd
import comfy.storage
import comfy.utils

from .model_paths import resolve_model
from .text_model import SenseNovaTextModel


@dataclass(frozen=True)
class SenseNovaModel:
    image_model: object
    text_model: object
    clip: object
    vae: object
    tokenizer: object

    def get_models(self):
        return [self.image_model, self.text_model]


def load_model(model_name, model_path="", dtype="auto", download_model=False):
    files, weight_map = resolve_model(model_name, model_path, download_model)
    state = {}
    for path in files:
        shard = comfy.utils.load_torch_file(str(path), safe_load=True)
        overlap = state.keys() & shard.keys()
        if overlap:
            raise ValueError(f"Duplicate checkpoint tensor: {next(iter(overlap))}")
        state.update(shard)
    if weight_map is not None and weight_map.keys() - state.keys():
        raise ValueError("SenseNova checkpoint does not match its shard index.")
    prefix = comfy.model_detection.unet_prefix_from_state_dict(state)
    config = comfy.model_detection.model_config_from_unet(state, prefix)
    if config is None or config.unet_config.get("image_model") != "sensenova_u15":
        raise ValueError("This loader requires a complete SenseNova-U1.5-8B-MoT or SFT checkpoint.")
    text_state = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
    if "language_model.lm_head.weight" not in text_state:
        raise ValueError("Visual Q&A and interleaved generation require language_model.lm_head.weight from the full official checkpoint.")
    options = {} if dtype == "auto" else {"dtype": {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]}
    image_model, clip, vae, _ = comfy.sd.load_state_dict_guess_config(state, model_options=options)
    text = SenseNovaTextModel(torch.device("meta"), image_model.model.get_dtype_inference(), comfy.ops.manual_cast)
    text_state = {key: value for key, value in text_state.items() if key.startswith(("vision_model.", "language_model.")) and "_mot_gen" not in key}
    text.load_state_dict(text_state, strict=True, assign=True)
    text_patcher = comfy.model_patcher.CoreModelPatcher(text, load_device=mm.get_torch_device(), offload_device=mm.unet_offload_device(), fast_disk=comfy.storage.state_dict_fast_disk(text_state))
    tokenizer = clip.tokenizer.sensenova_u15.tokenizer
    return SenseNovaModel(image_model, text_patcher, clip, vae, tokenizer)


def apply_lora(handle, lora_path, strength=1.0):
    path = Path(lora_path.strip()).resolve()
    if path.suffix != ".safetensors" or not path.is_file():
        raise FileNotFoundError(f"Select a local SenseNova LoRA Safetensors file: {path}")
    lora = comfy.utils.load_torch_file(str(path), safe_load=True)
    mapping = {key.removesuffix(".weight"): f"diffusion_model.{key}" for key in handle.image_model.model.diffusion_model.state_dict() if key.endswith(".weight")}
    patches = comfy.lora.load_lora(lora, mapping)
    if not patches:
        raise ValueError("No matching SenseNova LoRA tensors were found.")
    image_model = handle.image_model.clone()
    image_model.add_patches(patches, strength)
    text_model = handle.text_model.clone()
    text_keys = text_model.model.state_dict().keys()
    text_model.add_patches({key.removeprefix("diffusion_model."): value for key, value in patches.items() if key.removeprefix("diffusion_model.") in text_keys}, strength)
    return replace(handle, image_model=image_model, text_model=text_model)
