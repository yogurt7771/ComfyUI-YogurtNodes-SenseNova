import json
import math

import torch

import comfy.model_management as mm
import comfy.utils
from comfy.k_diffusion.sampling import sample_euler
from comfy.ldm.sensenova import conditioning as native
from comfy.ldm.sensenova.sampling import resolution_noise_scale, upstream_sigmas
from comfy.text_encoders.sensenova import SYSTEM_MESSAGE


INTERLEAVE_SYSTEM = (
    "You are a multimodal assistant capable of reasoning with both text and images. You support two modes:\n\n"
    "Think Mode: When reasoning is needed, you MUST start with a <think></think> block and place all reasoning inside it. "
    "You MUST interleave text with generated images using tags like <image1>, <image2>. Images can ONLY be generated "
    "between <think> and </think>, and may be referenced in the final answer.\n\n"
    "Non-Think Mode: When no reasoning is needed, directly provide the answer without reasoning. Do not use tags like "
    "<image1>, <image2>; present any images naturally alongside the text.\n\nAfter the think block, always provide a concise, "
    "user-facing final answer. The answer may include text, images, or both. Match the user's language in both reasoning and the final answer."
)
EMPTY_THINK = "<think>\n\n</think>\n\n"


def ordered_images(images):
    images = images or {}
    names = sorted(images, key=lambda name: (0, int(name.removeprefix("image"))) if name.removeprefix("image").isdigit() else (1, name))
    return native.split_reference_batches([images[name] for name in names])


def image_block(image):
    count = math.ceil(image.shape[-3] / 32) * math.ceil(image.shape[-2] / 32)
    return "<img>" + "<IMG_CONTEXT>" * count + "</img>"


def prompt_ids(tokenizer, prompt, images=(), system_message="", suffix=""):
    missing = len(images) - prompt.count("<image>")
    if missing > 0:
        labels = [f"Image-{index + 1}:<image>\n" for index in range(missing)] if len(images) > 1 and "<image>" not in prompt else ["<image>\n"] * missing
        prompt = "".join(labels) + prompt
    for image in images:
        prompt = prompt.replace("<image>", image_block(image), 1)
    system = f"<|im_start|>system\n{system_message}<|im_end|>\n" if system_message else ""
    text = system + f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n" + suffix
    return tokenizer.encode(text, add_special_tokens=False)


def prefix_inputs(tokens, images, device, dtype):
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    grids = [(math.ceil(image.shape[-3] / 32), math.ceil(image.shape[-2] / 32)) for image in images]
    indexes = native.thw_indexes(ids, grids)
    mask = native.block_causal_mask(indexes, dtype=dtype)
    pixels = [native.preprocess_reference(image).to(device=device, dtype=dtype) for image in images]
    return ids, pixels or None, indexes, mask


def prefix_memory(length, dtype):
    return mm.dtype_size(dtype) * (length * length + 2 * 42 * length * 8 * 128 + 8 * length * 4096)


def text_prefill(handle, tokens, images):
    mm.unload_model_and_clones(handle.image_model)
    patcher = handle.text_model
    mm.load_models_gpu([patcher], memory_required=prefix_memory(len(tokens), patcher.model.dtype))
    patcher.pre_run()
    try:
        inputs = prefix_inputs(tokens, images, patcher.load_device, patcher.model.dtype)
        logits, keys, values, _ = patcher.model.prefill(*inputs, transformer_options=patcher.model_options.get("transformer_options"))
    finally:
        patcher.cleanup()
    next_time = sum(token != native.IMAGE_CONTEXT_ID for token in tokens) + tokens.count(native.IMAGE_START_ID)
    return logits, keys, values, next_time


def choose_token(logits, generated, generator, do_sample=False, temperature=0.7, top_p=0.9, top_k=20, repetition_penalty=1.0):
    logits = logits[0].clone()
    if repetition_penalty != 1.0 and generated:
        indexes = torch.tensor(list(set(generated)), device=logits.device, dtype=torch.long)
        selected = logits[indexes]
        logits[indexes] = torch.where(selected < 0, selected * repetition_penalty, selected / repetition_penalty)
    if not do_sample or temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k > 0:
        threshold = logits.topk(min(top_k, logits.numel())).values[-1]
        logits.masked_fill_(logits < threshold, float("-inf"))
    if top_p < 1.0:
        sorted_logits, indexes = logits.sort(descending=True)
        remove = sorted_logits.softmax(-1).cumsum(-1) - sorted_logits.softmax(-1) >= top_p
        remove[0] = False
        logits[indexes[remove]] = float("-inf")
    return int(torch.multinomial(logits.softmax(-1), 1, generator=generator))


def text_step(handle, token, keys, values, position):
    patcher = handle.text_model
    mm.throw_exception_if_processing_interrupted()
    ids = torch.tensor([[token]], dtype=torch.long, device=patcher.load_device)
    return patcher.model.decode_token(ids, keys, values, position, transformer_options=patcher.model_options.get("transformer_options"))


def answer(handle, question, images, max_new_tokens=1024, seed=0, do_sample=False, temperature=0.7, top_p=0.9, top_k=20, repetition_penalty=1.0):
    tokens = prompt_ids(handle.tokenizer, question, images)
    logits, keys, values, position = text_prefill(handle, tokens, images)
    generator = torch.Generator(device=handle.text_model.load_device).manual_seed(seed)
    generated = []
    progress = comfy.utils.ProgressBar(max_new_tokens)
    handle.text_model.pre_run()
    try:
        for index in range(max_new_tokens):
            mm.throw_exception_if_processing_interrupted()
            token = choose_token(logits, generated, generator, do_sample, temperature, top_p, top_k, repetition_penalty)
            if token == native.IM_END_ID:
                break
            generated.append(token)
            logits, keys, values = text_step(handle, token, keys, values, position)
            position += 1
            progress.update_absolute(index + 1)
    finally:
        handle.text_model.cleanup()
    return handle.tokenizer.decode(generated, skip_special_tokens=True).strip()


def image_prefix(handle, tokens, images):
    model = handle.image_model.model.diffusion_model
    inputs = prefix_inputs(tokens, images, handle.image_model.load_device, handle.image_model.model.get_dtype_inference())
    return model.preprocess_prefix(*inputs)


def sample_image(handle, tokens, images, width, height, seed=0, steps=50, cfg=4.0, image_cfg=1.0, shift=3.0, batch_size=1, conditional_prefix=None, image_only_tokens=None):
    mm.unload_model_and_clones(handle.text_model)
    patcher = handle.image_model
    dtype = patcher.model.get_dtype_inference()
    device = patcher.load_device
    negative_tokens = image_only_tokens if image_only_tokens is not None else prompt_ids(handle.tokenizer, "<image>" * len(images), images, suffix="<img>")
    unconditional_tokens = prompt_ids(handle.tokenizer, "", suffix="<img>")
    lengths = [len(tokens)]
    if cfg != 1.0 or image_cfg != 1.0:
        lengths.extend((len(negative_tokens), len(unconditional_tokens)))
    memory = patcher.memory_required([batch_size, 3, height, width]) + sum(prefix_memory(length, dtype) for length in lengths)
    mm.load_models_gpu([patcher], memory_required=memory)
    patcher.pre_run()
    try:
        conditional = conditional_prefix if conditional_prefix is not None else image_prefix(handle, tokens, images)
        image_only = image_prefix(handle, negative_tokens, images) if (cfg != 1.0 or image_cfg != 1.0) and (image_cfg == 1.0 or cfg != image_cfg) else None
        unconditional = image_prefix(handle, unconditional_tokens, []) if image_cfg != 1.0 else None
        sigmas = upstream_sigmas(steps, shift, device=device)
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn((batch_size, 3, height, width), device=device, dtype=torch.float32, generator=generator)
        noise *= resolution_noise_scale(height, width)
        progress = comfy.utils.ProgressBar(steps)

        def predict(x, sigma, prefix):
            keys, values, time = prefix
            return patcher.model.apply_model(x, sigma, prefix_keys=keys, prefix_values=values, prefix_time=time, transformer_options=patcher.model_options.get("transformer_options", {}).copy())

        def denoise(x, sigma):
            mm.throw_exception_if_processing_interrupted()
            positive = predict(x, sigma, conditional)
            if cfg == 1.0 and image_cfg == 1.0:
                return positive
            if image_cfg == 1.0:
                negative = predict(x, sigma, image_only)
                return negative + cfg * (positive - negative)
            negative = predict(x, sigma, unconditional)
            if cfg == image_cfg:
                return negative + cfg * (positive - negative)
            reference = predict(x, sigma, image_only)
            return negative + cfg * (positive - reference) + image_cfg * (reference - negative)

        result = sample_euler(denoise, noise, sigmas, callback=lambda info: progress.update_absolute(info["i"] + 1), disable=True)
    finally:
        patcher.cleanup()
    return (result.movedim(1, -1).float() * 0.5 + 0.5).clamp(0, 1).to(mm.intermediate_device())


def generate(handle, prompt, images, **settings):
    tokens = prompt_ids(handle.tokenizer, prompt, images, SYSTEM_MESSAGE, EMPTY_THINK + "<img>")
    return sample_image(handle, tokens, images, **settings)


def interleave(handle, prompt, images, width=1536, height=1536, seed=0, steps=50, cfg=4.0, image_cfg=1.0, shift=3.0, max_images=4, max_new_tokens=8192, think_mode=False, do_sample=False, temperature=0.7, top_p=0.9, top_k=20, repetition_penalty=1.0):
    images = list(images)
    tokens = prompt_ids(handle.tokenizer, prompt, images, INTERLEAVE_SYSTEM, "" if think_mode else EMPTY_THINK)
    logits, keys, values, position = text_prefill(handle, tokens, images)
    negative_tokens = prompt_ids(handle.tokenizer, "<image>" * len(images), images)
    output_images, segments, generated, chunk = [], [], [], []
    generator = torch.Generator(device=handle.text_model.load_device).manual_seed(seed)
    progress = comfy.utils.ProgressBar(max_new_tokens)
    handle.text_model.pre_run()
    try:
        for index in range(max_new_tokens):
            mm.throw_exception_if_processing_interrupted()
            token = choose_token(logits, generated, generator, do_sample, temperature, top_p, top_k, repetition_penalty)
            if token == native.IM_END_ID:
                break
            if token == native.IMAGE_START_ID and len(output_images) >= max_images:
                break
            tokens.append(token)
            generated.append(token)
            logits, keys, values = text_step(handle, token, keys, values, position)
            position += 1
            if token != native.IMAGE_START_ID:
                chunk.append(token)
            else:
                if chunk:
                    segments.append({"type": "text", "text": handle.tokenizer.decode(chunk, skip_special_tokens=True)})
                    chunk = []
                prefix = (keys, values, torch.tensor([position], dtype=torch.long, device=handle.image_model.load_device))
                handle.text_model.cleanup()
                negative_tokens.append(native.IMAGE_START_ID)
                image = sample_image(handle, tokens, images, width, height, seed + len(output_images), steps, cfg, image_cfg, shift, conditional_prefix=prefix, image_only_tokens=negative_tokens)
                del keys, values, prefix, logits
                output_images.append(image)
                segments.append({"type": "image", "index": len(output_images) - 1})
                images.append(image)
                count = math.ceil(height / 32) * math.ceil(width / 32)
                appended = [native.IMAGE_CONTEXT_ID] * count + [native.IMAGE_END_ID]
                tokens.extend(appended)
                negative_tokens.extend(appended)
                logits, keys, values, position = text_prefill(handle, tokens, images)
                handle.text_model.pre_run()
            progress.update_absolute(index + 1)
    finally:
        handle.text_model.cleanup()
    if chunk:
        segments.append({"type": "text", "text": handle.tokenizer.decode(chunk, skip_special_tokens=True)})
    text = "".join(segment["text"] if segment["type"] == "text" else f"<image{segment['index'] + 1}>" for segment in segments)
    return text, output_images, json.dumps(segments, ensure_ascii=False)
