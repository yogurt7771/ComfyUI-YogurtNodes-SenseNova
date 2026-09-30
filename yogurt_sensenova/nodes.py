from comfy_api.latest import io

from . import runtime
from .loader import apply_lora, load_model
from .model_paths import MODELS


SenseNova = io.Custom("YOGURT_SENSENOVA")


def references():
    return io.Autogrow.Input("images", template=io.Autogrow.TemplatePrefix(io.Image.Input("image"), prefix="image", min=0, max=100), optional=True,
                             tooltip="Ordered reference images. Each socket can also contain an image batch.")


def image_settings():
    return [
        io.Int.Input("width", default=1536, min=32, max=16384, step=32),
        io.Int.Input("height", default=1536, min=32, max=16384, step=32),
        io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF),
        io.Int.Input("steps", default=50, min=1, max=1000),
        io.Float.Input("cfg", default=4.0, min=0.0, max=100.0, step=0.1),
        io.Float.Input("image_cfg", default=1.0, min=0.0, max=100.0, step=0.1),
        io.Float.Input("shift", default=3.0, min=0.01, max=100.0, step=0.01),
    ]


def text_settings():
    return [
        io.Boolean.Input("do_sample", default=False),
        io.Float.Input("temperature", default=0.7, min=0.0, max=100.0, step=0.05),
        io.Float.Input("top_p", default=0.9, min=0.0, max=1.0, step=0.01),
        io.Int.Input("top_k", default=20, min=0, max=151936),
        io.Float.Input("repetition_penalty", default=1.0, min=0.01, max=100.0, step=0.05),
    ]


class YogurtSenseNovaModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="YogurtSenseNovaModelLoader", display_name="Yogurt SenseNova Model Loader", category="YogurtSenseNova/Loaders",
                         description="Load full official U1.5 RL/SFT Safetensors shards. MODEL, CLIP and VAE also work with native ComfyUI nodes.",
                         inputs=[io.Combo.Input("model_name", options=list(MODELS)),
                                 io.String.Input("model_path", default="", tooltip="Optional local checkpoint directory or complete Safetensors file."),
                                 io.Combo.Input("dtype", options=["auto", "bfloat16", "float32"]),
                                 io.Boolean.Input("download_model", default=False, tooltip="Explicitly download missing official weights to the configured SenseNovaU1 model directory; model_path must be empty.")],
                         outputs=[SenseNova.Output(display_name="sensenova"), io.Model.Output(), io.Clip.Output(), io.Vae.Output()])

    @classmethod
    def execute(cls, model_name, model_path="", dtype="auto", download_model=False):
        handle = load_model(model_name, model_path, dtype, download_model)
        return io.NodeOutput(handle, handle.image_model, handle.clip, handle.vae)


class YogurtSenseNovaApplyLoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="YogurtSenseNovaApplyLoRA", display_name="Yogurt SenseNova Apply LoRA", category="YogurtSenseNova/Loaders",
                         inputs=[SenseNova.Input("sensenova"), io.String.Input("lora_path", default=""), io.Float.Input("strength", default=1.0, min=-100.0, max=100.0, step=0.05)],
                         outputs=[SenseNova.Output(display_name="sensenova"), io.Model.Output()])

    @classmethod
    def execute(cls, sensenova, lora_path, strength=1.0):
        handle = apply_lora(sensenova, lora_path, strength)
        return io.NodeOutput(handle, handle.image_model)


class YogurtSenseNovaGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="YogurtSenseNovaGenerate", display_name="Yogurt SenseNova Generate / Edit", category="YogurtSenseNova",
                         inputs=[SenseNova.Input("sensenova"), io.String.Input("prompt", multiline=True), *image_settings(),
                                 io.Int.Input("batch_size", default=1, min=1, max=64), references()], outputs=[io.Image.Output()])

    @classmethod
    def execute(cls, sensenova, prompt, width=1536, height=1536, seed=42, steps=50, cfg=4.0, image_cfg=1.0, shift=3.0, batch_size=1, images=None):
        image = runtime.generate(sensenova, prompt, runtime.ordered_images(images), width=width, height=height, seed=seed, steps=steps, cfg=cfg, image_cfg=image_cfg, shift=shift, batch_size=batch_size)
        return io.NodeOutput(image)


class YogurtSenseNovaVQA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="YogurtSenseNovaVQA", display_name="Yogurt SenseNova Visual Q&A", category="YogurtSenseNova", is_output_node=True,
                         inputs=[SenseNova.Input("sensenova"), io.String.Input("question", multiline=True),
                                 io.Int.Input("max_new_tokens", default=1024, min=1, max=262144),
                                 io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF), *text_settings(), references()],
                         outputs=[io.String.Output(display_name="answer")])

    @classmethod
    def execute(cls, sensenova, question, max_new_tokens=1024, seed=42, do_sample=False, temperature=0.7, top_p=0.9, top_k=20, repetition_penalty=1.0, images=None):
        text = runtime.answer(sensenova, question, runtime.ordered_images(images), max_new_tokens, seed, do_sample, temperature, top_p, top_k, repetition_penalty)
        return io.NodeOutput(text, ui={"text": [text]})


class YogurtSenseNovaInterleave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="YogurtSenseNovaInterleave", display_name="Yogurt SenseNova Interleaved Generation", category="YogurtSenseNova", is_output_node=True,
                         inputs=[SenseNova.Input("sensenova"), io.String.Input("prompt", multiline=True), *image_settings(),
                                 io.Int.Input("max_images", default=4, min=0, max=1024),
                                 io.Int.Input("max_new_tokens", default=8192, min=1, max=262144),
                                 io.Boolean.Input("think_mode", default=False), *text_settings(), references()],
                         outputs=[io.String.Output(display_name="text"), io.Image.Output(display_name="images", is_output_list=True), io.String.Output(display_name="content_json")])

    @classmethod
    def execute(cls, sensenova, prompt, width=1536, height=1536, seed=42, steps=50, cfg=4.0, image_cfg=1.0, shift=3.0, max_images=4, max_new_tokens=8192, think_mode=False, do_sample=False, temperature=0.7, top_p=0.9, top_k=20, repetition_penalty=1.0, images=None):
        text, output_images, content = runtime.interleave(sensenova, prompt, runtime.ordered_images(images), width, height, seed, steps, cfg, image_cfg, shift, max_images, max_new_tokens, think_mode, do_sample, temperature, top_p, top_k, repetition_penalty)
        return io.NodeOutput(text, output_images, content, ui={"text": [text]})


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (YogurtSenseNovaModelLoader, YogurtSenseNovaApplyLoRA, YogurtSenseNovaGenerate, YogurtSenseNovaVQA, YogurtSenseNovaInterleave)}
NODE_DISPLAY_NAME_MAPPINGS = {name: cls.define_schema().display_name for name, cls in NODE_CLASS_MAPPINGS.items()}
