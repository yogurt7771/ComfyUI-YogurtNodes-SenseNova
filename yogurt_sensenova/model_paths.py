import json
from pathlib import Path

import folder_paths
from huggingface_hub import hf_hub_download
from safetensors import safe_open


MODELS = ("SenseNova-U1.5-8B-MoT", "SenseNova-U1.5-8B-MoT-SFT")
folder_paths.add_model_folder_path("SenseNovaU1", str(Path(folder_paths.models_dir) / "SenseNovaU1"))


def contained(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Model file must stay inside {root}: {name}")
    return path


def read_index(directory):
    with (directory / "model.safetensors.index.json").open(encoding="utf-8") as source:
        return json.load(source)["weight_map"]


def resolve_model(model_name, model_path="", download_model=False):
    if model_name not in MODELS:
        raise ValueError(f"Unknown official model: {model_name}")
    roots = [Path(p) for p in folder_paths.get_folder_paths("SenseNovaU1")]
    if model_path.strip():
        directory = Path(model_path.strip()).resolve()
        if directory.is_file():
            if directory.suffix != ".safetensors":
                raise ValueError("SenseNova checkpoints must use Safetensors.")
            with safe_open(str(directory), framework="pt"):
                pass
            return [directory], None
    else:
        candidates = [contained(root, name) for root in roots for name in (model_name, f"sensenova--{model_name}")]
        directory = next((p for p in candidates if (p / "model.safetensors.index.json").is_file()), candidates[0])
    index_path = directory / "model.safetensors.index.json"
    if download_model and not model_path.strip():
        if not index_path.is_file():
            directory = contained(roots[0], model_name)
            hf_hub_download(f"sensenova/{model_name}", "model.safetensors.index.json", local_dir=str(directory), token=False)
        weight_map = read_index(directory)
        missing = [name for name in sorted(set(weight_map.values())) if not contained(directory, name).is_file()]
        if missing and directory.parent not in [p.resolve() for p in roots]:
            raise ValueError("Automatic downloads require a configured SenseNovaU1 model directory.")
        for name in missing:
            hf_hub_download(f"sensenova/{model_name}", name, local_dir=str(directory), token=False)
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing {index_path}. Select an official model directory or enable download_model without model_path.")
    weight_map = read_index(directory)
    files = [contained(directory, name) for name in sorted(set(weight_map.values()))]
    missing = [p.name for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete SenseNova model in {directory}: {', '.join(missing)}")
    for path in files:
        with safe_open(str(path), framework="pt") as source:
            expected = {key for key, name in weight_map.items() if name == path.name}
            absent = expected - set(source.keys())
            if absent:
                raise ValueError(f"Model shard {path} is missing {next(iter(absent))}.")
    return files, weight_map
