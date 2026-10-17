"""Default model-scoped output locations for collection and selection."""
import json
from pathlib import Path


def model_output_dir(model_path, stage):
    name = Path(model_path).resolve().name
    if not name:
        raise ValueError("The BF16 model path must name a model directory")
    return Path("outputs") / name / stage


def collection_output_dir(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    model = config_path.parent / config["models"]["bf16"]
    return model_output_dir(model, "collection")
