import json
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def load_config(config_path: str) -> dict:
    """
    Load experiment configuration from JSON or YAML file.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if path.suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)

    if path.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("pyyaml is required to load YAML configs")
        with open(path, "r") as f:
            return yaml.safe_load(f)

    raise ValueError("Config file must be .json or .yaml")
