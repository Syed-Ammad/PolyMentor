import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load a YAML config file and return as a dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_all_configs(config_dir: str = "configs") -> dict:
    """Load all configs from the configs/ directory."""
    config_dir = Path(config_dir)
    configs = {}
    for yaml_file in config_dir.glob("*.yaml"):
        name = yaml_file.stem.replace("_config", "")
        configs[name] = load_config(str(yaml_file))
    return configs
