"""Configuration: one YAML file plus command-line overrides ("key=value")."""
from pathlib import Path
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


class Config(dict):
    """A dict with attribute access: cfg.epochs == cfg['epochs']."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


def apply_overrides(cfg, overrides):
    """Apply ["epochs=2", "act=relu", ...] in place. Unknown keys are errors (catches typos)."""
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override '{item}' must look like key=value")
        if key not in cfg:
            raise KeyError(f"unknown config key '{key}'. Valid keys: {sorted(cfg)}")
        value = yaml.safe_load(raw)
        default = cfg[key]
        if isinstance(default, (int, float)) and not isinstance(default, bool) and isinstance(value, (int, float, str)):
            value = int(float(value)) if isinstance(default, int) else float(value)   # '1e-4' -> 0.0001
        cfg[key] = value
    return cfg


def load_config(path=None, overrides=None):
    with open(path or DEFAULT_CONFIG) as f:
        cfg = Config(yaml.safe_load(f))
    return apply_overrides(cfg, overrides)
