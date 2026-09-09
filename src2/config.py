"""src2/config.py — loads config.yaml and exposes get_config()."""
from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
_cached_config: dict | None = None


def get_config() -> dict:
    """Return the parsed config dict (cached after first load)."""
    global _cached_config
    if _cached_config is None:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
            _cached_config = yaml.safe_load(fh)
    return _cached_config
