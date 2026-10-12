"""
src/__init__.py  –  Package initialiser.
Loads central configuration from config/settings.yaml.
"""
import os
import sys
import types
import yaml
import logging

# Ensure Windows Application Control does not block optional sklearn native modules
if sys.platform == "win32":
    _blocked_mods = [
        "sklearn.svm._libsvm_sparse",
        "sklearn.svm._libsvm",
        "sklearn.svm._liblinear",
    ]
    for _mod in _blocked_mods:
        if _mod not in sys.modules:
            try:
                __import__(_mod)
            except (ImportError, OSError):
                sys.modules[_mod] = types.ModuleType(_mod)

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "settings.yaml",
)


def load_config(path: str | None = None) -> dict:
    """Return the project configuration dictionary."""
    path = path or _CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()

# --------------- logging setup ---------------
_log_cfg = CONFIG.get("logging", {})
logging.basicConfig(
    level=getattr(logging, _log_cfg.get("level", "INFO")),
    format=_log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"),
)
logger = logging.getLogger("shipment_eta")
