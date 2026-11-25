"""
config/__init__.py
"""
import os
import importlib.util
from config.settings import AppSettings, TradingMode, settings
from config.symbols import SymbolConfig, SYMBOL_REGISTRY

# Backward compatibility for legacy MT5 config references
cfg = None
try:
    spec = importlib.util.spec_from_file_location(
        "root_config",
        os.path.join(os.path.dirname(__file__), "..", "config.py")
    )
    if spec and spec.loader:
        root_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(root_config)
        cfg = getattr(root_config, "cfg", None)
except Exception:
    pass

__all__ = ["AppSettings", "TradingMode", "settings", "SymbolConfig", "SYMBOL_REGISTRY", "cfg"]
