"""Helpers for dynamically importing classes based on dotted paths."""

from __future__ import annotations

import importlib


def load_attr(path: str):
    module_name, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)

