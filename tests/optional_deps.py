from __future__ import annotations

import importlib.util
import unittest


def require_modules(*module_names: str) -> None:
    missing = [
        module_name
        for module_name in module_names
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        raise unittest.SkipTest(
            "optional test dependencies are unavailable: " + ", ".join(sorted(missing))
        )
