"""Training loops, imported lazily to keep optional dependencies isolated."""

from importlib import import_module


def __getattr__(name):
    if name in {"idfraud_trainer"}:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["idfraud_trainer"]
