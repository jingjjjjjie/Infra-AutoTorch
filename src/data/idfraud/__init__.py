'''
Data package: dataset, dataloader, transforms, preprocessing.
'''
from .transforms import build_transform


def __getattr__(name):
    """Keep training-only dataloader imports out of inference startup."""
    if name == "create_dataloaders":
        from .dataloader import create_dataloaders

        return create_dataloaders
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["build_transform", "create_dataloaders"]
