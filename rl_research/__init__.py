"""Top-level package for the rl-research framework."""

from importlib import metadata


def __getattr__(name: str) -> str:
    if name == "__version__":
        try:
            return metadata.version("rl-research")
        except metadata.PackageNotFoundError:  # pragma: no cover - during local development
            return "0.0.0"
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = ["__version__"]
