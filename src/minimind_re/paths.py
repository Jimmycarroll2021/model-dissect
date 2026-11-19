"""Portable isolated artifact paths; configuration checks are not an OS sandbox."""
import os
from pathlib import Path


def artifact_root():
    configured = os.environ.get("MODEL_DISSECT_DATA")
    if configured is None:
        requested = Path.cwd() / ".model-dissect-data"
    else:
        requested = Path(configured)
        if not configured or not requested.is_absolute():
            raise ValueError("MODEL_DISSECT_DATA must name an absolute dedicated directory")
    root = requested.resolve()
    checkout = Path(__file__).resolve().parents[2]
    # Never allow a broad location containing the checkout, working directory,
    # or home. Resolve first so symlinks cannot disguise those locations.
    protected = (Path.home().resolve(), Path.cwd().resolve(), checkout)
    if root == Path(root.anchor) or root.parent == Path(root.anchor) or any(
        location.is_relative_to(root) for location in protected
    ):
        raise ValueError("MODEL_DISSECT_DATA must be a dedicated artifact directory, not a broad root")
    return root


def artifact_path(relative=""):
    root = artifact_root()
    component = Path(relative)
    if component.is_absolute() or ".." in component.parts:
        raise ValueError("Artifact path must be relative and contain no parent traversal")
    result = (root / component).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Artifact path resolves outside its dedicated root")
    return result
