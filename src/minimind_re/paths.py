"""Portable publication artifact root; no directory creation or model execution."""
import os
from pathlib import Path

def artifact_root():
    return Path(os.environ.get("MODEL_DISSECT_DATA", ".model-dissect-data")).expanduser().resolve()

def artifact_path(relative=""):
    return artifact_root() / relative
