"""Local safetensors identity checks; never deserialize pickle checkpoints."""
import hashlib
from pathlib import Path
from collections.abc import Mapping


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(directory, expected_hashes: Mapping[str, str], required):
    """Expected hashes come from the separately admitted artifact manifest."""
    root = Path(directory).resolve(strict=True)
    paths = {}
    for name in required:
        expected = expected_hashes.get(name)
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"Missing/invalid expected SHA256: {name}")
        if Path(name).name != name:
            raise ValueError("Only direct child filenames are accepted")
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Not a regular local artifact: {name}")
        if sha256(path) != expected:
            raise ValueError(f"SHA256 mismatch: {name}")
        paths[name] = path
    return paths
