"""Keep synthetic temporary files in the configured portable artifact root."""
from minimind_re.paths import artifact_path
artifact_path().mkdir(parents=True, exist_ok=True)
