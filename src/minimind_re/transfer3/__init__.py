"""Source-assisted third-model full-prefill candidate."""
from .gpt2_candidate import GPT2Candidate, load_admitted_artifact

__all__ = ["GPT2Candidate", "load_admitted_artifact"]
