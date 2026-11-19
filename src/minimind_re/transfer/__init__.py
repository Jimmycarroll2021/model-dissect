"""Architecture-specific second-model transfer; no native model forward dependency."""
from .gpt_neox_candidate import GPTNeoXCandidate, NeoXConfig, CandidateOutput

__all__ = ["GPTNeoXCandidate", "NeoXConfig", "CandidateOutput"]
