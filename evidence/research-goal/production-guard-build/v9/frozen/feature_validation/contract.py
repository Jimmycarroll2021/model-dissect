"""Strict, versioned provenance contract. Identity syntax is not identity proof."""
from dataclasses import dataclass
import re


class InvalidEvidence(ValueError):
    pass


@dataclass(frozen=True)
class Artifact:
    role: str
    revision: str
    sha256: str

    def validate(self):
        if not self.role or not self.revision or not re.fullmatch('[0-9a-f]{64}', self.sha256):
            raise InvalidEvidence('missing or invalid immutable artifact identity')


@dataclass(frozen=True)
class Binding:
    artifacts: tuple[Artifact, ...]
    layer: int
    feature: int
    width: int
    features: int
    epsilon: float
    kind: str = 'synthetic'
    eligibility: str = 'ENGINEERING_ONLY'
    input_site: str = 'pre_feedforward_layernorm.input'
    output_site: str = 'post_feedforward_layernorm.output'
    downstream: str = 'ordinary_original'
    encoder_layout: str = 'features_by_width'
    activation: str = 'jumprelu_strict_gt'
    input_centering: str = 'none'
    activation_normalization: str = 'none'
    schema: int = 1

    def validate(self):
        import math
        roles = [a.role for a in self.artifacts]
        required = {'model', 'config', 'tokenizer', 'transcoder', 'dictionary_metadata',
                    'runtime', 'corpus', 'protocol'}
        if set(roles) != required or len(roles) != len(set(roles)):
            raise InvalidEvidence('all distinct provenance roles required')
        for artifact in self.artifacts:
            artifact.validate()
        if any(type(v) is not int for v in (self.schema, self.width, self.features, self.layer, self.feature)):
            raise InvalidEvidence('geometry and indices must be integers')
        if self.schema != 1 or self.width <= 0 or self.features <= 0:
            raise InvalidEvidence('invalid schema or geometry')
        if not 0 <= self.feature < self.features or self.layer < 0:
            raise InvalidEvidence('invalid layer or feature')
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise InvalidEvidence('epsilon must be explicitly bound and positive')
        expected = ('pre_feedforward_layernorm.input', 'post_feedforward_layernorm.output',
                    'ordinary_original', 'features_by_width', 'jumprelu_strict_gt', 'none', 'none')
        if (self.input_site, self.output_site, self.downstream, self.encoder_layout,
            self.activation, self.input_centering, self.activation_normalization) != expected:
            raise InvalidEvidence('unsupported operation convention')
        if self.kind not in {'synthetic', 'target'}:
            raise InvalidEvidence('unknown evidence kind')
        if self.eligibility not in {'ENGINEERING_ONLY', 'INCOMPLETE'}:
            raise InvalidEvidence('this implementation cannot certify target eligibility')
        if self.kind == 'target' and (self.layer, self.feature) != (20, 15589):
            raise InvalidEvidence('frozen target identity mismatch')

    def require_synthetic(self):
        self.validate()
        if self.kind != 'synthetic' or self.eligibility != 'ENGINEERING_ONLY':
            raise InvalidEvidence('target runtime binding/admission is not implemented')


@dataclass(frozen=True)
class EvidencePacket:
    binding: Binding
    input_ids: tuple[int, ...]
    answer_ids: tuple[int, int]
    exposure: str
    arm: str
    arrays: tuple[Artifact, ...]
    status: str = 'INCOMPLETE'

    def validate(self):
        self.binding.validate()
        if not 1 <= len(self.input_ids) <= 64 or any(type(x) is not int or x < 0 for x in self.input_ids):
            raise InvalidEvidence('invalid input token identity')
        if len(self.answer_ids) != 2 or any(type(x) is not int or x < 0 for x in self.answer_ids) or self.answer_ids[0] == self.answer_ids[1]:
            raise InvalidEvidence('invalid answer token identity')
        if self.exposure not in {'discovery', 'engineering', 'held_out'}:
            raise InvalidEvidence('missing exposure classification')
        if self.binding.kind == 'synthetic' and self.exposure != 'engineering':
            raise InvalidEvidence('synthetic evidence is engineering only')
        arms = {'baseline', 'disabled', 'identity', 'ablation', 'matched', 'random', 'rescue', 'cross_patch', 'restoration'}
        if self.arm not in arms or self.status not in {'INCOMPLETE', 'ENGINEERING_ONLY'}:
            raise InvalidEvidence('invalid arm or unestablished scientific verdict')
        names = [a.role for a in self.arrays]
        required = {'pre_mlp_residual', 'normalized_input', 'coefficients', 'original_output',
                    'reconstruction', 'reconstruction_residual', 'delta', 'edited_output', 'final_logits'}
        if set(names) != required or len(names) != len(set(names)):
            raise InvalidEvidence('missing or duplicate raw array evidence')
        for artifact in self.arrays:
            artifact.validate()
