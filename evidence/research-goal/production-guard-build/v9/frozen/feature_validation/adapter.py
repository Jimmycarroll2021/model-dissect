"""NumPy FP32 synthetic seam, explicitly separate from any model integration."""
from dataclasses import dataclass
import numpy as np
from .contract import Binding, InvalidEvidence


def checked(value, shape=None):
    a = np.asarray(value)
    if a.dtype != np.float32 or (shape is not None and a.shape != shape) or not np.isfinite(a).all():
        raise InvalidEvidence('expected finite FP32 array with bound geometry')
    return a


def gain_free_rms(residual, epsilon):
    x = checked(residual)
    if x.ndim != 3 or x.shape[0] != 1 or not 1 <= x.shape[1] <= 64 or x.shape[2] == 0:
        raise InvalidEvidence('expected batch-one residual [1, tokens<=64, width]')
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise InvalidEvidence('invalid RMS epsilon')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            result = x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True, dtype=np.float32) + np.float32(epsilon))
        except FloatingPointError as exc:
            raise InvalidEvidence('nonfinite RMS intermediate') from exc
    return checked(result)


@dataclass(frozen=True)
class Dictionary:
    encoder: np.ndarray
    encoder_bias: np.ndarray
    decoder: np.ndarray
    decoder_bias: np.ndarray
    threshold: np.ndarray

    def snapshot(self, binding):
        f, d = binding.features, binding.width
        values = [checked(a, s).copy() for a, s in zip(
            (self.encoder, self.encoder_bias, self.decoder, self.decoder_bias, self.threshold),
            ((f,d), (f,), (f,d), (d,), (f,)))]
        if (values[-1] < 0).any():
            raise InvalidEvidence('negative JumpReLU threshold')
        for a in values:
            a.flags.writeable = False
        return Dictionary(*values)

    def encode(self, normalized):
        z = checked(normalized @ self.encoder.T + self.encoder_bias)
        # Deliberately no SAE decoder-bias centering and no activation scaling.
        return checked(np.where(z > self.threshold, z, np.float32(0)))

    def decode(self, coefficients):
        return checked(coefficients @ self.decoder + self.decoder_bias)


class EditSession:
    """One synthetic forward's capture/edit transaction; never installs model hooks.

    Each desired coefficient is relative to the stored baseline, so rescue cannot
    accidentally recompute an error term and cancel an intervention. Call downstream
    separately on returned output plus the original residual; no attention is cached.
    """
    def __init__(self, binding: Binding, dictionary: Dictionary):
        binding.require_synthetic()
        self.binding = binding
        self.dictionary = dictionary.snapshot(binding)
        self.active = False
        self.capture = None

    def __enter__(self):
        if self.active:
            raise InvalidEvidence('nested transaction')
        self.active = True
        return self

    def __exit__(self, *exc):
        self.capture = None
        self.active = False

    def capture_boundary(self, residual, original_output, *, layer, input_site, output_site):
        if not self.active or self.capture is not None:
            raise InvalidEvidence('capture requires a fresh active transaction')
        b = self.binding
        if (layer, input_site, output_site) != (b.layer, b.input_site, b.output_site):
            raise InvalidEvidence('wrong layer or boundary')
        x = checked(residual)
        u = gain_free_rms(x, b.epsilon)
        if x.shape[-1] != b.width:
            raise InvalidEvidence('wrong residual width')
        y = checked(original_output, x.shape)
        c = self.dictionary.encode(u)
        reconstruction = self.dictionary.decode(c)
        self.capture = {k: v.copy() for k, v in {
            'pre_mlp_residual': x, 'normalized_input': u, 'coefficients': c,
            'original_output': y, 'reconstruction': reconstruction,
            'reconstruction_residual': checked(y-reconstruction)}.items()}
        for value in self.capture.values():
            value.flags.writeable = False

    def edit(self, new_coefficient, *, position, feature, enabled=True):
        if not self.active or self.capture is None:
            raise InvalidEvidence('missing active boundary capture')
        b, c = self.binding, self.capture
        y = c['original_output']
        if type(position) is not int or position != y.shape[1]-1 or feature != b.feature:
            raise InvalidEvidence('only bound feature at final real token is supported')
        if not np.isscalar(new_coefficient) or not np.isfinite(new_coefficient) or new_coefficient < 0:
            raise InvalidEvidence('invalid coefficient')
        old = c['coefficients'][0, position, feature]
        new = np.float32(new_coefficient)
        if not np.isfinite(new):
            raise InvalidEvidence('coefficient cannot be represented as FP32')
        result, delta = y.copy(), np.zeros_like(y)
        if enabled and new != old:
            delta[0, position] = (new-old) * self.dictionary.decoder[feature]
            result[0, position] += delta[0, position]
        return {'delta': checked(delta), 'edited_output': checked(result)}
