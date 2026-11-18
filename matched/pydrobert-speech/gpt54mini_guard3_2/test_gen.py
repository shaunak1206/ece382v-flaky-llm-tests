import numpy as np
import pytest

from pydrobert.speech import config
from pydrobert.speech.filters import (
    LinearFilterBank,
    TriangularOverlappingFilterBank,
    Fbank,
    GaborFilterBank,
    ComplexGammatoneFilterBank,
    BartlettWindow,
    BlackmanWindow,
    HammingWindow,
    HannWindow,
    GammaWindow,
)
from pydrobert.speech.scales import MelScaling


class _DummyLinearFilterBank(LinearFilterBank):
    def __init__(self, supports, sampling_rate=1000):
        self._supports = tuple(supports)
        self._rate = sampling_rate

    @property
    def is_real(self):
        return True

    @property
    def is_analytic(self):
        return False

    @property
    def is_zero_phase(self):
        return True

    @property
    def num_filts(self):
        return len(self._supports)

    @property
    def sampling_rate(self):
        return self._rate

    @property
    def supports_hz(self):
        return tuple((0.0, 0.0) for _ in self._supports)

    @property
    def supports(self):
        return self._supports

    def get_impulse_response(self, filt_idx, width):
        return np.zeros(width, dtype=np.float64)

    def get_frequency_response(self, filt_idx, width, half=False):
        return np.zeros(width // 2 + 1 if half else width, dtype=np.float64)

    def get_truncated_response(self, filt_idx, width):
        return 0, np.zeros(0, dtype=np.float64)


def test_linear_filter_bank_supports_ms_converts_samples_to_milliseconds():
    bank = _DummyLinearFilterBank([(-2, 3), (4, 10)], sampling_rate=2000)
    assert bank.supports_ms == ((-1.0, 1.5), (2.0, 5.0))


def test_triangular_constructor_rejects_invalid_frequency_ranges():
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), low_hz=-1)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), low_hz=2000, high_hz=1000, sampling_rate=8000)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), low_hz=20, high_hz=5000, sampling_rate=8000)


def test_triangular_properties_have_expected_shapes_and_ordering():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=4, low_hz=20, high_hz=4000, sampling_rate=8000)
    assert bank.num_filts == 4
    assert bank.is_zero_phase is True
    assert bank.is_real is True
    assert bank.is_analytic is False
    assert len(bank.centers_hz) == 4
    assert len(bank.supports_hz) == 4
    assert len(bank.supports) == 4
    assert all(lo < hi for lo, hi in bank.supports_hz)
    assert all(s[0] < s[1] for s in bank.supports)


def test_triangular_frequency_response_peak_matches_center_bin():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=3, low_hz=100, high_hz=3500, sampling_rate=8000)
    width = 256
    idx = 1
    left, mid, right = bank._vertices[idx:idx + 3]
    resp = bank.get_frequency_response(idx, width, half=True)
    center_bin = int(round(width * mid / bank.sampling_rate))
    assert resp[center_bin] == pytest.approx(1.0, abs=1e-12)
    left_bin = int(np.ceil(width * left / bank.sampling_rate))
    right_bin = int(width * right / bank.sampling_rate)
    assert resp[left_bin] == pytest.approx(0.0, abs=1e-12)
    assert resp[right_bin] == pytest.approx(0.0, abs=1e-12)


def test_triangular_full_frequency_response_matches_half_spectrum_for_real_bank():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=2, low_hz=100, high_hz=3000, sampling_rate=8000)
    width = 128
    resp_half = bank.get_frequency_response(0, width, half=True)
    resp_full = bank.get_frequency_response(0, width, half=False)
    half_width = width // 2 + 1
    assert np.allclose(resp_full[:half_width], resp_half)
    assert np.allclose(resp_full, resp_full[::-1])


def test_triangular_truncated_response_reconstructs_half_spectrum():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=3, low_hz=80, high_hz=3500, sampling_rate=8000)
    width = 256
    idx = 0
    left_idx, trnc = bank.get_truncated_response(idx, width)
    half = bank.get_frequency_response(idx, width, half=True)
    recon = np.zeros_like(half)
    recon[left_idx:left_idx + len(trnc)] = trnc
    assert np.allclose(recon, half)


def test_triangular_invalid_frequency_response_width_asserts_for_too_small_width():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=3, low_hz=80, high_hz=3500, sampling_rate=8000)
    with pytest.raises(AssertionError):
        bank.get_frequency_response(0, 2, half=True)
    with pytest.raises(AssertionError):
        bank.get_truncated_response(0, 2)


def test_fbank_is_square_root_of_triangular_response_for_nonanalytic_bank():
    tri = TriangularOverlappingFilterBank(MelScaling(), num_filts=4, low_hz=100, high_hz=3000, sampling_rate=8000)
    fbank = Fbank(num_filts=4, low_hz=100, high_hz=3000, sampling_rate=8000, analytic=False)
    width = 256
    idx = 2
    tri_half = tri.get_frequency_response(idx, width, half=True)
    fb_half = fbank.get_frequency_response(idx, width, half=True)
    assert np.allclose(fb_half ** 2, tri_half, atol=1e-12, rtol=1e-12)


def test_fbank_truncated_response_squares_to_triangular_truncated_response():
    tri = TriangularOverlappingFilterBank(MelScaling(), num_filts=3, low_hz=100, high_hz=3000, sampling_rate=8000)
    fbank = Fbank(num_filts=3, low_hz=100, high_hz=3000, sampling_rate=8000, analytic=False)
    width = 256
    idx = 1
    tri_left, tri_trnc = tri.get_truncated_response(idx, width)
    fb_left, fb_trnc = fbank.get_truncated_response(idx, width)
    assert tri_left == fb_left
    assert np.allclose(fb_trnc ** 2, tri_trnc, atol=1e-12, rtol=1e-12)


def test_gabor_frequency_response_is_complex_and_nonnegative_magnitude():
    bank = GaborFilterBank(MelScaling(), num_filts=3, low_hz=100, high_hz=3000, sampling_rate=8000)
    width = 128
    resp = bank.get_frequency_response(1, width, half=False)
    assert np.iscomplexobj(resp)
    assert np.all(np.isfinite(resp))
    assert np.all(np.abs(resp) >= 0)


def test_gabor_truncated_response_reconstructs_full_frequency_response():
    bank = GaborFilterBank(MelScaling(), num_filts=3, low_hz=100, high_hz=3000, sampling_rate=8000)
    width = 128
    idx = 1
    left_idx, trnc = bank.get_truncated_response(idx, width)
    full = bank.get_frequency_response(idx, width, half=False)
    recon = np.zeros(width, dtype=trnc.dtype)
    wrap = min(left_idx + len(trnc), width) - left_idx
    recon[left_idx:left_idx + wrap] = trnc[:wrap]
    recon[:len(trnc) - wrap] = trnc[wrap:]
    assert np.allclose(recon, full, atol=1e-12, rtol=1e-12)


def test_complex_gammatone_rejects_invalid_order():
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=0)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=-3)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=2.5)


def test_complex_gammatone_frequency_response_and_truncated_response_shapes():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=2, low_hz=100, high_hz=3000, sampling_rate=8000, order=4)
    width = 64
    resp = bank.get_frequency_response(0, width, half=True)
    left_idx, trnc = bank.get_truncated_response(0, width)
    assert np.iscomplexobj(resp)
    assert np.iscomplexobj(trnc)
    assert resp.shape == (width // 2 + 1,)
    assert 0 <= left_idx < width
    assert trnc.ndim == 1
    assert trnc.size > 0


def test_window_functions_return_expected_endpoint_values():
    width = 9
    assert BartlettWindow().get_impulse_response(width)[0] == pytest.approx(0.0, abs=1e-15)
    assert BlackmanWindow().get_impulse_response(width)[0] == pytest.approx(0.0, abs=1e-15)
    assert HammingWindow().get_impulse_response(width)[0] == pytest.approx(0.0, abs=1e-15)
    assert HannWindow().get_impulse_response(width)[0] == pytest.approx(0.0, abs=1e-15)


def test_gamma_window_handles_edge_cases_and_has_expected_peak_location():
    assert GammaWindow(order=3, peak=0.75).get_impulse_response(0).size == 0
    assert GammaWindow(order=3, peak=0.75).get_impulse_response(1).tolist() == [1.0]
    win = GammaWindow(order=4, peak=0.75).get_impulse_response(16)
    assert win.shape == (16,)
    assert np.all(np.isfinite(win))
    assert np.argmax(win) in {10, 11, 12}  # near the requested peak


def test_triangular_and_fbank_supports_have_same_number_of_filters():
    tri = TriangularOverlappingFilterBank(MelScaling(), num_filts=5, low_hz=50, high_hz=3000, sampling_rate=8000)
    fb = Fbank(num_filts=5, low_hz=50, high_hz=3000, sampling_rate=8000)
    assert tri.num_filts == fb.num_filts == 5
    assert len(tri.supports_hz) == len(fb.supports_hz) == 5
