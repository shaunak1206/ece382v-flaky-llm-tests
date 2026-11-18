import numpy as np
import pytest

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
from pydrobert.speech.util import hertz_to_angular


def test_linear_filter_bank_is_abstract():
    with pytest.raises(TypeError):
        LinearFilterBank()


def test_triangular_filter_bank_basic_properties():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=5, low_hz=20.0, high_hz=4000.0, sampling_rate=16000)
    assert bank.num_filts == 5
    assert bank.sampling_rate == 16000
    assert bank.is_real is True
    assert bank.is_analytic is False
    assert bank.is_zero_phase is True
    assert len(bank.centers_hz) == 5
    assert len(bank.supports_hz) == 5
    assert len(bank.supports) == 5
    assert len(bank.supports_ms) == 5
    assert all(lo < mid < hi for (lo, hi), mid in zip(bank.supports_hz, bank.centers_hz))


def test_triangular_filter_bank_invalid_frequency_range_raises():
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), low_hz=-1.0)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), low_hz=2000.0, high_hz=1000.0, sampling_rate=8000)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), low_hz=20.0, high_hz=5000.0, sampling_rate=8000)


def test_triangular_frequency_response_half_matches_expected_triangle():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=3, low_hz=20.0, high_hz=4000.0, sampling_rate=16000)
    width = 256
    idx = 1
    resp = bank.get_frequency_response(idx, width, half=True)
    left, mid, right = bank.supports_hz[idx][0], bank.centers_hz[idx], bank.supports_hz[idx][1]
    left_bin = int(np.ceil(width * left / bank.sampling_rate))
    mid_bin = int(np.ceil(width * mid / bank.sampling_rate))
    right_bin = int(width * right / bank.sampling_rate)
    assert resp.ndim == 1
    assert resp[left_bin] == pytest.approx(0.0, abs=1e-12)
    assert resp[mid_bin] == pytest.approx(1.0, rel=1e-12, abs=1e-12)
    assert resp[right_bin] == pytest.approx(0.0, abs=1e-12)
    assert np.all(resp >= -1e-12)


def test_triangular_frequency_response_full_real_is_symmetric():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=3, low_hz=20.0, high_hz=4000.0, sampling_rate=16000)
    width = 255
    resp = bank.get_frequency_response(0, width, half=False)
    assert resp.shape == (width,)
    assert np.allclose(resp.imag, 0.0)
    assert np.allclose(resp, resp[::-1])


def test_triangular_truncated_response_reconstructs_full_half_spectrum():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=4, low_hz=20.0, high_hz=4000.0, sampling_rate=16000)
    width = 257
    idx = 2
    bin_idx, trnc = bank.get_truncated_response(idx, width)
    full_half = bank.get_frequency_response(idx, width, half=True)
    recon = np.zeros_like(full_half)
    recon[bin_idx:bin_idx + len(trnc)] = trnc
    assert np.allclose(recon, full_half)


def test_fbank_isqrt_frequency_response_and_impulse_response_shape():
    bank = Fbank(num_filts=4, low_hz=20.0, high_hz=4000.0, sampling_rate=16000, analytic=False)
    width = 128
    resp = bank.get_frequency_response(1, width, half=True)
    imp = bank.get_impulse_response(1, width)
    assert resp.shape == (width // 2 + 1,)
    assert np.all(resp >= -1e-12)
    assert imp.shape == (width,)
    assert np.isrealobj(imp)
    assert np.max(resp) <= 1.0 + 1e-12


def test_fbank_truncated_response_reconstructs_half_spectrum():
    bank = Fbank(num_filts=4, low_hz=20.0, high_hz=4000.0, sampling_rate=16000, analytic=False)
    width = 129
    idx = 1
    bin_idx, trnc = bank.get_truncated_response(idx, width)
    half = bank.get_frequency_response(idx, width, half=True)
    recon = np.zeros_like(half)
    recon[bin_idx:bin_idx + len(trnc)] = trnc
    assert np.allclose(recon, half)


def test_gabor_filter_bank_basic_properties_and_shapes():
    bank = GaborFilterBank(MelScaling(), num_filts=3, low_hz=20.0, high_hz=4000.0, sampling_rate=16000)
    width = 64
    assert bank.is_real is False
    assert bank.is_zero_phase is True
    assert bank.num_filts == 3
    assert len(bank.centers_hz) == 3
    resp = bank.get_frequency_response(0, width, half=False)
    imp = bank.get_impulse_response(0, width)
    assert resp.shape == (width,)
    assert imp.shape == (width,)
    assert np.iscomplexobj(resp)
    assert np.iscomplexobj(imp)
    assert np.all(np.real(resp) >= 0.0)


def test_gabor_truncated_response_reconstructs_full_response():
    bank = GaborFilterBank(MelScaling(), num_filts=3, low_hz=20.0, high_hz=4000.0, sampling_rate=16000)
    width = 64
    idx = 1
    bin_idx, trnc = bank.get_truncated_response(idx, width)
    full = bank.get_frequency_response(idx, width, half=False)
    recon = np.zeros(width, dtype=trnc.dtype)
    wrap = min(bin_idx + len(trnc), width) - bin_idx
    recon[bin_idx:bin_idx + wrap] = trnc[:wrap]
    recon[:len(trnc) - wrap] = trnc[wrap:]
    assert np.allclose(recon, full)


def test_gabor_truncated_response_falls_back_to_full_when_wraps_entire_period():
    bank = GaborFilterBank(MelScaling(), num_filts=1, low_hz=20.0, high_hz=3000.0, sampling_rate=8000)
    width = 8
    idx = 0
    bin_idx, trnc = bank.get_truncated_response(idx, width)
    full = bank.get_frequency_response(idx, width, half=False)
    assert bin_idx == 0
    assert trnc.shape == full.shape
    assert np.allclose(trnc, full)


def test_complex_gammatone_invalid_order_raises():
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=0)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=-1)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=2.5)


def test_complex_gammatone_basic_properties_and_shapes():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=3, low_hz=20.0, high_hz=4000.0, sampling_rate=16000, order=4)
    width = 64
    assert bank.is_real is False
    assert bank.is_zero_phase is False
    assert bank.order == 4
    assert bank.num_filts == 3
    assert len(bank.centers_hz) == 3
    resp = bank.get_frequency_response(0, width, half=False)
    imp = bank.get_impulse_response(0, width)
    assert resp.shape == (width,)
    assert imp.shape == (width,)
    assert np.iscomplexobj(resp)
    assert np.iscomplexobj(imp)
    assert np.max(np.abs(resp)) > 0.0


def test_complex_gammatone_truncated_response_reconstructs_full_response():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=3, low_hz=20.0, high_hz=4000.0, sampling_rate=16000, order=4)
    width = 64
    idx = 1
    bin_idx, trnc = bank.get_truncated_response(idx, width)
    full = bank.get_frequency_response(idx, width, half=False)
    recon = np.zeros(width, dtype=trnc.dtype)
    wrap = min(bin_idx + len(trnc), width) - bin_idx
    recon[bin_idx:bin_idx + wrap] = trnc[:wrap]
    recon[:len(trnc) - wrap] = trnc[wrap:]
    assert np.allclose(recon, full)


def test_windows_have_expected_lengths_and_symmetry():
    for window in [BartlettWindow(), BlackmanWindow(), HammingWindow(), HannWindow()]:
        w = window.get_impulse_response(9)
        assert w.shape == (9,)
        assert np.allclose(w, w[::-1])
        assert np.isclose(w[0], 0.0)
        assert np.isclose(w[-1], 0.0)


def test_gamma_window_width_zero_and_one():
    gw = GammaWindow(order=4, peak=0.75)
    assert gw.get_impulse_response(0).shape == (0,)
    assert np.allclose(gw.get_impulse_response(1), np.array([1.0]))


def test_gamma_window_peak_location_and_nonnegativity():
    gw = GammaWindow(order=4, peak=0.75)
    w = gw.get_impulse_response(64)
    assert w.shape == (64,)
    assert np.all(w >= 0.0)
    assert np.argmax(w) in {15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}
