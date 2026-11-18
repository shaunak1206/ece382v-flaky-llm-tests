import math

import numpy as np
import pytest

from pydrobert.speech import config
from pydrobert.speech.filters import (
    BartlettWindow,
    BlackmanWindow,
    ComplexGammatoneFilterBank,
    Fbank,
    GammaWindow,
    GaborFilterBank,
    HammingWindow,
    HannWindow,
    TriangularOverlappingFilterBank,
)


def _reconstruct_from_truncated_real(width, bin_idx, trnc):
    full = np.zeros(width, dtype=trnc.dtype)
    full[bin_idx:bin_idx + len(trnc)] = trnc
    return full


def _reconstruct_from_truncated_complex(width, bin_idx, trnc):
    full = np.zeros(width, dtype=trnc.dtype)
    wrap = min(bin_idx + len(trnc), width) - bin_idx
    full[bin_idx:bin_idx + wrap] = trnc[:wrap]
    full[:len(trnc) - wrap] = trnc[wrap:]
    return full


def test_triangular_bank_basic_properties():
    bank = TriangularOverlappingFilterBank("mel", num_filts=5, low_hz=100, high_hz=3000, sampling_rate=8000)
    assert bank.num_filts == 5
    assert bank.sampling_rate == 8000
    assert bank.is_zero_phase is True
    assert bank.is_real is True
    assert bank.is_analytic is False
    assert len(bank.centers_hz) == 5
    assert len(bank.supports_hz) == 5
    assert len(bank.supports) == 5


def test_triangular_bank_invalid_frequency_range_raises():
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank("mel", low_hz=-1)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank("mel", low_hz=1000, high_hz=500)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank("mel", low_hz=100, high_hz=50000, sampling_rate=16000)


def test_triangular_frequency_response_has_triangular_shape():
    bank = TriangularOverlappingFilterBank("mel", num_filts=3, low_hz=100, high_hz=2000, sampling_rate=8000)
    width = 256
    resp = bank.get_frequency_response(1, width, half=False)
    left, mid, right = bank.supports_hz[1]
    left_idx = int(np.ceil(width * left / bank.sampling_rate))
    mid_idx = int(np.round(width * bank.centers_hz[1] / bank.sampling_rate))
    right_idx = int(width * right / bank.sampling_rate)

    assert resp.dtype == np.float64
    assert resp[left_idx] == pytest.approx(0.0, abs=1e-12)
    assert resp[mid_idx] == pytest.approx(1.0, rel=1e-12, abs=1e-12)
    assert resp[right_idx] == pytest.approx(0.0, abs=1e-12)
    assert np.all(resp[left_idx:mid_idx + 1] >= -1e-12)
    assert np.all(resp[mid_idx:right_idx + 1] >= -1e-12)


def test_triangular_half_spectrum_matches_full_nonanalytic():
    bank = TriangularOverlappingFilterBank("mel", num_filts=4, low_hz=80, high_hz=3000, sampling_rate=8000)
    width = 255
    full = bank.get_frequency_response(0, width, half=False)
    half = bank.get_frequency_response(0, width, half=True)
    expected_half_width = (width + 1) // 2
    assert len(half) == expected_half_width
    np.testing.assert_allclose(half, full[:expected_half_width], atol=0, rtol=0)


def test_triangular_truncated_response_reconstructs_full_half_spectrum():
    bank = TriangularOverlappingFilterBank("mel", num_filts=4, low_hz=80, high_hz=3000, sampling_rate=8000)
    width = 256
    bin_idx, trnc = bank.get_truncated_response(1, width)
    full = bank.get_frequency_response(1, width, half=False)
    half_width = width // 2 + 1
    half = _reconstruct_from_truncated_real(half_width, bin_idx, trnc)
    np.testing.assert_allclose(half, full[:half_width], atol=1e-12, rtol=1e-12)


def test_fbank_impulse_response_nonanalytic_is_real_and_matches_irfft():
    bank = Fbank(num_filts=4, low_hz=80, high_hz=3000, sampling_rate=8000, analytic=False)
    width = 128
    imp = bank.get_impulse_response(1, width)
    freq_half = bank.get_frequency_response(1, width, half=True)
    expected = np.fft.irfft(freq_half, n=width)
    assert np.isrealobj(imp)
    np.testing.assert_allclose(imp, expected, atol=1e-12, rtol=1e-12)


def test_fbank_analytic_impulse_response_matches_ifft():
    bank = Fbank(num_filts=4, low_hz=80, high_hz=3000, sampling_rate=8000, analytic=True)
    width = 128
    imp = bank.get_impulse_response(1, width)
    freq_full = bank.get_frequency_response(1, width, half=False)
    expected = np.fft.ifft(freq_full)
    assert np.iscomplexobj(imp)
    np.testing.assert_allclose(imp, expected, atol=1e-12, rtol=1e-12)


def test_fbank_truncated_response_reconstructs_half_spectrum():
    bank = Fbank(num_filts=4, low_hz=80, high_hz=3000, sampling_rate=8000, analytic=False)
    width = 256
    bin_idx, trnc = bank.get_truncated_response(2, width)
    full = bank.get_frequency_response(2, width, half=False)
    half_width = width // 2 + 1
    half = _reconstruct_from_truncated_real(half_width, bin_idx, trnc)
    np.testing.assert_allclose(half, full[:half_width], atol=1e-12, rtol=1e-12)


def test_gabor_bank_properties_and_shapes():
    bank = GaborFilterBank("mel", num_filts=4, low_hz=80, high_hz=3000, sampling_rate=8000)
    assert bank.is_real is False
    assert isinstance(bank.is_analytic, bool)
    assert bank.is_zero_phase is True
    assert bank.num_filts == 4
    assert bank.scaled_l2_norm is False
    assert bank.erb is False
    width = 128
    resp = bank.get_frequency_response(0, width, half=False)
    imp = bank.get_impulse_response(0, width)
    assert resp.dtype == np.complex128
    assert imp.dtype == np.complex128
    assert len(resp) == width
    assert len(imp) == width


def test_gabor_half_spectrum_length_and_nonnegativity():
    bank = GaborFilterBank("mel", num_filts=3, low_hz=80, high_hz=3000, sampling_rate=8000)
    width = 255
    half = bank.get_frequency_response(1, width, half=True)
    assert len(half) == (width + 1) // 2
    assert np.all(np.isfinite(half))
    assert np.all(np.real(half) >= 0)


def test_gabor_truncated_response_reconstructs_full_response_when_no_wrap():
    bank = GaborFilterBank("mel", num_filts=4, low_hz=80, high_hz=3000, sampling_rate=8000)
    width = 256
    bin_idx, trnc = bank.get_truncated_response(1, width)
    full = bank.get_frequency_response(1, width, half=False)
    recon = _reconstruct_from_truncated_complex(width, bin_idx, trnc)
    np.testing.assert_allclose(recon, full, atol=1e-10, rtol=1e-10)


def test_gabor_truncated_response_falls_back_to_full_when_support_wraps():
    # Low frequencies make wrap-around much more likely
    bank = GaborFilterBank("mel", num_filts=2, low_hz=1, high_hz=50, sampling_rate=8000)
    width = 64
    bin_idx, trnc = bank.get_truncated_response(0, width)
    full = bank.get_frequency_response(0, width, half=False)
    if bin_idx == 0:
        np.testing.assert_allclose(trnc, full, atol=1e-10, rtol=1e-10)


def test_complex_gammatone_bank_invalid_order_raises():
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", order=0)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", order=-1)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", order=1.5)


def test_complex_gammatone_bank_basic_properties():
    bank = ComplexGammatoneFilterBank("mel", num_filts=3, low_hz=80, high_hz=3000, sampling_rate=8000, order=4)
    assert bank.is_real is False
    assert isinstance(bank.is_analytic, bool)
    assert bank.is_zero_phase is False
    assert bank.order == 4
    assert bank.num_filts == 3
    assert len(bank.centers_hz) == 3
    assert len(bank.supports) == 3


def test_complex_gammatone_frequency_response_and_impulse_shapes():
    bank = ComplexGammatoneFilterBank("mel", num_filts=3, low_hz=80, high_hz=3000, sampling_rate=8000, order=4)
    width = 128
    resp = bank.get_frequency_response(1, width, half=False)
    imp = bank.get_impulse_response(1, width)
    assert resp.dtype == np.complex128
    assert imp.dtype == np.complex128
    assert len(resp) == width
    assert len(imp) == width
    assert np.all(np.isfinite(resp))
    assert np.all(np.isfinite(imp))


def test_complex_gammatone_truncated_response_reconstructs_full_when_possible():
    bank = ComplexGammatoneFilterBank("mel", num_filts=3, low_hz=80, high_hz=3000, sampling_rate=8000, order=4)
    width = 128
    bin_idx, trnc = bank.get_truncated_response(1, width)
    full = bank.get_frequency_response(1, width, half=False)
    recon = _reconstruct_from_truncated_complex(width, bin_idx, trnc)
    np.testing.assert_allclose(recon, full, atol=1e-10, rtol=1e-10)


def test_windows_are_symmetric_and_length_preserving():
    for win in [BartlettWindow(), BlackmanWindow(), HammingWindow(), HannWindow()]:
        arr = win.get_impulse_response(9)
        assert len(arr) == 9
        np.testing.assert_allclose(arr, arr[::-1], atol=1e-15, rtol=0)
        assert np.all(np.isfinite(arr))


def test_gamma_window_edge_cases_and_peak_behavior():
    win = GammaWindow(order=4, peak=0.75)
    empty = win.get_impulse_response(0)
    one = win.get_impulse_response(1)
    arr = win.get_impulse_response(16)

    assert empty.shape == (0,)
    np.testing.assert_allclose(one, np.array([1.0]))
    assert len(arr) == 16
    assert np.all(arr >= 0)
    assert np.argmax(arr) >= 8  # peak should be in the latter part of the window


def test_gamma_window_order_one_has_valid_shape():
    win = GammaWindow(order=1, peak=0.5)
    arr = win.get_impulse_response(10)
    assert len(arr) == 10
    assert np.all(arr >= 0)
    assert np.all(np.isfinite(arr))
