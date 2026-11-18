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
from pydrobert.speech.scales import MelScaling


def _reconstruct_full_from_truncated_real(width, bin_idx, trnc):
    full = np.zeros(width, dtype=trnc.dtype)
    full[bin_idx:bin_idx + len(trnc)] = trnc
    if bin_idx:
        full[width - bin_idx - len(trnc) + 1:width - bin_idx + 1] = trnc[:None if bin_idx else 0:-1].conj()
    return full


def _reconstruct_full_from_truncated_complex(width, bin_idx, trnc):
    full = np.zeros(width, dtype=trnc.dtype)
    wrap = min(bin_idx + len(trnc), width) - bin_idx
    full[bin_idx:bin_idx + wrap] = trnc[:wrap]
    full[:len(trnc) - wrap] = trnc[wrap:]
    return full


def test_triangular_invalid_frequency_range_raises():
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), low_hz=-1.0)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), low_hz=1000.0, high_hz=1000.0)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), low_hz=1000.0, high_hz=800.0)


def test_triangular_basic_properties_and_supports_ms():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=4, low_hz=20.0, high_hz=4000.0, sampling_rate=16000)
    assert bank.is_real is True
    assert bank.is_analytic is False
    assert bank.is_zero_phase is True
    assert bank.num_filts == 4
    assert bank.sampling_rate == 16000
    assert len(bank.centers_hz) == 4
    assert len(bank.supports_hz) == 4
    assert len(bank.supports) == 4
    assert len(bank.supports_ms) == 4
    for (s0, s1), (ms0, ms1) in zip(bank.supports, bank.supports_ms):
        assert ms0 == pytest.approx(s0 * 1000 / bank.sampling_rate)
        assert ms1 == pytest.approx(s1 * 1000 / bank.sampling_rate)


def test_triangular_frequency_response_half_matches_full_half_spectrum():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=3, low_hz=100.0, high_hz=3000.0, sampling_rate=8000)
    width = 32
    filt_idx = 1
    full = bank.get_frequency_response(filt_idx, width, half=False)
    half = bank.get_frequency_response(filt_idx, width, half=True)
    assert half.shape == (width // 2 + 1,)
    assert np.all(half >= 0)
    np.testing.assert_allclose(half, full[:width // 2 + 1])
    np.testing.assert_allclose(full, full[::-1])


def test_triangular_truncated_response_reconstructs_full_frequency_response():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=4, low_hz=50.0, high_hz=3500.0, sampling_rate=8000)
    width = 64
    filt_idx = 2
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    full = bank.get_frequency_response(filt_idx, width, half=False)
    recon = np.zeros(width, dtype=full.dtype)
    recon[bin_idx:bin_idx + len(trnc)] = trnc
    np.testing.assert_allclose(recon[: len(full)], full)


def test_triangular_impulse_response_is_real_and_symmetric():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=2, low_hz=200.0, high_hz=3000.0, sampling_rate=8000)
    width = 33
    h = bank.get_impulse_response(0, width)
    assert h.dtype == np.float64
    np.testing.assert_allclose(h, h[::-1])
    assert np.isclose(h.imag if np.iscomplexobj(h) else 0.0, 0.0)


def test_analytic_triangular_frequency_response_is_one_sided():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=3, low_hz=100.0, high_hz=3000.0, sampling_rate=8000, analytic=True)
    width = 64
    full = bank.get_frequency_response(1, width, half=False)
    assert bank.is_real is False
    assert bank.is_analytic is True
    assert np.all(full[width // 2 + 1 :] == 0)
    assert np.all(full >= 0)


def test_fbank_impulse_response_matches_inverse_transform_real_case():
    bank = Fbank(num_filts=3, low_hz=100.0, high_hz=3000.0, sampling_rate=8000, analytic=False)
    width = 48
    filt_idx = 1
    h = bank.get_impulse_response(filt_idx, width)
    fr = bank.get_frequency_response(filt_idx, width, half=True)
    expected = np.fft.irfft(fr, n=width)
    np.testing.assert_allclose(h, expected)


def test_fbank_truncated_response_reconstructs_half_spectrum():
    bank = Fbank(num_filts=4, low_hz=50.0, high_hz=3400.0, sampling_rate=8000, analytic=False)
    width = 64
    filt_idx = 1
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    half_width = (width + width % 2) // 2 + 1 - width % 2
    half = np.zeros(half_width, dtype=trnc.dtype)
    half[bin_idx:bin_idx + len(trnc)] = trnc
    np.testing.assert_allclose(half, bank.get_frequency_response(filt_idx, width, half=True))


def test_gabor_basic_properties_and_frequency_response_shape():
    bank = GaborFilterBank(MelScaling(), num_filts=3, low_hz=100.0, high_hz=3000.0, sampling_rate=8000)
    width = 32
    fr = bank.get_frequency_response(0, width, half=False)
    assert bank.is_real is False
    assert bank.is_zero_phase is True
    assert fr.dtype == np.complex128
    assert fr.shape == (width,)
    assert np.all(np.isfinite(fr))
    assert np.all(np.real(fr) >= 0)


def test_gabor_truncated_response_reconstructs_full_frequency_response():
    bank = GaborFilterBank(MelScaling(), num_filts=3, low_hz=100.0, high_hz=3000.0, sampling_rate=8000)
    width = 64
    filt_idx = 1
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    full = bank.get_frequency_response(filt_idx, width, half=False)
    recon = _reconstruct_full_from_truncated_complex(width, bin_idx, trnc)
    np.testing.assert_allclose(recon, full)


def test_gabor_impulse_response_is_conjugate_symmetric():
    bank = GaborFilterBank(MelScaling(), num_filts=2, low_hz=100.0, high_hz=2500.0, sampling_rate=8000)
    width = 21
    h = bank.get_impulse_response(0, width)
    assert h.dtype == np.complex128
    np.testing.assert_allclose(h[1:], h[:0:-1].conj())


def test_gammatone_invalid_order_raises():
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=0)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=-1)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=1.5)


def test_gammatone_basic_properties_and_truncated_response_reconstructs():
    bank = ComplexGammatoneFilterBank(
        MelScaling(), num_filts=3, low_hz=100.0, high_hz=3000.0, sampling_rate=8000, order=4
    )
    width = 64
    filt_idx = 1
    assert bank.is_real is False
    assert bank.is_zero_phase is False
    assert bank.order == 4
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    full = bank.get_frequency_response(filt_idx, width, half=False)
    recon = _reconstruct_full_from_truncated_complex(width, bin_idx, trnc)
    np.testing.assert_allclose(recon, full)


def test_window_functions_are_deterministic_and_have_expected_endpoints():
    widths = [1, 2, 7]
    windows = [
        BartlettWindow(),
        BlackmanWindow(),
        HammingWindow(),
        HannWindow(),
        GammaWindow(order=4, peak=0.75),
    ]
    for win in windows:
        for width in widths:
            a = win.get_impulse_response(width)
            b = win.get_impulse_response(width)
            np.testing.assert_allclose(a, b)
            assert a.shape == (width,)
    assert np.array_equal(GammaWindow(order=4, peak=0.75).get_impulse_response(0), np.array([], dtype=float))
    assert np.array_equal(GammaWindow(order=4, peak=0.75).get_impulse_response(1), np.array([1.0]))
