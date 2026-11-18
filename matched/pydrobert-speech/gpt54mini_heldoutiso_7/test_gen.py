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
    LinearFilterBank,
    TriangularOverlappingFilterBank,
    WindowFunction,
)
from pydrobert.speech.scales import MelScaling


def _reconstruct_from_truncated_real(bank, filt_idx, width):
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    full = np.zeros(width, dtype=trnc.dtype)
    full[bin_idx:bin_idx + len(trnc)] = trnc
    return full


def _reconstruct_from_truncated_complex(bank, filt_idx, width):
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    full = np.zeros(width, dtype=trnc.dtype)
    wrap = min(bin_idx + len(trnc), width) - bin_idx
    full[bin_idx:bin_idx + wrap] = trnc[:wrap]
    full[:len(trnc) - wrap] = trnc[wrap:]
    return full


def test_linear_filter_bank_cannot_be_instantiated():
    with pytest.raises(TypeError):
        LinearFilterBank()


def test_window_function_cannot_be_instantiated():
    with pytest.raises(TypeError):
        WindowFunction()


def test_triangular_bank_basic_properties_and_centers():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=5, low_hz=100, high_hz=2000, sampling_rate=8000)
    assert bank.num_filts == 5
    assert bank.sampling_rate == 8000
    assert bank.is_zero_phase is True
    assert bank.is_real is True
    assert bank.is_analytic is False
    assert len(bank.centers_hz) == 5
    assert len(bank.supports_hz) == 5
    assert len(bank.supports) == 5
    assert all(lo < hi for lo, hi in bank.supports_hz)


def test_triangular_bank_rejects_invalid_frequency_range():
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), low_hz=-1)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), low_hz=1000, high_hz=500, sampling_rate=8000)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), low_hz=100, high_hz=5000, sampling_rate=8000)


def test_triangular_frequency_response_half_and_full_match_symmetry():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=3, low_hz=50, high_hz=2000, sampling_rate=8000)
    width = 64
    full = bank.get_frequency_response(1, width, half=False)
    half = bank.get_frequency_response(1, width, half=True)
    assert full.dtype == np.float64
    assert half.dtype == np.float64
    assert len(full) == width
    assert len(half) == width // 2 + 1
    np.testing.assert_allclose(full[: len(half)], half)
    np.testing.assert_allclose(full[0], full[-0])
    np.testing.assert_allclose(full, full[::-1])


def test_triangular_truncated_response_reconstructs_full_spectrum():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=4, low_hz=80, high_hz=3000, sampling_rate=8000)
    width = 128
    full = bank.get_frequency_response(2, width, half=False)
    recon = _reconstruct_from_truncated_real(bank, 2, width)
    np.testing.assert_allclose(recon, full, atol=1e-12, rtol=1e-12)


def test_triangular_impulse_response_is_real_and_even():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=3, low_hz=100, high_hz=2000, sampling_rate=8000)
    width = 33
    imp = bank.get_impulse_response(0, width)
    assert np.isrealobj(imp)
    np.testing.assert_allclose(imp[1:], imp[:0:-1], atol=1e-12, rtol=1e-12)


def test_fbank_real_frequency_response_is_nonnegative_and_half_spectrum_matches():
    bank = Fbank(num_filts=4, low_hz=100, high_hz=3000, sampling_rate=8000, analytic=False)
    width = 128
    full = bank.get_frequency_response(1, width, half=False)
    half = bank.get_frequency_response(1, width, half=True)
    assert full.dtype == np.float64
    assert np.all(full >= 0)
    np.testing.assert_allclose(full[: len(half)], half)
    assert full.max() <= 1.0 + 1e-12


def test_fbank_truncated_response_reconstructs_full_spectrum():
    bank = Fbank(num_filts=4, low_hz=80, high_hz=3000, sampling_rate=8000, analytic=False)
    width = 128
    full = bank.get_frequency_response(2, width, half=False)
    recon = _reconstruct_from_truncated_real(bank, 2, width)
    np.testing.assert_allclose(recon, full, atol=1e-12, rtol=1e-12)


def test_fbank_analytic_impulse_response_is_complex_and_matches_ifft():
    bank = Fbank(num_filts=3, low_hz=100, high_hz=2500, sampling_rate=8000, analytic=True)
    width = 64
    imp = bank.get_impulse_response(0, width)
    freq = bank.get_frequency_response(0, width, half=False)
    expected = np.fft.ifft(freq)
    assert np.iscomplexobj(imp)
    np.testing.assert_allclose(imp, expected, atol=1e-12, rtol=1e-12)


def test_fbank_invalid_frequency_range_raises():
    with pytest.raises(ValueError):
        Fbank(low_hz=-1)
    with pytest.raises(ValueError):
        Fbank(low_hz=2000, high_hz=1000, sampling_rate=8000)
    with pytest.raises(ValueError):
        Fbank(low_hz=100, high_hz=5000, sampling_rate=8000)


def test_gabor_bank_basic_properties_and_centers():
    bank = GaborFilterBank(MelScaling(), num_filts=4, low_hz=50, high_hz=3000, sampling_rate=8000)
    assert bank.is_real is False
    assert bank.is_zero_phase is True
    assert bank.num_filts == 4
    assert len(bank.centers_hz) == 4
    assert len(bank.supports_hz) == 4
    assert len(bank.supports) == 4


def test_gabor_frequency_response_and_truncated_reconstruct():
    bank = GaborFilterBank(MelScaling(), num_filts=3, low_hz=80, high_hz=2500, sampling_rate=8000)
    width = 128
    full = bank.get_frequency_response(1, width, half=False)
    recon = _reconstruct_from_truncated_complex(bank, 1, width)
    np.testing.assert_allclose(recon, full, atol=1e-10, rtol=1e-10)
    assert np.iscomplexobj(full)


def test_gabor_impulse_response_is_complex_and_length_matches():
    bank = GaborFilterBank(MelScaling(), num_filts=2, low_hz=100, high_hz=2000, sampling_rate=8000)
    width = 37
    imp = bank.get_impulse_response(0, width)
    assert imp.shape == (width,)
    assert np.iscomplexobj(imp)


def test_gabor_truncated_returns_full_response_when_support_wraps():
    bank = GaborFilterBank(MelScaling(), num_filts=1, low_hz=1, high_hz=3900, sampling_rate=8000)
    width = 64
    bin_idx, trnc = bank.get_truncated_response(0, width)
    full = bank.get_frequency_response(0, width, half=False)
    if bin_idx == 0 and len(trnc) == len(full):
        np.testing.assert_allclose(trnc, full, atol=1e-10, rtol=1e-10)
    else:
        recon = _reconstruct_from_truncated_complex(bank, 0, width)
        np.testing.assert_allclose(recon, full, atol=1e-10, rtol=1e-10)


def test_complex_gammatone_invalid_order_raises():
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=0)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=1.5)


def test_complex_gammatone_basic_properties_and_frequency_response():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=3, low_hz=100, high_hz=3000, sampling_rate=8000, order=4)
    width = 128
    full = bank.get_frequency_response(1, width, half=False)
    half = bank.get_frequency_response(1, width, half=True)
    assert bank.is_real is False
    assert bank.is_zero_phase is False
    assert np.iscomplexobj(full)
    assert len(full) == width
    assert len(half) == width // 2 + 1
    np.testing.assert_allclose(full[: len(half)], half, atol=1e-10, rtol=1e-10)


def test_complex_gammatone_truncated_reconstructs_full_frequency_response():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=2, low_hz=200, high_hz=3000, sampling_rate=8000, order=4)
    width = 64
    full = bank.get_frequency_response(0, width, half=False)
    recon = _reconstruct_from_truncated_complex(bank, 0, width)
    np.testing.assert_allclose(recon, full, atol=1e-10, rtol=1e-10)


def test_complex_gammatone_impulse_response_has_expected_shape_and_complex_dtype():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=2, low_hz=200, high_hz=3000, sampling_rate=8000, order=4)
    width = 31
    imp = bank.get_impulse_response(0, width)
    assert imp.shape == (width,)
    assert np.iscomplexobj(imp)


def test_window_functions_return_expected_shapes_and_symmetry():
    for window in [BartlettWindow(), BlackmanWindow(), HammingWindow(), HannWindow()]:
        w5 = window.get_impulse_response(5)
        w1 = window.get_impulse_response(1)
        assert w5.shape == (5,)
        assert w1.shape == (1,)
        np.testing.assert_allclose(w5, w5[::-1], atol=1e-12, rtol=1e-12)


def test_gamma_window_edge_cases_and_peak_location():
    w0 = GammaWindow(order=4, peak=0.75).get_impulse_response(0)
    w1 = GammaWindow(order=4, peak=0.75).get_impulse_response(1)
    w = GammaWindow(order=4, peak=0.75).get_impulse_response(9)
    assert w0.size == 0
    np.testing.assert_array_equal(w1, np.array([1.0]))
    assert w.shape == (9,)
    assert np.all(w >= 0)
    assert np.argmax(w) <= 8
