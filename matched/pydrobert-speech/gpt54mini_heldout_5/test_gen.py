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
)
from pydrobert.speech.scales import MelScaling


def _reconstruct_full_from_truncated_real(width, bin_idx, trnc):
    full = np.zeros(width, dtype=trnc.dtype)
    full[bin_idx:bin_idx + len(trnc)] = trnc
    return full


def _reconstruct_full_from_truncated_complex(width, bin_idx, trnc):
    full = np.zeros(width, dtype=trnc.dtype)
    wrap = min(bin_idx + len(trnc), width) - bin_idx
    full[bin_idx:bin_idx + wrap] = trnc[:wrap]
    full[:len(trnc) - wrap] = trnc[wrap:]
    return full


def test_linear_filter_bank_is_abstract():
    with pytest.raises(TypeError):
        LinearFilterBank()


def test_triangular_bank_basic_properties():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=4, low_hz=20, high_hz=2000, sampling_rate=8000)
    assert bank.num_filts == 4
    assert bank.sampling_rate == 8000
    assert bank.is_real is True
    assert bank.is_analytic is False
    assert bank.is_zero_phase is True
    assert len(bank.centers_hz) == 4
    assert len(bank.supports_hz) == 4
    assert len(bank.supports) == 4
    assert all(lo < hi for lo, hi in bank.supports_hz)


def test_triangular_bank_invalid_frequency_range_raises():
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), low_hz=-1)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), low_hz=200, high_hz=100)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), low_hz=20, high_hz=5000, sampling_rate=8000)


def test_triangular_frequency_response_half_and_full_match_triangle_shape():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=3, low_hz=100, high_hz=2000, sampling_rate=8000)
    width = 64
    filt_idx = 1
    full = bank.get_frequency_response(filt_idx, width, half=False)
    half = bank.get_frequency_response(filt_idx, width, half=True)
    assert full.dtype == np.float64
    assert half.dtype == np.float64
    assert len(full) == width
    assert len(half) == width // 2 + 1
    assert np.all(full >= 0)
    assert np.all(half >= 0)
    # Full spectrum is symmetric for non-analytic triangular bank
    assert np.allclose(full[1:width // 2], full[:0:-1][:width // 2 - 1])


def test_triangular_truncated_response_reconstructs_frequency_response():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=3, low_hz=100, high_hz=2000, sampling_rate=8000)
    width = 80
    filt_idx = 0
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    full = bank.get_frequency_response(filt_idx, width, half=True)
    recon = np.zeros_like(full)
    recon[bin_idx:bin_idx + len(trnc)] = trnc
    assert np.allclose(recon, full)


def test_triangular_impulse_response_is_zero_phase_and_symmetric():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=2, low_hz=100, high_hz=2000, sampling_rate=8000)
    width = 31
    ir = bank.get_impulse_response(0, width)
    assert ir.dtype == np.float64
    assert np.allclose(ir, ir[::-1])
    assert np.isclose(np.imag(ir).sum(), 0.0)


def test_fbank_properties_and_frequency_response():
    bank = Fbank(num_filts=4, low_hz=100, high_hz=2000, sampling_rate=8000)
    assert bank.num_filts == 4
    assert bank.is_real is True
    assert bank.is_analytic is False
    assert bank.is_zero_phase is True
    width = 64
    fr = bank.get_frequency_response(2, width, half=True)
    assert fr.dtype == np.float64
    assert len(fr) == width // 2 + 1
    assert np.all(fr >= 0)
    assert np.max(fr) <= 1.0 + 1e-12


def test_fbank_truncated_response_reconstructs_half_spectrum():
    bank = Fbank(num_filts=4, low_hz=100, high_hz=2000, sampling_rate=8000)
    width = 63
    filt_idx = 1
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    half_width = (width + width % 2) // 2 + 1 - width % 2
    half = np.zeros(half_width, dtype=trnc.dtype)
    half[bin_idx:bin_idx + len(trnc)] = trnc
    assert np.allclose(half, bank.get_frequency_response(filt_idx, width, half=True))


def test_fbank_impulse_response_matches_irfft_of_half_spectrum():
    bank = Fbank(num_filts=3, low_hz=100, high_hz=1800, sampling_rate=8000)
    width = 64
    fr_half = bank.get_frequency_response(0, width, half=True)
    ir = bank.get_impulse_response(0, width)
    assert np.allclose(ir, np.fft.irfft(fr_half, n=width))


def test_gabor_bank_basic_properties_and_complex_response():
    bank = GaborFilterBank(MelScaling(), num_filts=4, low_hz=100, high_hz=2000, sampling_rate=8000)
    assert bank.is_real is False
    assert isinstance(bank.is_analytic, bool)
    assert bank.is_zero_phase is True
    width = 64
    fr = bank.get_frequency_response(0, width, half=False)
    assert fr.dtype == np.complex128
    assert len(fr) == width
    assert np.all(np.isfinite(fr))


def test_gabor_bank_truncated_response_matches_frequency_response():
    bank = GaborFilterBank(MelScaling(), num_filts=4, low_hz=100, high_hz=2000, sampling_rate=8000)
    width = 64
    filt_idx = 1
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    full = bank.get_frequency_response(filt_idx, width, half=False)
    recon = _reconstruct_full_from_truncated_complex(width, bin_idx, trnc)
    assert np.allclose(recon, full)


def test_gabor_bank_scaled_l2_norm_changes_frequency_response_amplitude():
    bank0 = GaborFilterBank(MelScaling(), num_filts=2, low_hz=100, high_hz=1000, sampling_rate=8000, scale_l2_norm=False)
    bank1 = GaborFilterBank(MelScaling(), num_filts=2, low_hz=100, high_hz=1000, sampling_rate=8000, scale_l2_norm=True)
    width = 64
    fr0 = bank0.get_frequency_response(0, width, half=False)
    fr1 = bank1.get_frequency_response(0, width, half=False)
    assert not np.allclose(fr0, fr1)
    assert np.max(np.abs(fr1)) != pytest.approx(np.max(np.abs(fr0)))


def test_gabor_bank_is_analytic_for_high_frequency_range():
    bank = GaborFilterBank(MelScaling(), num_filts=3, low_hz=1500, high_hz=3900, sampling_rate=8000)
    assert bank.is_analytic is True


def test_complex_gammatone_bank_basic_properties():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=4, low_hz=100, high_hz=2000, sampling_rate=8000)
    assert bank.is_real is False
    assert bank.is_zero_phase is False
    assert bank.order == 4
    assert bank.num_filts == 4
    assert len(bank.centers_hz) == 4


def test_complex_gammatone_bank_invalid_order_raises():
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=0)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=1.5)


def test_complex_gammatone_frequency_response_and_truncation_reconstruct():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=3, low_hz=100, high_hz=2000, sampling_rate=8000)
    width = 64
    filt_idx = 0
    full = bank.get_frequency_response(filt_idx, width, half=False)
    assert full.dtype == np.complex128
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    recon = _reconstruct_full_from_truncated_complex(width, bin_idx, trnc)
    assert np.allclose(recon, full)


def test_complex_gammatone_impulse_response_is_complex_and_finite():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=2, low_hz=100, high_hz=1000, sampling_rate=8000)
    ir = bank.get_impulse_response(0, 32)
    assert ir.dtype == np.complex128
    assert np.all(np.isfinite(ir))


def test_window_functions_even_and_unit_normalized_behavior():
    for window_cls in (BartlettWindow, BlackmanWindow, HammingWindow, HannWindow):
        window = window_cls().get_impulse_response(9)
        assert len(window) == 9
        assert window[0] == pytest.approx(0.0)
        assert window[-1] == pytest.approx(0.0)
        assert np.all(window >= -1e-12)


def test_gamma_window_edge_cases_and_peak_location():
    w0 = GammaWindow(order=4, peak=0.75).get_impulse_response(0)
    w1 = GammaWindow(order=4, peak=0.75).get_impulse_response(1)
    assert w0.shape == (0,)
    assert np.allclose(w1, np.array([1.0]))
    w = GammaWindow(order=4, peak=0.75).get_impulse_response(32)
    assert len(w) == 32
    assert np.all(w >= 0)
    assert np.argmax(w) >= 20


def test_supports_ms_converts_from_samples_to_milliseconds():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=2, low_hz=100, high_hz=1000, sampling_rate=8000)
    supports_ms = bank.supports_ms
    supports = bank.supports
    for (s0, s1), (m0, m1) in zip(supports, supports_ms):
        assert m0 == pytest.approx(s0 * 1000 / bank.sampling_rate)
        assert m1 == pytest.approx(s1 * 1000 / bank.sampling_rate)


def test_effective_support_threshold_available_and_positive():
    assert config.EFFECTIVE_SUPPORT_THRESHOLD > 0
