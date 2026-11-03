import numpy as np
import pytest

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
        full[width - bin_idx - len(trnc) + 1:width - bin_idx + 1] = trnc[:0:-1]
    return full


def _reconstruct_half_from_truncated(width, bin_idx, trnc):
    half_width = (width + width % 2) // 2 + 1 - width % 2
    half = np.zeros(half_width, dtype=trnc.dtype)
    half[bin_idx:bin_idx + len(trnc)] = trnc
    return half


def test_triangular_bank_basic_properties():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=4, low_hz=20, high_hz=4000, sampling_rate=16000)
    assert bank.num_filts == 4
    assert bank.sampling_rate == 16000
    assert bank.is_zero_phase is True
    assert bank.is_real is True
    assert bank.is_analytic is False
    assert len(bank.centers_hz) == 4
    assert len(bank.supports_hz) == 4
    assert len(bank.supports) == 4


def test_triangular_bank_invalid_frequency_range_raises():
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), low_hz=-1)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), low_hz=1000, high_hz=1000)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), low_hz=1000, high_hz=900)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), high_hz=10000, sampling_rate=16000)


def test_triangular_frequency_response_matches_truncated_reconstruction():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=6, low_hz=50, high_hz=6000, sampling_rate=16000)
    width = 64
    filt_idx = 2
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    full = bank.get_frequency_response(filt_idx, width, half=False)
    assert np.allclose(_reconstruct_full_from_truncated_real(width, bin_idx, trnc), full)
    half = bank.get_frequency_response(filt_idx, width, half=True)
    assert np.allclose(_reconstruct_half_from_truncated(width, bin_idx, trnc), half)


def test_triangular_frequency_response_is_symmetric_when_not_analytic():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=3, low_hz=100, high_hz=4000, sampling_rate=16000)
    resp = bank.get_frequency_response(1, 32, half=False)
    assert np.allclose(resp, resp[::-1])


def test_triangular_analytic_bank_suppresses_negative_frequencies():
    bank = TriangularOverlappingFilterBank(
        MelScaling(), num_filts=3, low_hz=100, high_hz=4000, sampling_rate=16000, analytic=True
    )
    resp = bank.get_frequency_response(1, 32, half=False)
    assert np.all(resp >= 0)
    assert np.allclose(resp[17:], 0.0)


def test_fbank_basic_properties_and_half_response_shape():
    bank = Fbank(num_filts=5, low_hz=50, high_hz=5000, sampling_rate=16000)
    assert bank.num_filts == 5
    assert bank.is_zero_phase is True
    assert bank.is_real is True
    assert bank.is_analytic is False
    resp = bank.get_frequency_response(2, 63, half=True)
    assert resp.shape == ((63 + 1) // 2,)
    assert np.all(resp >= 0)


def test_fbank_impulse_response_matches_irfft_for_non_analytic_case():
    bank = Fbank(num_filts=4, low_hz=50, high_hz=4000, sampling_rate=16000, analytic=False)
    width = 64
    filt_idx = 1
    freq = bank.get_frequency_response(filt_idx, width, half=True)
    expected = np.fft.irfft(freq, n=width)
    actual = bank.get_impulse_response(filt_idx, width)
    assert np.allclose(actual, expected)


def test_fbank_analytic_impulse_response_matches_ifft():
    bank = Fbank(num_filts=4, low_hz=50, high_hz=4000, sampling_rate=16000, analytic=True)
    width = 64
    filt_idx = 1
    freq = bank.get_frequency_response(filt_idx, width, half=False)
    expected = np.fft.ifft(freq)
    actual = bank.get_impulse_response(filt_idx, width)
    assert np.allclose(actual, expected)


def test_gabor_bank_properties_and_frequency_response_shapes():
    bank = GaborFilterBank(MelScaling(), num_filts=5, low_hz=50, high_hz=6000, sampling_rate=16000)
    assert bank.is_real is False
    assert bank.is_zero_phase is True
    assert bank.num_filts == 5
    assert bank.scaled_l2_norm is False
    assert bank.erb is False
    full = bank.get_frequency_response(2, 64, half=False)
    half = bank.get_frequency_response(2, 65, half=True)
    assert full.shape == (64,)
    assert half.shape == ((65 + 1) // 2,)
    assert np.all(full >= 0)
    assert np.all(half >= 0)


def test_gabor_frequency_response_and_truncated_response_agree():
    bank = GaborFilterBank(MelScaling(), num_filts=4, low_hz=50, high_hz=5000, sampling_rate=16000)
    width = 64
    filt_idx = 1
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    full = bank.get_frequency_response(filt_idx, width, half=False)
    recon = np.zeros(width, dtype=full.dtype)
    recon[bin_idx:bin_idx + len(trnc)] = trnc
    assert np.allclose(recon, full)


def test_gabor_impulse_response_is_inverse_fft_of_frequency_response():
    bank = GaborFilterBank(MelScaling(), num_filts=4, low_hz=50, high_hz=5000, sampling_rate=16000)
    width = 32
    filt_idx = 2
    expected = np.fft.ifft(bank.get_frequency_response(filt_idx, width, half=False))
    actual = bank.get_impulse_response(filt_idx, width)
    assert np.allclose(actual, expected)


def test_gabor_truncated_response_full_period_when_aliasing_large():
    bank = GaborFilterBank(MelScaling(), num_filts=1, low_hz=20, high_hz=8000, sampling_rate=16000)
    width = 16
    bin_idx, trnc = bank.get_truncated_response(0, width)
    if bank._wrap_supports_ang[0] >= 2 * np.pi:
        assert bin_idx == 0
        assert np.allclose(trnc, bank.get_frequency_response(0, width, half=False))


def test_complex_gammatone_bank_basic_properties_and_supports():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=4, low_hz=50, high_hz=5000, sampling_rate=16000)
    assert bank.is_real is False
    assert bank.is_zero_phase is False
    assert bank.num_filts == 4
    assert bank.order == 4
    assert len(bank.centers_hz) == 4
    assert len(bank.supports) == 4
    assert len(bank.supports_hz) == 4


def test_complex_gammatone_invalid_order_raises():
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=0)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=-1)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=2.5)


def test_complex_gammatone_frequency_response_shapes_and_truncation():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=3, low_hz=50, high_hz=6000, sampling_rate=16000)
    width = 32
    filt_idx = 1
    full = bank.get_frequency_response(filt_idx, width, half=False)
    half = bank.get_frequency_response(filt_idx, width, half=True)
    assert full.shape == (32,)
    assert half.shape == (32 // 2 + 1,)
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    recon = np.zeros(width, dtype=full.dtype)
    wrap = min(bin_idx + len(trnc), width) - bin_idx
    recon[bin_idx:bin_idx + wrap] = trnc[:wrap]
    recon[:len(trnc) - wrap] = trnc[wrap:]
    assert np.allclose(recon, full)


def test_complex_gammatone_impulse_response_has_complex_dtype():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=2, low_hz=50, high_hz=4000, sampling_rate=16000)
    resp = bank.get_impulse_response(0, 16)
    assert resp.dtype == np.complex128
    assert resp.shape == (16,)


@pytest.mark.parametrize(
    "cls",
    [BartlettWindow, BlackmanWindow, HammingWindow, HannWindow],
)
def test_window_functions_return_normalized_symmetric_windows(cls):
    win = cls().get_impulse_response(9)
    assert win.shape == (9,)
    assert np.allclose(win, win[::-1])
    assert np.isfinite(win).all()
    assert win[0] == pytest.approx(0.0)
    assert win[-1] == pytest.approx(0.0)


def test_gamma_window_edge_cases_and_peak_behavior():
    gw = GammaWindow(order=4, peak=0.75)
    assert gw.get_impulse_response(0).size == 0
    assert np.allclose(gw.get_impulse_response(1), np.array([1.0]))
    resp = gw.get_impulse_response(32)
    assert resp.shape == (32,)
    assert np.all(resp >= 0)
    assert np.argmax(resp) <= 31
    assert resp[-1] == pytest.approx(0.0)
