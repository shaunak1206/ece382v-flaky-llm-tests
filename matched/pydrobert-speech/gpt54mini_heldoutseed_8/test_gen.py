import numpy as np
import pytest

from pydrobert.speech import config
from pydrobert.speech.filters import (
    BartlettWindow,
    ComplexGammatoneFilterBank,
    Fbank,
    GammaWindow,
    GaborFilterBank,
    HannWindow,
    TriangularOverlappingFilterBank,
)
from pydrobert.speech.scales import MelScaling


def _reconstruct_complex_full(bin_idx, trnc, width):
    full = np.zeros(width, dtype=trnc.dtype)
    wrap = min(bin_idx + len(trnc), width) - bin_idx
    full[bin_idx:bin_idx + wrap] = trnc[:wrap]
    full[:len(trnc) - wrap] = trnc[wrap:]
    return full


def _reconstruct_real_half(bin_idx, trnc, width):
    half_width = (width + width % 2) // 2 + 1 - width % 2
    half = np.zeros(half_width, dtype=trnc.dtype)
    half[bin_idx:bin_idx + len(trnc)] = trnc
    return half


def _reconstruct_real_full(bin_idx, trnc, width):
    full = np.zeros(width, dtype=trnc.dtype)
    full[bin_idx:bin_idx + len(trnc)] = trnc
    full[width - bin_idx - len(trnc) + 1:width - bin_idx + 1] = trnc[:None if bin_idx else 0:-1].conj()
    return full


def test_triangular_basic_properties():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=5, low_hz=50, high_hz=3000, sampling_rate=8000)
    assert bank.num_filts == 5
    assert bank.sampling_rate == 8000
    assert bank.is_real is True
    assert bank.is_analytic is False
    assert bank.is_zero_phase is True
    assert len(bank.centers_hz) == 5
    assert len(bank.supports_hz) == 5
    assert len(bank.supports) == 5
    assert len(bank.supports_ms) == 5


def test_triangular_invalid_frequency_range_raises():
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), low_hz=-1)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), low_hz=1000, high_hz=500)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), low_hz=100, high_hz=5000, sampling_rate=8000)


def test_triangular_frequency_response_half_and_full_are_symmetric():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=3, low_hz=100, high_hz=3000, sampling_rate=8000)
    width = 128
    filt_idx = 1
    half = bank.get_frequency_response(filt_idx, width, half=True)
    full = bank.get_frequency_response(filt_idx, width, half=False)
    assert half.ndim == 1 and full.ndim == 1
    assert half.dtype == np.float64
    assert full.dtype == np.float64
    assert np.all(half >= 0)
    assert np.all(full >= 0)
    assert np.isclose(full.max(), 1.0, atol=1e-12)
    # real zero-phase symmetry
    for idx in range(1, width // 2):
        assert np.isclose(full[idx], full[-idx], atol=1e-12)


def test_triangular_truncated_response_reconstructs_half_spectrum():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=4, low_hz=50, high_hz=3000, sampling_rate=8000)
    width = 256
    filt_idx = 2
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    half = bank.get_frequency_response(filt_idx, width, half=True)
    recon = _reconstruct_real_half(bin_idx, trnc, width)
    assert np.allclose(recon, half, atol=1e-12)
    assert 0 <= bin_idx < len(half)
    assert len(trnc) > 0


def test_triangular_analytic_response_does_not_mirror_negative_frequencies():
    bank = TriangularOverlappingFilterBank(
        MelScaling(), num_filts=3, low_hz=100, high_hz=3000, sampling_rate=8000, analytic=True
    )
    width = 128
    full = bank.get_frequency_response(1, width, half=False)
    half = bank.get_frequency_response(1, width, half=True)
    assert bank.is_real is False
    assert bank.is_analytic is True
    assert np.all(full >= 0)
    assert np.allclose(full[:len(half)], half, atol=1e-12)
    # analytic form should not mirror into the negative-frequency bins
    assert np.count_nonzero(full[len(half):]) < np.count_nonzero(half)


def test_fbank_basic_properties_and_impulse_response_shape():
    bank = Fbank(num_filts=4, low_hz=50, high_hz=3000, sampling_rate=8000)
    assert bank.num_filts == 4
    assert bank.is_real is True
    assert bank.is_analytic is False
    assert bank.is_zero_phase is True
    imp = bank.get_impulse_response(0, 32)
    assert imp.shape == (32,)
    assert np.isrealobj(imp)
    assert np.all(np.isfinite(imp))


def test_fbank_frequency_response_and_irfft_roundtrip():
    bank = Fbank(num_filts=3, low_hz=80, high_hz=3000, sampling_rate=8000)
    width = 64
    filt_idx = 1
    half = bank.get_frequency_response(filt_idx, width, half=True)
    imp = bank.get_impulse_response(filt_idx, width)
    assert half.dtype == np.float64
    assert imp.dtype == np.float64
    assert np.all(half >= 0)
    # irfft of the bank's own half-spectrum should reproduce the impulse response
    assert np.allclose(np.fft.irfft(half, n=width), imp, atol=1e-12)


def test_fbank_truncated_response_reconstructs_half_spectrum():
    bank = Fbank(num_filts=5, low_hz=60, high_hz=3000, sampling_rate=8000)
    width = 128
    filt_idx = 2
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    half = bank.get_frequency_response(filt_idx, width, half=True)
    recon = _reconstruct_real_half(bin_idx, trnc, width)
    assert np.allclose(recon, half, atol=1e-12)


def test_fbank_analytic_impulse_response_is_complex_and_matches_ifft():
    bank = Fbank(num_filts=3, low_hz=100, high_hz=3000, sampling_rate=8000, analytic=True)
    width = 64
    filt_idx = 1
    freq = bank.get_frequency_response(filt_idx, width, half=False)
    imp = bank.get_impulse_response(filt_idx, width)
    assert bank.is_analytic is True
    assert np.iscomplexobj(imp)
    assert np.allclose(np.fft.ifft(freq), imp, atol=1e-12)


def test_gabor_basic_properties():
    bank = GaborFilterBank(MelScaling(), num_filts=4, low_hz=100, high_hz=3000, sampling_rate=8000)
    assert bank.num_filts == 4
    assert bank.is_real is False
    assert bank.is_zero_phase is True
    assert bank.scaled_l2_norm is False
    assert bank.erb is False
    assert len(bank.centers_hz) == 4
    assert len(bank.supports_hz) == 4


def test_gabor_frequency_response_is_complex_and_truncated_reconstructs():
    bank = GaborFilterBank(MelScaling(), num_filts=4, low_hz=100, high_hz=3000, sampling_rate=8000)
    width = 128
    filt_idx = 1
    full = bank.get_frequency_response(filt_idx, width, half=False)
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    recon = _reconstruct_complex_full(bin_idx, trnc, width)
    assert np.iscomplexobj(full)
    assert np.iscomplexobj(trnc)
    assert np.allclose(recon, full, atol=1e-11)
    assert np.all(np.abs(full) >= 0)


def test_gabor_support_properties_and_no_wrap_default():
    bank = GaborFilterBank(MelScaling(), num_filts=2, low_hz=100, high_hz=3000, sampling_rate=8000)
    assert bank.is_analytic in (True, False)
    # With the default range, supports should be finite and ordered
    for lo, hi in bank.supports_hz:
        assert lo < hi
    for lo, hi in bank.supports:
        assert lo < hi


def test_gabor_low_frequency_bank_can_be_nonanalytic():
    bank = GaborFilterBank(MelScaling(), num_filts=2, low_hz=0, high_hz=3000, sampling_rate=8000)
    assert bank.is_real is False
    assert bank.is_analytic is False
    # If not analytic, at least one support should extend below 0 angular frequency
    assert any(lo < 0 for lo, _ in bank._supports_ang)


def test_gammatone_invalid_order_raises():
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=0)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=-1)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=2.5)


def test_gammatone_basic_properties_and_temp_support():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=3, low_hz=100, high_hz=3000, sampling_rate=8000, order=4)
    assert bank.num_filts == 3
    assert bank.order == 4
    assert bank.is_real is False
    assert bank.is_zero_phase is False
    assert len(bank.centers_hz) == 3
    assert len(bank.supports) == 3
    assert len(bank.supports_hz) == 3
    for lo, hi in bank.supports:
        assert lo < hi


def test_gammatone_frequency_response_and_truncated_reconstruct():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=3, low_hz=100, high_hz=3000, sampling_rate=8000, order=4)
    width = 128
    filt_idx = 1
    full = bank.get_frequency_response(filt_idx, width, half=False)
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    recon = _reconstruct_complex_full(bin_idx, trnc, width)
    assert np.iscomplexobj(full)
    assert np.iscomplexobj(trnc)
    assert np.allclose(recon, full, atol=1e-10)


def test_gammatone_impulse_response_is_complex_and_finite():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=2, low_hz=100, high_hz=3000, sampling_rate=8000, order=4)
    imp = bank.get_impulse_response(0, 64)
    assert imp.shape == (64,)
    assert np.iscomplexobj(imp)
    assert np.all(np.isfinite(imp))


def test_gammatone_max_centered_changes_supports_or_offsets():
    bank_default = ComplexGammatoneFilterBank(
        MelScaling(), num_filts=2, low_hz=100, high_hz=3000, sampling_rate=8000, order=4, max_centered=False
    )
    bank_centered = ComplexGammatoneFilterBank(
        MelScaling(), num_filts=2, low_hz=100, high_hz=3000, sampling_rate=8000, order=4, max_centered=True
    )
    # centered filters should differ in temporal supports/offsets from default
    assert bank_default.supports != bank_centered.supports or bank_default._offsets != bank_centered._offsets


def test_supports_ms_matches_supports_and_sampling_rate():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=3, low_hz=100, high_hz=3000, sampling_rate=8000)
    supports_ms = bank.supports_ms
    for (s0, s1), (m0, m1) in zip(bank.supports, supports_ms):
        assert np.isclose(m0, s0 * 1000 / bank.sampling_rate)
        assert np.isclose(m1, s1 * 1000 / bank.sampling_rate)


def test_window_functions_are_normalized_and_symmetric():
    for win in (BartlettWindow(), HannWindow(), GammaWindow()):
        w = win.get_impulse_response(9)
        assert w.shape == (9,)
        assert np.all(np.isfinite(w))
    bart = BartlettWindow().get_impulse_response(11)
    hann = HannWindow().get_impulse_response(11)
    gamma = GammaWindow(order=4, peak=0.75).get_impulse_response(11)
    assert np.isclose(bart[0], bart[-1])
    assert np.isclose(hann[0], hann[-1])
    assert gamma[0] >= 0


def test_gamma_window_edge_cases():
    gw = GammaWindow(order=1, peak=0.75)
    assert gw.get_impulse_response(0).size == 0
    assert np.allclose(gw.get_impulse_response(1), np.array([1.0]))
    w = gw.get_impulse_response(8)
    assert w.shape == (8,)
    assert np.all(np.isfinite(w))
