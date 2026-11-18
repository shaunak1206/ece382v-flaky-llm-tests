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
    LinearFilterBank,
    TriangularOverlappingFilterBank,
)
from pydrobert.speech.scales import MelScaling


def test_linear_filter_bank_is_abstract():
    with pytest.raises(TypeError):
        LinearFilterBank()


def test_triangular_bank_basic_properties_and_centers():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=5, low_hz=100, high_hz=3000)
    assert bank.num_filts == 5
    assert bank.sampling_rate == 16000
    assert bank.is_zero_phase is True
    assert bank.is_real is True
    assert bank.is_analytic is False
    assert len(bank.centers_hz) == 5
    assert len(bank.supports_hz) == 5
    assert len(bank.supports) == 5
    assert all(a < b for a, b in bank.supports_hz)
    assert all(a < b for a, b in bank.supports)
    assert all(bank.centers_hz[i] < bank.centers_hz[i + 1] for i in range(4))


def test_triangular_bank_invalid_frequency_ranges_raise():
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), low_hz=-1)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), low_hz=1000, high_hz=999)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), low_hz=100, high_hz=9000, sampling_rate=16000)


def test_triangular_frequency_response_half_and_full_shape_and_symmetry():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=3, low_hz=100, high_hz=3000)
    width = 128
    full = bank.get_frequency_response(1, width, half=False)
    half = bank.get_frequency_response(1, width, half=True)

    assert full.shape == (width,)
    assert half.shape == (width // 2 + 1,)
    assert full.dtype == np.float64
    assert half.dtype == np.float64
    np.testing.assert_allclose(full[0], half[0])
    np.testing.assert_allclose(full[width // 2], half[-1])

    # symmetry for the real, non-analytic case
    np.testing.assert_allclose(full[1:width // 2], full[:0:-1][: width // 2 - 1])


def test_triangular_truncated_response_matches_full_response():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=4, low_hz=100, high_hz=4000)
    width = 256
    filt_idx = 2
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    full = bank.get_frequency_response(filt_idx, width, half=False)

    assert isinstance(bin_idx, int)
    assert trnc.ndim == 1
    assert len(trnc) > 0
    expected = np.zeros(width, dtype=full.dtype)
    expected[bin_idx:bin_idx + len(trnc)] = trnc
    np.testing.assert_allclose(expected, full)


def test_triangular_analytic_bank_frequency_response_has_no_negative_frequency_mirroring():
    bank = TriangularOverlappingFilterBank(
        MelScaling(), num_filts=3, low_hz=100, high_hz=3000, analytic=True
    )
    width = 128
    full = bank.get_frequency_response(1, width, half=False)
    half = bank.get_frequency_response(1, width, half=True)

    assert bank.is_real is False
    assert bank.is_analytic is True
    assert full.shape == (width,)
    assert half.shape == (width // 2 + 1,)
    # values should be nonnegative and only occupy the positive-frequency side
    assert np.all(full >= 0)
    np.testing.assert_allclose(full[: width // 2 + 1], half)


def test_fbank_basic_properties_and_frequency_response():
    bank = Fbank(num_filts=5, low_hz=100, high_hz=3000)
    assert bank.num_filts == 5
    assert bank.is_zero_phase is True
    assert bank.is_real is True
    assert bank.is_analytic is False
    assert len(bank.centers_hz) == 5
    width = 128
    full = bank.get_frequency_response(2, width, half=False)
    half = bank.get_frequency_response(2, width, half=True)
    assert full.shape == (width,)
    assert half.shape == (width // 2 + 1,)
    assert np.all(full >= 0)
    np.testing.assert_allclose(full[: width // 2 + 1], half)


def test_fbank_get_impulse_response_matches_irfft_of_half_spectrum():
    bank = Fbank(num_filts=3, low_hz=100, high_hz=3000)
    width = 64
    freq_half = bank.get_frequency_response(1, width, half=True)
    imp = bank.get_impulse_response(1, width)
    expected = np.fft.irfft(freq_half, n=width)
    np.testing.assert_allclose(imp, expected)


def test_fbank_truncated_response_reconstructs_half_spectrum():
    bank = Fbank(num_filts=4, low_hz=100, high_hz=4000)
    width = 128
    filt_idx = 1
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    half_width = (width + width % 2) // 2 + 1 - width % 2
    half = np.zeros(half_width, dtype=np.float64)
    half[bin_idx:bin_idx + len(trnc)] = trnc
    np.testing.assert_allclose(half, bank.get_frequency_response(filt_idx, width, half=True))


def test_gabor_bank_properties_and_impulse_frequency_duality():
    bank = GaborFilterBank(MelScaling(), num_filts=4, low_hz=100, high_hz=3000)
    assert bank.is_real is False
    assert bank.is_zero_phase is True
    assert bank.num_filts == 4
    width = 64
    imp = bank.get_impulse_response(1, width)
    freq = bank.get_frequency_response(1, width, half=False)
    assert imp.shape == (width,)
    assert freq.shape == (width,)
    assert imp.dtype == np.complex128
    assert freq.dtype == np.complex128
    # DC response should be finite and nonnegative
    assert np.isfinite(freq[0].real)
    assert np.isfinite(freq[0].imag)
    np.testing.assert_allclose(freq.imag, 0, atol=1e-12)


def test_gabor_truncated_response_matches_frequency_response_or_wraps():
    bank = GaborFilterBank(MelScaling(), num_filts=4, low_hz=100, high_hz=3000)
    width = 128
    filt_idx = 2
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    full = bank.get_frequency_response(filt_idx, width, half=False)

    assert isinstance(bin_idx, int)
    assert trnc.ndim == 1
    expected = np.zeros(width, dtype=np.complex128)
    wrap = min(bin_idx + len(trnc), width) - bin_idx
    expected[bin_idx:bin_idx + wrap] = trnc[:wrap]
    expected[:len(trnc) - wrap] = trnc[wrap:]
    np.testing.assert_allclose(expected, full)


def test_gabor_bank_scale_l2_norm_changes_frequency_response():
    bank_a = GaborFilterBank(MelScaling(), num_filts=3, low_hz=100, high_hz=3000, scale_l2_norm=False)
    bank_b = GaborFilterBank(MelScaling(), num_filts=3, low_hz=100, high_hz=3000, scale_l2_norm=True)
    width = 128
    fa = bank_a.get_frequency_response(1, width)
    fb = bank_b.get_frequency_response(1, width)
    assert bank_a.scaled_l2_norm is False
    assert bank_b.scaled_l2_norm is True
    assert not np.allclose(fa, fb)


def test_complex_gammatone_invalid_order_raises():
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=0)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=-1)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=2.5)


def test_complex_gammatone_basic_properties_and_dimensions():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=3, low_hz=100, high_hz=3000, order=4)
    assert bank.is_real is False
    assert bank.is_zero_phase is False
    assert bank.order == 4
    assert bank.num_filts == 3
    width = 64
    imp = bank.get_impulse_response(1, width)
    freq = bank.get_frequency_response(1, width)
    assert imp.shape == (width,)
    assert freq.shape == (width,)
    assert imp.dtype == np.complex128
    assert freq.dtype == np.complex128
    assert np.isfinite(freq).all()


def test_complex_gammatone_truncated_response_matches_frequency_response():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=4, low_hz=100, high_hz=3000, order=4)
    width = 128
    filt_idx = 1
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    full = bank.get_frequency_response(filt_idx, width)

    assert isinstance(bin_idx, int)
    expected = np.zeros(width, dtype=np.complex128)
    wrap = min(bin_idx + len(trnc), width) - bin_idx
    expected[bin_idx:bin_idx + wrap] = trnc[:wrap]
    expected[:len(trnc) - wrap] = trnc[wrap:]
    np.testing.assert_allclose(expected, full)


def test_window_functions_basic_shapes_and_symmetry():
    widths = [1, 2, 7, 8]
    for win in [BartlettWindow(), BlackmanWindow(), HammingWindow(), HannWindow()]:
        for width in widths:
            arr = win.get_impulse_response(width)
            assert arr.shape == (width,)
            assert arr.dtype == np.float64
            assert np.all(np.isfinite(arr))
            np.testing.assert_allclose(arr, arr[::-1])


def test_gamma_window_special_cases_and_peak_behavior():
    w0 = GammaWindow(order=4, peak=0.75).get_impulse_response(0)
    w1 = GammaWindow(order=4, peak=0.75).get_impulse_response(1)
    assert w0.size == 0
    np.testing.assert_array_equal(w1, np.array([1.0]))

    w = GammaWindow(order=4, peak=0.75).get_impulse_response(32)
    assert w.shape == (32,)
    assert np.all(w >= 0)
    assert np.argmax(w) in {23, 24, 25}  # near the requested peak


def test_gamma_window_order_one_differs_from_higher_order():
    w1 = GammaWindow(order=1, peak=0.75).get_impulse_response(16)
    w4 = GammaWindow(order=4, peak=0.75).get_impulse_response(16)
    assert w1.shape == w4.shape == (16,)
    assert np.all(np.isfinite(w1))
    assert np.all(np.isfinite(w4))
    assert not np.allclose(w1, w4)
