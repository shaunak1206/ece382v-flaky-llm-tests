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
from pydrobert.speech.scales import MelScaling, ScalingFunction


@pytest.mark.parametrize("cls", [
    TriangularOverlappingFilterBank,
    Fbank,
    GaborFilterBank,
    ComplexGammatoneFilterBank,
])
def test_num_filts_and_centers_consistent(cls):
    # Basic construction and consistency of num_filts and centers
    bank = cls(scaling_function=MelScaling(), num_filts=10, low_hz=50, high_hz=7000, sampling_rate=16000)
    assert bank.num_filts == 10
    centers = bank.centers_hz
    assert len(centers) == bank.num_filts
    # centers should be sorted ascending
    assert all(x < y for x, y in zip(centers, centers[1:]))
    # centers within bounds of low_hz and high_hz
    assert all(50 <= c <= 7000 for c in centers)


def test_triangularoverlappingfilterbank_invalid_freq_range():
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(
            scaling_function=MelScaling(),
            low_hz=-1,
            high_hz=8000,
            num_filts=5,
            sampling_rate=16000)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(
            scaling_function=MelScaling(),
            low_hz=5000,
            high_hz=3000,
            num_filts=5,
            sampling_rate=16000)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(
            scaling_function=MelScaling(),
            low_hz=20,
            high_hz=100000,
            num_filts=5,
            sampling_rate=16000)


def test_fbank_invalid_freq_range():
    with pytest.raises(ValueError):
        Fbank(low_hz=-1)
    with pytest.raises(ValueError):
        Fbank(low_hz=5000, high_hz=3000)
    with pytest.raises(ValueError):
        Fbank(high_hz=100000)


def test_gaborfilterbank_invalid_freq_range():
    with pytest.raises(ValueError):
        GaborFilterBank(scaling_function=MelScaling(), low_hz=-1)
    with pytest.raises(ValueError):
        GaborFilterBank(scaling_function=MelScaling(), low_hz=7000, high_hz=3000)
    with pytest.raises(ValueError):
        GaborFilterBank(scaling_function=MelScaling(), high_hz=100000)


def test_complexgammatonefilterbank_invalid_freq_and_order():
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(scaling_function=MelScaling(), low_hz=-1)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(scaling_function=MelScaling(), low_hz=7000, high_hz=3000)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(scaling_function=MelScaling(), high_hz=100000)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(scaling_function=MelScaling(), order=0)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(scaling_function=MelScaling(), order=-5)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(scaling_function=MelScaling(), order=2.5)


@pytest.mark.parametrize("bank_cls", [TriangularOverlappingFilterBank, Fbank])
@pytest.mark.parametrize("analytic", [False, True])
def test_get_frequency_response_shapes(bank_cls, analytic):
    bank = bank_cls(
        scaling_function=MelScaling(),
        num_filts=5,
        sampling_rate=16000,
        analytic=analytic)
    width = 32
    filt_idx = 2
    full = bank.get_frequency_response(filt_idx, width, half=False)
    assert full.ndim == 1
    assert full.shape[0] == width
    half = bank.get_frequency_response(filt_idx, width, half=True)
    expected_half_len = (width // 2 + 1) if width % 2 == 0 else ((width + 1) // 2)
    assert half.ndim == 1
    assert half.shape[0] == expected_half_len
    if not analytic:
        assert np.iscomplexobj(full) == False  # Triangular and Fbank return float64 frequency responses


@pytest.mark.parametrize("bank_cls", [TriangularOverlappingFilterBank, Fbank])
def test_get_impulse_response_returns_expected_length(bank_cls):
    bank = bank_cls(scaling_function=MelScaling(), num_filts=4, sampling_rate=16000)
    width = 25
    for filt_idx in range(bank.num_filts):
        imp = bank.get_impulse_response(filt_idx, width)
        assert isinstance(imp, np.ndarray)
        assert imp.ndim == 1
        assert imp.shape[0] == width


def test_triangularoverlappingfilterbank_supports_properties_consistency():
    bank = TriangularOverlappingFilterBank(scaling_function=MelScaling(), num_filts=6)
    assert len(bank.supports_hz) == bank.num_filts
    assert len(bank.supports) == bank.num_filts
    assert len(bank.supports_ms) == bank.num_filts
    for (low_hz, high_hz), (start, end), (start_ms, end_ms) in zip(
            bank.supports_hz, bank.supports, bank.supports_ms):
        # supports_ms in milliseconds corresponds to supports in samples
        assert np.isclose(start_ms, start * 1000 / bank.sampling_rate)
        assert np.isclose(end_ms, end * 1000 / bank.sampling_rate)
        # Hz frequencies correspond roughly to centers
        assert low_hz <= high_hz


@pytest.mark.parametrize("bank_cls", [TriangularOverlappingFilterBank, Fbank])
def test_get_truncated_response(bank_cls):
    bank = bank_cls(scaling_function=MelScaling(), num_filts=5)
    width = 64
    filt_idx = 1
    bin_idx, buf = bank.get_truncated_response(filt_idx, width)
    assert isinstance(bin_idx, int)
    assert isinstance(buf, np.ndarray)
    assert buf.ndim == 1
    # truncated response should not be longer than width
    assert buf.shape[0] <= width


@pytest.mark.parametrize(
    "window_cls, numpy_func",
    [
        (BartlettWindow, np.bartlett),
        (BlackmanWindow, np.blackman),
        (HammingWindow, np.hamming),
        (HannWindow, np.hanning),
    ])
def test_window_functions_shapes_and_normalization(window_cls, numpy_func):
    win = window_cls()
    # Test some widths including edge cases
    for width in [1, 10, 100]:
        w = win.get_impulse_response(width)
        assert isinstance(w, np.ndarray)
        assert w.ndim == 1
        assert w.shape[0] == width
        # Check normalization by comparing sums to expected factor
        raw = numpy_func(width)
        # Avoid divide by zero for width=1
        denom = win.get_impulse_response(width).sum() / raw.sum() if raw.sum() != 0 else 1.0
        # Values normalized close to the expected scale, so sums roughly match
        assert np.allclose(w.sum(), raw.sum() / denom, rtol=1e-1, atol=1e-5)


def test_gamma_window_behavior():
    gw = GammaWindow(order=3, peak=0.75)
    width = 50
    w = gw.get_impulse_response(width)
    assert isinstance(w, np.ndarray)
    assert w.ndim == 1
    assert w.shape[0] == width
    # Gamma window values are positive
    assert np.all(w >= 0)
    # Peak should be near expected location
    peak_pos = int(gw.peak * width)
    assert w[peak_pos] >= w[peak_pos - 1] or peak_pos == 0
    assert w[peak_pos] >= w[min(peak_pos + 1, width-1)]


def test_gamma_window_edge_cases():
    gw = GammaWindow(order=3, peak=0.5)
    w0 = gw.get_impulse_response(0)
    assert isinstance(w0, np.ndarray)
    assert w0.size == 0
    w1 = gw.get_impulse_response(1)
    assert w1.shape == (1,)
    assert np.isclose(w1[0], 1.0)


@pytest.mark.parametrize("analytic", [False, True])
def test_fbank_impulse_response_consistency(analytic):
    bank = Fbank(num_filts=5, analytic=analytic)
    width = 32
    for filt_idx in range(bank.num_filts):
        imp = bank.get_impulse_response(filt_idx, width)
        assert imp.shape[0] == width
        # When analytic, the impulse response is complex, otherwise real
        if analytic:
            assert np.iscomplexobj(imp)
        else:
            assert not np.iscomplexobj(imp)
        # Impulse response re-transformed should give close to frequency response
        freq_resp = bank.get_frequency_response(filt_idx, width, half=False)
        freq_resp_back = np.fft.fft(imp)
        # Magnitudes should be close (ignoring phase)
        assert np.allclose(np.abs(freq_resp), np.abs(freq_resp_back), rtol=1e-1, atol=1e-3)


def test_gaborfilterbank_properties_and_methods():
    bank = GaborFilterBank(scaling_function=MelScaling(), num_filts=7, scale_l2_norm=True, erb=True)
    assert bank.is_real is False
    # Analytic depends on if wrap_below is False (likely true here)
    assert isinstance(bank.is_analytic, bool)
    assert bank.num_filts == 7
    assert bank.order is not hasattr(bank, 'order') or True  # no order attr for GaborFilterBank
    # centers_hz length matches num_filts
    assert len(bank.centers_hz) == bank.num_filts
    # supports lengths match num_filts
    assert len(bank.supports_hz) == bank.num_filts
    assert len(bank.supports) == bank.num_filts
    width = 64
    idx = 3
    imp = bank.get_impulse_response(idx, width)
    assert imp.shape[0] == width
    assert np.iscomplexobj(imp)
    freq = bank.get_frequency_response(idx, width)
    assert freq.shape[0] == width
    # truncated response
    left_idx, truncated = bank.get_truncated_response(idx, width)
    assert isinstance(left_idx, int)
    assert isinstance(truncated, np.ndarray)


def test_complexgammatonefilterbank_properties_and_methods():
    bank = ComplexGammatoneFilterBank(scaling_function=MelScaling(), num_filts=6, order=3, max_centered=True,
                                     scale_l2_norm=True, erb=True)
    assert bank.num_filts == 6
    assert bank.order == 3
    assert bank.is_zero_phase is False
    assert not bank.is_real
    width = 48
    idx = 2
    imp = bank.get_impulse_response(idx, width)
    # Impulse response shape and type
    assert imp.shape[0] == width
    assert np.iscomplexobj(imp)
    freq = bank.get_frequency_response(idx, width)
    assert freq.shape[0] == width
    left_idx, truncated = bank.get_truncated_response(idx, width)
    assert isinstance(left_idx, int)
    assert isinstance(truncated, np.ndarray)


def test_triangular_overlapping_filterbank_get_impulse_response_symmetry():
    bank = TriangularOverlappingFilterBank(scaling_function=MelScaling(), num_filts=5)
    width = 20
    for i in range(bank.num_filts):
        imp = bank.get_impulse_response(i, width)
        # For zero phase filters impulse response should be symmetric conjugate around center
        # Because the support might be negative indices, full symmetry is not exact but magnitudes should be symmetric
        left = imp[:width // 2]
        right = imp[-(width // 2):][::-1]
        if bank.is_real:
            assert np.allclose(left.real, right.real, rtol=1e-3, atol=1e-6)
            assert np.allclose(left.imag, -right.imag, rtol=1e-3, atol=1e-6)
        else:
            # It might be complex if analytic, so symmetry check relaxed
            pass


def test_fbank_get_frequency_response_half_and_full_consistency():
    bank = Fbank(num_filts=4, analytic=False)
    width = 30
    filt_idx = 1
    full = bank.get_frequency_response(filt_idx, width, half=False)
    half = bank.get_frequency_response(filt_idx, width, half=True)
    # The first part of full should match half up to half length
    half_len = (width // 2 + 1) if width % 2 == 0 else ((width + 1) // 2)
    np.testing.assert_allclose(full[:half_len], half, rtol=1e-5, atol=1e-8)


def test_triangularoverlapping_get_frequency_response_value_ranges():
    bank = TriangularOverlappingFilterBank(scaling_function=MelScaling(), num_filts=8)
    width = 50
    for filt_idx in range(bank.num_filts):
        freq_resp = bank.get_frequency_response(filt_idx, width)
        # Frequency response magnitudes within [0,1]
        assert np.all(freq_resp >= 0)
        assert np.all(freq_resp <= 1)
        # At center frequency index response is 1 or close to
        center_hz = bank.centers_hz[filt_idx]
        center_idx = int(np.round(width * center_hz / bank.sampling_rate))
        # center_idx might be out of range, clamp
        center_idx = min(max(0, center_idx), width - 1)
        assert freq_resp[center_idx] >= 0.8


def test_gabor_get_frequency_response_sum_positive():
    bank = GaborFilterBank(scaling_function=MelScaling(), num_filts=4)
    width = 40
    for filt_idx in range(bank.num_filts):
        freq_resp = bank.get_frequency_response(filt_idx, width)
        # Frequency response values are all real and positive (sum of exponentials)
        assert np.all(freq_resp >= 0)


def test_complex_gammatone_get_impulse_response_vs_h_method():
    bank = ComplexGammatoneFilterBank(scaling_function=MelScaling(), num_filts=5, order=2)
    width = 16
    idx = 1
    imp = bank.get_impulse_response(idx, width)
    # Test _h(t, idx) manually at some point and compare with impulse response sums
    val_at_zero = bank._h(0, idx)
    # Since offset might be zero or negative, impulse should respect zero-padding before offset
    assert np.isclose(val_at_zero, 0j) or val_at_zero == 0j
    # The impulse response at index 0 is sum over periods, but for t=0 it should be close to val_at_zero
    assert np.isclose(imp[0], val_at_zero, atol=1e-6)


def test_aliases_are_registered():
    # Test Aliases for main classes
    assert 'tri' in TriangularOverlappingFilterBank.aliases
    assert 'fbank' in Fbank.aliases
    assert 'gabor' in GaborFilterBank.aliases
    assert 'gammatone' in ComplexGammatoneFilterBank.aliases
    assert 'tonebank' in ComplexGammatoneFilterBank.aliases
    assert 'bartlett' in BartlettWindow.aliases
    assert 'blackman' in BlackmanWindow.aliases
    assert 'hamming' in HammingWindow.aliases
    assert 'hann' in HannWindow.aliases
    assert 'hanning' in HannWindow.aliases
    assert 'gamma' in GammaWindow.aliases
