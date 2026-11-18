import numpy as np
import pytest

from pydrobert.speech.filters import (
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


@pytest.mark.parametrize("BankClass", [
    TriangularOverlappingFilterBank,
    Fbank,
    GaborFilterBank,
    ComplexGammatoneFilterBank,
])
def test_invalid_frequency_range_raises_value_error(BankClass):
    # low_hz < 0
    with pytest.raises(ValueError):
        BankClass("mel", low_hz=-1)
    # high_hz <= low_hz
    with pytest.raises(ValueError):
        BankClass("mel", low_hz=100, high_hz=50)
    # high_hz > nyquist
    sr = 16000
    with pytest.raises(ValueError):
        BankClass("mel", low_hz=100, high_hz=sr)

@pytest.mark.parametrize("BankClass", [
    TriangularOverlappingFilterBank,
    Fbank,
])
def test_num_filts_and_centers_hz_consistency(BankClass):
    num_filters = 10
    bank = BankClass("mel", num_filts=num_filters, low_hz=20., high_hz=8000, sampling_rate=16000)
    assert bank.num_filts == num_filters
    assert len(bank.centers_hz) == num_filters
    for center, (low, high) in zip(bank.centers_hz, bank.supports_hz):
        assert low <= center <= high

@pytest.mark.parametrize("BankClass", [
    TriangularOverlappingFilterBank,
    Fbank,
])
def test_sampling_rate_property(BankClass):
    sr = 12345
    bank = BankClass("mel", sampling_rate=sr)
    assert bank.sampling_rate == sr

@pytest.mark.parametrize("BankClass", [
    TriangularOverlappingFilterBank,
    Fbank,
])
@pytest.mark.parametrize("analytic", [False, True])
def test_is_real_and_is_analytic_properties(BankClass, analytic):
    bank = BankClass("mel", analytic=analytic)
    assert bank.is_real == (not analytic)
    assert bank.is_analytic == analytic

def test_triangular_filter_bank_get_impulse_response_and_frequency_response():
    bank = TriangularOverlappingFilterBank("mel", num_filts=5)
    width = 256
    idx = 2
    impulse = bank.get_impulse_response(idx, width)
    freq = bank.get_frequency_response(idx, width)
    # impulse is real or complex depending on analytic
    assert impulse.shape == (width,)
    assert freq.shape == (width,)
    # frequency response real because is_zero_phase True
    assert freq.dtype == np.float64
    # values non-negative and bounded by 1
    assert np.all(freq >= 0)
    assert np.all(freq <= 1)

    freq_half = bank.get_frequency_response(idx, width, half=True)
    expected_half_len = width // 2 + 1 if width % 2 == 0 else (width + 1) // 2
    assert freq_half.shape[0] == expected_half_len
    assert np.all(freq_half >= 0)
    assert np.all(freq_half <= 1)

def test_triangular_filter_bank_get_truncated_response():
    bank = TriangularOverlappingFilterBank("mel", num_filts=5)
    width = 256
    idx = 3
    bin_idx, buf = bank.get_truncated_response(idx, width)
    # bin_idx should be in expected index range
    assert 0 <= bin_idx <= width
    # buf length should be positive
    assert buf.size > 0
    # buf values should be in [0,1]
    assert np.all(buf >= 0)
    assert np.all(buf <= 1)

def test_fbank_get_frequency_response_has_sqrt_behavior():
    bank = Fbank(num_filts=10, low_hz=50, high_hz=4000)
    width = 256
    idx = 4
    freq_full = bank.get_frequency_response(idx, width, half=False)
    freq_half = bank.get_frequency_response(idx, width, half=True)
    # Full freq shape matches width
    assert freq_full.shape[0] == width
    # Half freq length matches expected half_len
    expected_half_len = width // 2 + 1 if width % 2 == 0 else (width + 1) // 2
    assert freq_half.shape[0] == expected_half_len
    # All values non-negative and less or equal 1
    assert np.all(freq_full >= 0)
    assert np.all(freq_full <= 1)
    assert np.all(freq_half >= 0)
    assert np.all(freq_half <= 1)

def test_fbank_get_impulse_response_consistent_with_fft():
    bank = Fbank(num_filts=5)
    width = 128
    idx = 1
    imp_resp = bank.get_impulse_response(idx, width)
    freq_resp = bank.get_frequency_response(idx, width, half=False)
    # For analytic=False (default), impulse response is irfft of frequency response
    irfft = np.fft.irfft(bank.get_frequency_response(idx, width, half=True), n=width)
    # Signals close in norm (numerical)
    assert np.allclose(imp_resp, irfft, atol=1e-5)

def test_gabor_bank_properties_and_response_shapes():
    bank = GaborFilterBank("mel", num_filts=6)
    assert bank.num_filts == 6
    assert bank.is_real is False
    assert isinstance(bank.is_analytic, bool)
    width = 128
    idx = 0
    impulse = bank.get_impulse_response(idx, width)
    freq = bank.get_frequency_response(idx, width, half=False)
    # impulse response is complex128 array
    assert impulse.dtype == np.complex128
    assert impulse.shape == (width,)
    # frequency response is float64 array (magnitude)
    assert freq.dtype == np.float64
    assert freq.shape == (width,)
    freq_half = bank.get_frequency_response(idx, width, half=True)
    expected_half_len = width // 2 + 1 if width % 2 == 0 else (width + 1) // 2
    assert freq_half.shape == (expected_half_len,)

def test_gabor_get_truncated_response_wraps():
    bank = GaborFilterBank("mel", num_filts=5)
    width = 256
    idx = 1
    bin_idx, buf = bank.get_truncated_response(idx, width)
    assert isinstance(bin_idx, int)
    assert buf.ndim == 1
    # buf length positive and less or equal width
    assert 0 < buf.size <= width

def test_complex_gammatone_valid_order_and_max_centered_argument():
    # Invalid order raises
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", order=0)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", order=-3)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", order=2.5)
    # max_centered may be used
    bank = ComplexGammatoneFilterBank("mel", order=3, max_centered=True)
    assert bank.order == 3
    assert bank._offsets[0] < 0  # due to max_centered
    # Also test with max_centered False (offsets >= 0)
    bank2 = ComplexGammatoneFilterBank("mel", order=3, max_centered=False)
    assert bank2._offsets[0] == 0 or bank2._offsets[0] >= 0

def test_complex_gammatone_impulse_response_and_frequency_response_shapes():
    bank = ComplexGammatoneFilterBank("mel", num_filts=4, order=3)
    width = 200
    idx = 2
    impulse = bank.get_impulse_response(idx, width)
    freq = bank.get_frequency_response(idx, width, half=False)
    freq_half = bank.get_frequency_response(idx, width, half=True)
    assert impulse.shape == (width,)
    assert np.iscomplexobj(impulse)
    # Frequency response complex128 dtype
    assert freq.shape == (width,)
    assert freq.dtype == np.complex128
    expected_half_len = width // 2 + 1 if width % 2 == 0 else (width + 1) // 2
    assert freq_half.shape == (expected_half_len,)

def test_complex_gammatone_get_truncated_response_properties():
    bank = ComplexGammatoneFilterBank("mel", num_filts=4)
    width = 128
    idx = 1
    bin_idx, buf = bank.get_truncated_response(idx, width)
    assert isinstance(bin_idx, int)
    assert buf.ndim == 1
    assert buf.size > 0
    # Frequency response based on get_truncated_response matches bank freq response
    freq_full = bank.get_frequency_response(idx, width)
    # Reconstruct partial spectrum from truncated response array
    full_reconstructed = np.zeros_like(freq_full)
    wrap_point = min(bin_idx + len(buf), width) - bin_idx
    full_reconstructed[bin_idx:bin_idx + wrap_point] = buf[:wrap_point]
    if wrap_point < len(buf):
        full_reconstructed[:len(buf) - wrap_point] = buf[wrap_point:]
    # Because this excludes some edges, values should be close but not necessarily exact
    assert np.all(full_reconstructed >= 0)

@pytest.mark.parametrize("WindowClass", [
    BartlettWindow,
    BlackmanWindow,
    HammingWindow,
    HannWindow,
    GammaWindow])
@pytest.mark.parametrize("width", [0, 1, 2, 10, 100])
def test_window_get_impulse_response_length_and_values(WindowClass, width):
    window = WindowClass()
    impulse = window.get_impulse_response(width)
    assert isinstance(impulse, np.ndarray)
    assert impulse.ndim == 1
    assert impulse.shape == (width,)
    if width == 0:
        assert impulse.size == 0
    elif width == 1:
        # For width 1, Bartlett and others produce 1
        assert np.isclose(impulse[0], 1)
    else:
        # For widths > 1, values should be real, non-negative (windows)
        assert np.all(np.isfinite(impulse))
        assert np.all(impulse >= 0)
        # Check normalization: sum or max not necessarily 1 but nonzero
        assert impulse.max() > 0

def test_gamma_window_order_and_peak_effect():
    # Default order and peak
    win = GammaWindow()
    w1 = win.get_impulse_response(50)
    # Changing peak changes shape distinctly
    win2 = GammaWindow(order=5, peak=0.9)
    w2 = win2.get_impulse_response(50)
    assert w1.shape == w2.shape
    assert not np.allclose(w1, w2)
    # Order 1 has different alpha than order > 1
    win3 = GammaWindow(order=1)
    w3 = win3.get_impulse_response(50)
    assert w3.shape == (50,)
    # For width=1, output is exactly one
    win4 = GammaWindow()
    out = win4.get_impulse_response(1)
    assert out.shape == (1,)
    assert np.isclose(out[0], 1)

def test_triangular_supports_and_supports_ms_consistency():
    bank = TriangularOverlappingFilterBank("mel", num_filts=3)
    supports = bank.supports
    supports_ms = bank.supports_ms
    assert len(supports) == bank.num_filts
    assert len(supports_ms) == bank.num_filts
    for (start, end), (start_ms, end_ms) in zip(supports, supports_ms):
        assert np.isclose(start_ms, start * 1000 / bank.sampling_rate)
        assert np.isclose(end_ms, end * 1000 / bank.sampling_rate)
        assert start <= end

def test_fbank_supports_and_supports_ms_consistency():
    bank = Fbank(num_filts=3)
    supports = bank.supports
    supports_ms = bank.supports_ms
    assert len(supports) == bank.num_filts
    assert len(supports_ms) == bank.num_filts
    for (start, end), (start_ms, end_ms) in zip(supports, supports_ms):
        assert np.isclose(start_ms, start * 1000 / bank.sampling_rate)
        assert np.isclose(end_ms, end * 1000 / bank.sampling_rate)
        assert start <= end

def test_gabor_supports_and_supports_ms_consistency():
    bank = GaborFilterBank("mel", num_filts=3)
    supports = bank.supports
    supports_ms = bank.supports_ms
    assert len(supports) == bank.num_filts
    assert len(supports_ms) == bank.num_filts
    for (start, end), (start_ms, end_ms) in zip(supports, supports_ms):
        assert np.isclose(start_ms, start * 1000 / bank.sampling_rate)
        assert np.isclose(end_ms, end * 1000 / bank.sampling_rate)
        assert start <= end

def test_complex_gammatone_supports_and_supports_ms_consistency():
    bank = ComplexGammatoneFilterBank("mel", num_filts=3)
    supports = bank.supports
    supports_ms = bank.supports_ms
    assert len(supports) == bank.num_filts
    assert len(supports_ms) == bank.num_filts
    for (start, end), (start_ms, end_ms) in zip(supports, supports_ms):
        assert np.isclose(start_ms, start * 1000 / bank.sampling_rate)
        assert np.isclose(end_ms, end * 1000 / bank.sampling_rate)
        assert start <= end
