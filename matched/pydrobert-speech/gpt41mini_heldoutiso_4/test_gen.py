import pytest
import numpy as np
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
    GammaWindow
)
from pydrobert.speech.scales import MelScaling, ScalingFunction

################################################################################
# Tests for TriangularOverlappingFilterBank
################################################################################

def test_triangular_overlapping_filter_bank_basic_properties():
    bank = TriangularOverlappingFilterBank(scaling_function="mel", num_filts=10, high_hz=4000, low_hz=20, sampling_rate=16000, analytic=False)
    assert bank.num_filts == 10
    assert bank.sampling_rate == 16000
    assert bank.is_real is True
    assert bank.is_analytic is False
    assert bank.is_zero_phase is True
    assert isinstance(bank.centers_hz, tuple)
    assert len(bank.centers_hz) == 10
    assert isinstance(bank.supports_hz, tuple)
    assert len(bank.supports_hz) == 10
    assert isinstance(bank.supports, tuple)
    assert len(bank.supports) == 10
    assert isinstance(bank.supports_ms, tuple)
    assert len(bank.supports_ms) == 10
    for (start, end) in bank.supports_ms:
        assert start <= end

def test_triangular_invalid_frequency_range():
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(scaling_function="mel", low_hz=-1)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(scaling_function="mel", low_hz=1000, high_hz=200, sampling_rate=16000)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(scaling_function="mel", high_hz=9000, sampling_rate=16000)

def test_triangular_get_impulse_response_real_and_analytic():
    bank_real = TriangularOverlappingFilterBank("mel", num_filts=3, analytic=False)
    bank_analytic = TriangularOverlappingFilterBank("mel", num_filts=3, analytic=True)
    width = 64
    for filt_idx in range(bank_real.num_filts):
        ir_real = bank_real.get_impulse_response(filt_idx, width)
        ir_analytic = bank_analytic.get_impulse_response(filt_idx, width)
        assert ir_real.shape == (width,)
        assert ir_analytic.shape == (width,)
        assert np.iscomplexobj(ir_analytic)
        assert not np.iscomplexobj(ir_real)

def test_triangular_get_frequency_response_half_and_full():
    bank = TriangularOverlappingFilterBank("mel", num_filts=5, analytic=False)
    width = 64
    filt_idx = 2
    freq_half = bank.get_frequency_response(filt_idx, width, half=True)
    freq_full = bank.get_frequency_response(filt_idx, width, half=False)
    # Half spectrum length depends on even/odd width
    expected_half_length = (width // 2 + 1) if (width % 2 == 0) else ((width + 1) // 2)
    assert freq_half.shape[0] == expected_half_length
    assert freq_full.shape[0] == width
    # ensure real dtype for non-analytic zero-phase filters
    assert freq_full.dtype == np.float64
    assert freq_half.dtype == np.float64

def test_triangular_get_truncated_response_consistency():
    bank = TriangularOverlappingFilterBank("mel", num_filts=5)
    width = 128
    filt_idx = 1
    bin_idx, truncated = bank.get_truncated_response(filt_idx, width)
    assert isinstance(bin_idx, int)
    assert isinstance(truncated, np.ndarray)
    assert truncated.ndim == 1
    # Reconstruct half frequency response from truncated data
    half_width = (width + width % 2) // 2 + 1 - width % 2
    half_response = np.zeros(half_width, dtype=np.float64)
    half_response[bin_idx:bin_idx + len(truncated)] = truncated
    # Check that values are in [0,1]
    assert np.all((half_response >= 0) & (half_response <= 1))

################################################################################
# Tests for Fbank (Mel triangular with sqrt)
################################################################################

def test_fbank_basic_properties_and_centers():
    fbank = Fbank(num_filts=12, low_hz=30., high_hz=7000., sampling_rate=16000, analytic=False)
    assert fbank.num_filts == 12
    assert isinstance(fbank.centers_hz, tuple)
    assert len(fbank.centers_hz) == 12
    assert fbank.is_real is True
    assert fbank.is_zero_phase is True
    assert fbank.is_analytic is False

def test_fbank_invalid_frequency_range_raises():
    with pytest.raises(ValueError):
        Fbank(low_hz=-5)
    with pytest.raises(ValueError):
        Fbank(low_hz=1000, high_hz=500, sampling_rate=16000)
    with pytest.raises(ValueError):
        Fbank(high_hz=16000, sampling_rate=16000)

def test_fbank_get_impulse_response_shapes():
    fbank = Fbank(num_filts=6, analytic=True)
    width = 64
    for filt_idx in range(fbank.num_filts):
        ir = fbank.get_impulse_response(filt_idx, width)
        assert isinstance(ir, np.ndarray)
        assert ir.shape == (width,)
        # analytic=True means complex output (ifft)
        assert np.iscomplexobj(ir)

    fbank2 = Fbank(num_filts=6, analytic=False)
    for filt_idx in range(fbank2.num_filts):
        ir2 = fbank2.get_impulse_response(filt_idx, width)
        # real output (irfft)
        assert not np.iscomplexobj(ir2)
        assert ir2.shape == (width,)

def test_fbank_get_frequency_response_half_and_full():
    fbank = Fbank(num_filts=5, analytic=False)
    width = 64
    filt_idx = 3
    freq_half = fbank.get_frequency_response(filt_idx, width, half=True)
    freq_full = fbank.get_frequency_response(filt_idx, width, half=False)
    expected_half_size = (width // 2 + 1) if width % 2 == 0 else ((width + 1) // 2)
    assert freq_half.shape[0] == expected_half_size
    assert freq_full.shape[0] == width
    # Values between 0 and 1 (due to sqrt of triangle)
    assert np.all((freq_full >= 0) & (freq_full <= 1))
    assert freq_half.dtype == np.float64

def test_fbank_get_truncated_response_values_and_index():
    fbank = Fbank(num_filts=5)
    width = 128
    for filt_idx in range(fbank.num_filts):
        bin_idx, truncated = fbank.get_truncated_response(filt_idx, width)
        assert isinstance(bin_idx, int)
        assert isinstance(truncated, np.ndarray)
        assert truncated.ndim == 1
        assert np.all((truncated >= 0) & (truncated <= 1))

################################################################################
# Tests for GaborFilterBank (complex analytic Gaussians)
################################################################################

def test_gabor_basic_properties():
    bank = GaborFilterBank("mel", num_filts=10, high_hz=4000, low_hz=50, sampling_rate=16000, scale_l2_norm=True, erb=True)
    assert bank.num_filts == 10
    assert bank.is_real is False
    assert isinstance(bank.centers_hz, tuple)
    assert isinstance(bank.supports_hz, tuple)
    assert isinstance(bank.supports, tuple)
    assert bank.scaled_l2_norm is True
    assert bank.erb is True

def test_gabor_get_impulse_response_and_shapes():
    bank = GaborFilterBank("mel", num_filts=3)
    width = 64
    for filt_idx in range(bank.num_filts):
        ir = bank.get_impulse_response(filt_idx, width)
        assert isinstance(ir, np.ndarray)
        assert ir.shape == (width,)
        assert np.iscomplexobj(ir)

def test_gabor_get_frequency_response_half_and_full():
    bank = GaborFilterBank("mel", num_filts=4)
    width = 64
    filt_idx = 1
    freq_full = bank.get_frequency_response(filt_idx, width, half=False)
    freq_half = bank.get_frequency_response(filt_idx, width, half=True)
    assert freq_full.shape[0] == width
    half_len = (width + 1) // 2 if width % 2 else (width // 2 + 1)
    assert freq_half.shape[0] == half_len
    # Complex128 dtype expected
    assert np.iscomplexobj(freq_full)
    assert np.iscomplexobj(freq_half)

def test_gabor_get_truncated_response_index_and_values():
    bank = GaborFilterBank("mel", num_filts=5)
    width = 128
    for filt_idx in range(bank.num_filts):
        bin_idx, truncated = bank.get_truncated_response(filt_idx, width)
        assert isinstance(bin_idx, int)
        assert isinstance(truncated, np.ndarray)
        assert truncated.ndim == 1
        # truncated values are floats (magnitude)
        assert truncated.dtype == np.float64 or np.issubdtype(truncated.dtype, np.floating)

################################################################################
# Tests for ComplexGammatoneFilterBank
################################################################################

def test_complex_gammatone_properties_and_invalid_order():
    bank = ComplexGammatoneFilterBank("mel", num_filts=6, order=3, scale_l2_norm=True, erb=True)
    assert bank.order == 3
    assert bank.num_filts == 6
    assert bank.scaled_l2_norm is True
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", order=0)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", order=-1)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", order=3.5)

def test_complex_gammatone_get_impulse_response_and_shape():
    bank = ComplexGammatoneFilterBank("mel", num_filts=5, order=4)
    width = 64
    for idx in range(bank.num_filts):
        ir = bank.get_impulse_response(idx, width)
        assert isinstance(ir, np.ndarray)
        assert ir.shape == (width,)
        assert np.iscomplexobj(ir)

def test_complex_gammatone_get_frequency_response_shapes_and_type():
    bank = ComplexGammatoneFilterBank("mel", num_filts=4, order=2)
    width = 64
    filt_idx = 2
    freq_full = bank.get_frequency_response(filt_idx, width, half=False)
    freq_half = bank.get_frequency_response(filt_idx, width, half=True)
    assert freq_full.shape[0] == width
    half_expected = (width + 1) // 2 if width % 2 else (width // 2 + 1)
    assert freq_half.shape[0] == half_expected
    assert np.iscomplexobj(freq_full)
    assert np.iscomplexobj(freq_half)

def test_complex_gammatone_get_truncated_response_values_and_index():
    bank = ComplexGammatoneFilterBank("mel", num_filts=3, order=3)
    width = 128
    for filt_idx in range(bank.num_filts):
        bin_idx, truncated = bank.get_truncated_response(filt_idx, width)
        assert isinstance(bin_idx, int)
        assert isinstance(truncated, np.ndarray)
        assert truncated.ndim == 1
        assert np.iscomplexobj(truncated) or np.issubdtype(truncated.dtype, np.floating)

################################################################################
# Tests for WindowFunction Subclasses
################################################################################

@pytest.mark.parametrize("window_cls", [
    BartlettWindow,
    BlackmanWindow,
    HammingWindow,
    HannWindow
])
@pytest.mark.parametrize("width", [1, 5, 10, 50])
def test_window_functions_return_correct_shapes_and_normalization(window_cls, width):
    window = window_cls()
    w = window.get_impulse_response(width)
    assert isinstance(w, np.ndarray)
    assert w.shape == (width,)
    # Values should be non-negative and normalized sum > 0
    assert np.all(w >= 0)
    assert w.sum() > 0

def test_gamma_window_empty_and_single_width():
    gamma = GammaWindow(order=4, peak=0.7)
    w0 = gamma.get_impulse_response(0)
    w1 = gamma.get_impulse_response(1)
    assert isinstance(w0, np.ndarray)
    assert w0.size == 0
    assert isinstance(w1, np.ndarray)
    assert w1.size == 1
    assert w1[0] == 1

def test_gamma_window_behavior_order_and_peak():
    widths = [5, 10]
    orders = [1, 3, 5]
    peaks = [0.25, 0.5, 0.75]
    for order in orders:
        for peak in peaks:
            gamma = GammaWindow(order=order, peak=peak)
            for width in widths:
                w = gamma.get_impulse_response(width)
                assert isinstance(w, np.ndarray)
                # Values non-negative, size equals width unless width=0
                assert w.size == width
                assert np.all(w >= 0)
                # For order == 1 alpha sets approximate rate, others larger decay
                if order == 1:
                    assert np.all(w[:width] >= 0)

################################################################################
# Additional internal consistency test examples
################################################################################

def test_triangular_supports_are_valid_intervals():
    bank = TriangularOverlappingFilterBank("mel", num_filts=8)
    supports = bank.supports
    for start, end in supports:
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert start < end

def test_fbank_supports_are_valid_intervals():
    bank = Fbank(num_filts=8)
    supports = bank.supports
    for start, end in supports:
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert start < end

def test_gabor_supports_are_valid_intervals():
    bank = GaborFilterBank("mel", num_filts=8)
    supports = bank.supports
    for start, end in supports:
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert start < end

def test_complex_gammatone_supports_are_valid_intervals():
    bank = ComplexGammatoneFilterBank("mel", num_filts=8)
    supports = bank.supports
    for start, end in supports:
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert start < end
