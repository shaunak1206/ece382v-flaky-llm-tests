import pytest
import numpy as np

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
from pydrobert.speech.scales import MelScaling, ScalingFunction


# ---- Helpers ----

def is_nonnegative_real_array(arr):
    return np.all(np.isreal(arr)) and np.all(arr >= 0)


# ---- Tests for TriangularOverlappingFilterBank ----

def test_triangular_init_default():
    bank = TriangularOverlappingFilterBank('mel', num_filts=10)
    assert bank.num_filts == 10
    assert bank.is_real
    assert not bank.is_analytic
    assert bank.is_zero_phase
    assert bank.sampling_rate == 16000
    assert callable(getattr(bank, "get_impulse_response"))
    assert callable(getattr(bank, "get_frequency_response"))
    assert callable(getattr(bank, "get_truncated_response"))

def test_triangular_invalid_freq_range_errors():
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank('mel', low_hz=-1)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank('mel', high_hz=100, low_hz=200)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank('mel', high_hz=9000, low_hz=20, sampling_rate=16000)  # 9000 > 16000/2=8000

def test_triangular_centers_and_supports_consistency():
    bank = TriangularOverlappingFilterBank('mel', num_filts=5)
    centers = bank.centers_hz
    supports_hz = bank.supports_hz
    supports = bank.supports
    # Centers length = filters count
    assert len(centers) == bank.num_filts
    assert len(supports_hz) == bank.num_filts
    assert len(supports) == bank.num_filts

    # centers are inside supports_hz ranges
    for i in range(bank.num_filts):
        low, high = supports_hz[i]
        center = centers[i]
        assert low <= center <= high

def test_triangular_impulse_response_shape_and_dtype():
    bank = TriangularOverlappingFilterBank('mel', num_filts=3, analytic=False)
    width = 30
    for idx in range(bank.num_filts):
        imp_resp = bank.get_impulse_response(idx, width)
        assert isinstance(imp_resp, np.ndarray)
        assert imp_resp.shape == (width,)
        assert imp_resp.dtype == np.float64
        # symmetry check (zero phase even function)
        np.testing.assert_allclose(imp_resp.real, imp_resp.real[::-1], atol=1e-12)

def test_triangular_impulse_response_analytic_dtype():
    bank = TriangularOverlappingFilterBank('mel', num_filts=3, analytic=True)
    imp_resp = bank.get_impulse_response(0, 20)
    assert imp_resp.dtype == np.complex128

def test_triangular_frequency_response_basic():
    bank = TriangularOverlappingFilterBank('mel', num_filts=3)
    width = 64
    for idx in range(bank.num_filts):
        freq_resp = bank.get_frequency_response(idx, width, half=False)
        assert freq_resp.shape == (width,)
        # Real dtype if zero phase and not analytic
        assert freq_resp.dtype == np.float64 or freq_resp.dtype == np.complex128
        # Values between 0 and 1 expected
        assert np.all(freq_resp >= 0)
        assert np.all(freq_resp <= 1)
    # Half spectrum responses shape
    h_resp = bank.get_frequency_response(0, width, half=True)
    expected_len = width // 2 + 1
    assert h_resp.shape[0] == expected_len or h_resp.shape[0] == (width + 1) // 2

def test_triangular_get_truncated_response():
    bank = TriangularOverlappingFilterBank('mel', num_filts=1)
    width = 100
    left_idx, buf = bank.get_truncated_response(0, width)
    # left_idx within valid DFT range
    assert 0 <= left_idx < width
    # buf is a numpy array of floats
    assert isinstance(buf, np.ndarray)
    assert buf.dtype == np.float64
    # All values between 0 and 1
    assert np.all(buf >= 0)
    assert np.all(buf <= 1)


# ---- Tests for Fbank ----

def test_fbank_init_and_properties():
    bank = Fbank(num_filts=5, low_hz=50, high_hz=7000, sampling_rate=16000)
    assert bank.num_filts == 5
    assert bank.is_real
    assert not bank.is_analytic
    assert bank.is_zero_phase
    assert abs(bank.sampling_rate - 16000) < 1e-6
    assert isinstance(bank.centers_hz, tuple)
    # supports_hz and supports lengths
    assert len(bank.supports_hz) == bank.num_filts
    assert len(bank.supports) == bank.num_filts

def test_fbank_impulse_response_half_and_full():
    bank = Fbank(num_filts=3, analytic=False)
    width = 50
    imp_resp_half = []
    imp_resp_full = []
    for i in range(bank.num_filts):
        imp_half = bank.get_impulse_response(i, width)
        imp_full = bank.get_frequency_response(i, width, half=False)
        imp_resp_half.append(imp_half)
        imp_resp_full.append(imp_full)
        # Check shape
        assert imp_half.shape == (width,)
        assert imp_full.shape == (width,)

def test_fbank_frequency_response_values():
    bank = Fbank(num_filts=3, analytic=False)
    width = 128
    for idx in range(bank.num_filts):
        freq_resp = bank.get_frequency_response(idx, width, half=True)
        # Values should be real and nonnegative
        assert freq_resp.dtype == np.float64
        assert np.all(freq_resp >= 0)

def test_fbank_get_truncated_response_sizes():
    bank = Fbank(num_filts=2)
    width = 128
    for idx in range(bank.num_filts):
        left_idx, truncated = bank.get_truncated_response(idx, width)
        assert 0 <= left_idx < width
        assert isinstance(truncated, np.ndarray)
        assert truncated.dtype == np.float64
        assert truncated.size > 0


# ---- Tests for GaborFilterBank ----

def test_gabor_init_basic_and_attributes():
    bank = GaborFilterBank('mel', num_filts=5)
    assert bank.num_filts == 5
    assert not bank.is_real   # always complex
    assert bank.is_zero_phase
    # Usually analytic (no wrap below for these parameters)
    assert isinstance(bank.is_analytic, bool)
    assert hasattr(bank, 'scaled_l2_norm')
    assert hasattr(bank, 'erb')
    assert isinstance(bank.centers_hz, tuple)
    assert len(bank.centers_hz) == bank.num_filts

def test_gabor_impulse_response_shape_and_type():
    bank = GaborFilterBank('mel', num_filts=3, scale_l2_norm=True)
    width = 40
    for idx in range(bank.num_filts):
        imp = bank.get_impulse_response(idx, width)
        assert isinstance(imp, np.ndarray)
        assert imp.shape == (width,)
        assert imp.dtype == np.complex128

def test_gabor_frequency_response_shape_and_realness():
    bank = GaborFilterBank('mel', num_filts=2)
    width = 80
    for idx in range(bank.num_filts):
        freq_resp = bank.get_frequency_response(idx, width, half=False)
        # dtype float64 (real) output expected here
        assert isinstance(freq_resp, np.ndarray)
        assert freq_resp.shape == (width,)
        assert freq_resp.dtype == np.float64
        # Values nonnegative
        assert np.all(freq_resp >= 0)

def test_gabor_get_truncated_response_wrap_condition():
    bank = GaborFilterBank('mel', num_filts=2)
    width = 100
    for idx in range(bank.num_filts):
        left_idx, truncated = bank.get_truncated_response(idx, width)
        assert isinstance(left_idx, int)
        assert isinstance(truncated, np.ndarray)
        assert truncated.dtype == np.float64
        assert truncated.size > 0


# ---- Tests for ComplexGammatoneFilterBank ----

def test_gammatone_init_and_properties():
    bank = ComplexGammatoneFilterBank('mel', num_filts=4, order=3, max_centered=True)
    assert bank.num_filts == 4
    assert bank.order == 3
    assert bank.is_zero_phase is False
    assert bank.is_real is False
    assert bank.sampling_rate == 16000

def test_gammatone_invalid_order_errors():
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank('mel', order=0)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank('mel', order=-5)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank('mel', order=2.5)

def test_gammatone_impulse_response_complex_and_shape():
    bank = ComplexGammatoneFilterBank('mel', num_filts=2, order=4)
    width = 25
    for i in range(bank.num_filts):
        imp = bank.get_impulse_response(i, width)
        assert isinstance(imp, np.ndarray)
        assert imp.shape == (width,)
        assert imp.dtype == np.complex128

def test_gammatone_frequency_response_complex_and_shape():
    bank = ComplexGammatoneFilterBank('mel', num_filts=3, order=2)
    width = 60
    for i in range(bank.num_filts):
        freq = bank.get_frequency_response(i, width, half=True)
        assert isinstance(freq, np.ndarray)
        assert freq.dtype == np.complex128
        # length correct depending on half param
        expected_len = (width // 2 + 1) if width % 2 == 0 else (width + 1) // 2
        assert freq.shape[0] == expected_len

def test_gammatone_get_truncated_response_basic():
    bank = ComplexGammatoneFilterBank('mel', num_filts=1)
    width = 100
    left_idx, truncated_resp = bank.get_truncated_response(0, width)
    assert isinstance(left_idx, int)
    assert left_idx >= 0
    assert isinstance(truncated_resp, np.ndarray)
    assert truncated_resp.dtype == np.complex128 or truncated_resp.dtype == np.float64
    assert truncated_resp.size > 0


# ---- Tests for window functions ----

@pytest.mark.parametrize(
    "WindowClass",
    [BartlettWindow, BlackmanWindow, HammingWindow, HannWindow]
)
def test_window_function_shapes_and_normalization(WindowClass):
    win = WindowClass()
    widths = [1, 5, 10]
    for w in widths:
        imp = win.get_impulse_response(w)
        assert isinstance(imp, np.ndarray)
        assert imp.shape == (w,)
        # Values should be floats
        assert np.issubdtype(imp.dtype, np.floating)
        if w > 1:
            # Peak value check for normalized windows (peak near 2)
            assert np.isclose(np.max(imp), 2, rtol=1e-2)

def test_gamma_window_edge_cases():
    w1 = GammaWindow(order=3, peak=0.5)
    # Width 0 returns empty array
    assert w1.get_impulse_response(0).size == 0
    # Width 1 returns array with one element == 1
    assert np.allclose(w1.get_impulse_response(1), np.array([1]))
    # Width 10 returns with correct shape and positive values
    w10 = w1.get_impulse_response(10)
    assert w10.shape == (10,)
    assert np.all(w10 >= 0)

def test_gamma_window_order_1_behavior():
    w2 = GammaWindow(order=1, peak=0.7)
    imp = w2.get_impulse_response(10)
    # Check shape and basic properties
    assert imp.shape == (10,)
    assert np.all(imp >= 0)
    # The output is not all zeros and contains positive values
    assert np.any(imp > 0)
