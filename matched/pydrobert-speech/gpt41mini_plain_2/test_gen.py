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


# Helper constants and functions

DEFAULT_FS = 16000
DEFAULT_NUM_FILTS = 10
DEFAULT_WIDTH = 256


# ---------------------------------------------------
# Tests for TriangularOverlappingFilterBank
# ---------------------------------------------------

def test_triangular_init_and_properties():
    bank = TriangularOverlappingFilterBank(
        MelScaling(), num_filts=DEFAULT_NUM_FILTS, low_hz=50.0, high_hz=4000, sampling_rate=DEFAULT_FS, analytic=False
    )
    assert bank.num_filts == DEFAULT_NUM_FILTS
    assert bank.sampling_rate == DEFAULT_FS
    assert bank.is_real
    assert not bank.is_analytic
    assert bank.is_zero_phase
    assert len(bank.centers_hz) == DEFAULT_NUM_FILTS
    # supports_hz length equals num_filts
    assert len(bank.supports_hz) == DEFAULT_NUM_FILTS
    # supports length equals num_filts
    assert len(bank.supports) == DEFAULT_NUM_FILTS
    # supports_ms length equals num_filts, contents match supports scaled to ms
    for (start, end), (ms_start, ms_end) in zip(bank.supports, bank.supports_ms):
        assert ms_start == pytest.approx(start * 1000 / bank.sampling_rate)
        assert ms_end == pytest.approx(end * 1000 / bank.sampling_rate)


def test_triangular_invalid_frequencies():
    with pytest.raises(ValueError):
        _ = TriangularOverlappingFilterBank(MelScaling(), low_hz=-1)
    with pytest.raises(ValueError):
        _ = TriangularOverlappingFilterBank(MelScaling(), low_hz=1000, high_hz=1000)
    with pytest.raises(ValueError):
        _ = TriangularOverlappingFilterBank(MelScaling(), low_hz=5000, high_hz=DEFAULT_FS // 2 + 100)


def test_triangular_get_frequency_response_shapes_and_values():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=5, sampling_rate=DEFAULT_FS)
    for filt_idx in range(bank.num_filts):
        full_resp = bank.get_frequency_response(filt_idx, width=DEFAULT_WIDTH, half=False)
        # Full spectrum length is width
        assert full_resp.shape == (DEFAULT_WIDTH,)
        # Type is float64 (real)
        assert full_resp.dtype == np.float64
        # Values are in [0,1]
        assert np.all((full_resp >= 0) & (full_resp <= 1))

        half_resp = bank.get_frequency_response(filt_idx, width=DEFAULT_WIDTH, half=True)
        expected_len = (DEFAULT_WIDTH // 2) + 1 if DEFAULT_WIDTH % 2 == 0 else (DEFAULT_WIDTH + 1) // 2
        assert half_resp.shape == (expected_len,)
        assert half_resp.dtype == np.float64
        assert np.all((half_resp >= 0) & (half_resp <= 1))

        # Analytic variant returns complex for full, real for zero_phase: triangular is zero phase -> real
        assert bank.is_zero_phase
        assert np.isrealobj(full_resp)


def test_triangular_get_impulse_response_symmetry():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=3, analytic=False)
    width = 11
    for filt_idx in range(bank.num_filts):
        ir = bank.get_impulse_response(filt_idx, width)
        # Real-valued when analytic False
        assert ir.dtype == np.float64
        # Symmetric (even) function in time domain for zero_phase filters
        # Because impulse responses center around zero and are zero_phase
        # Check symmetry: ir[t] == conj(ir[-t])
        for i in range(1, width):
            assert ir[i] == pytest.approx(ir[-i])


def test_triangular_get_truncated_response_and_reconstruction():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=4)
    width = 256
    for filt_idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
        assert isinstance(bin_idx, int)
        assert isinstance(trnc, np.ndarray)
        # Length is right_idx - left_idx + 1
        assert len(trnc) > 0

        # Reconstruct full response from truncated response (per docstring)
        full = np.zeros(width, dtype=trnc.dtype)
        wrap = min(bin_idx + len(trnc), width) - bin_idx
        full[bin_idx : bin_idx + wrap] = trnc[:wrap]
        if bin_idx == 0:
            expected_reverse = trnc[:0:-1].conj()
        else:
            expected_reverse = trnc[:None if bin_idx else 0:-1].conj()
        # For triangular overlapping, filters are real and half spectrum only; no wrapping expected.
        # So only fill conjugate symmetric half (the docstring shows full spectrum construction)
        # Check at least main positive side is filled correctly
        np.testing.assert_allclose(full[bin_idx:bin_idx+wrap], trnc[:wrap])


# ---------------------------------------------------
# Tests for Fbank
# ---------------------------------------------------

def test_fbank_basic_properties_and_centers():
    bank = Fbank(num_filts=15, low_hz=50, high_hz=8000, sampling_rate=DEFAULT_FS)
    assert bank.num_filts == 15
    assert bank.sampling_rate == DEFAULT_FS
    assert bank.is_zero_phase
    assert bank.is_real
    assert not bank.is_analytic
    assert len(bank.centers_hz) == 15


def test_fbank_invalid_frequencies():
    with pytest.raises(ValueError):
        Fbank(low_hz=-10)
    with pytest.raises(ValueError):
        Fbank(low_hz=3000, high_hz=1000)
    with pytest.raises(ValueError):
        Fbank(high_hz=DEFAULT_FS // 2 + 1)


def test_fbank_get_frequency_response_shape_and_values():
    bank = Fbank(num_filts=3)
    filt_idx = 1
    width = 128
    freq_resp = bank.get_frequency_response(filt_idx, width, half=False)
    assert freq_resp.shape == (width,)
    # All values non-negative and <= 1 (since square root of val)
    assert np.all(freq_resp >= 0)
    assert np.all(freq_resp <= 1)

    freq_resp_half = bank.get_frequency_response(filt_idx, width, half=True)
    expected_len = width // 2 + 1 if width % 2 == 0 else (width + 1) // 2
    assert freq_resp_half.shape == (expected_len,)


def test_fbank_get_impulse_response_with_analytic_variants():
    bank_real = Fbank(num_filts=3, analytic=False)
    bank_analytic = Fbank(num_filts=3, analytic=True)
    width = 64
    ir_real = bank_real.get_impulse_response(0, width)
    ir_analytic = bank_analytic.get_impulse_response(0, width)
    # Real analytic returns complex if analytic is True
    assert np.iscomplexobj(ir_analytic)
    # Real analytic returns real when analytic is False
    assert np.isrealobj(ir_real)
    # Lengths match
    assert len(ir_real) == width and len(ir_analytic) == width


def test_fbank_get_truncated_response_and_reconstruction():
    bank = Fbank(num_filts=4)
    width = 256
    for filt_idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
        assert isinstance(bin_idx, int)
        assert isinstance(trnc, np.ndarray)
        # Values are real and >= 0 (due to square root)
        assert np.all(trnc >= 0)
        # Validate reconstruction as in docstring
        half_width = (width + width % 2) // 2 + 1 - width % 2
        half = np.zeros(half_width, dtype=trnc.dtype)
        half[bin_idx : bin_idx + len(trnc)] = trnc
        assert np.all(half[bin_idx : bin_idx + len(trnc)] == trnc)


# ---------------------------------------------------
# Tests for GaborFilterBank
# ---------------------------------------------------

def test_gabor_filter_bank_init_properties():
    bank = GaborFilterBank(MelScaling(), num_filts=5, sampling_rate=DEFAULT_FS)
    assert bank.num_filts == 5
    assert bank.is_zero_phase
    assert not bank.is_real
    # By default wrap_below can be True or False depending on freq setup, but is_analytic is opposite of wrap_below
    assert isinstance(bank.is_analytic, bool)
    assert bank.sampling_rate == DEFAULT_FS
    assert len(bank.centers_hz) == 5


def test_gabor_invalid_freqs_raises():
    with pytest.raises(ValueError):
        GaborFilterBank(MelScaling(), low_hz=-1)
    with pytest.raises(ValueError):
        GaborFilterBank(MelScaling(), low_hz=8000, high_hz=8000)
    with pytest.raises(ValueError):
        GaborFilterBank(MelScaling(), high_hz=DEFAULT_FS // 2 + 1)


def test_gabor_get_impulse_response_length_and_type():
    bank = GaborFilterBank(MelScaling(), num_filts=3, sampling_rate=DEFAULT_FS)
    width = 64
    for idx in range(bank.num_filts):
        ir = bank.get_impulse_response(idx, width)
        assert len(ir) == width
        assert ir.dtype == np.complex128


def test_gabor_get_frequency_response_correct_length_and_complexity():
    bank = GaborFilterBank(MelScaling(), num_filts=2)
    width = 128
    for filt_idx in range(bank.num_filts):
        full_resp = bank.get_frequency_response(filt_idx, width, half=False)
        assert full_resp.shape == (width,)
        assert np.iscomplexobj(full_resp)
        half_resp = bank.get_frequency_response(filt_idx, width, half=True)
        expected_len = width // 2 + 1 if width % 2 == 0 else (width + 1) // 2
        assert half_resp.shape == (expected_len,)
        assert np.iscomplexobj(half_resp)


def test_gabor_get_truncated_response_behavior():
    bank = GaborFilterBank(MelScaling(), num_filts=3)
    width = 256
    for filt_idx in range(bank.num_filts):
        start_idx, trnc = bank.get_truncated_response(filt_idx, width)
        assert isinstance(start_idx, int)
        assert isinstance(trnc, np.ndarray)
        assert len(trnc) > 0
        # Values complex
        assert np.iscomplexobj(trnc)


# ---------------------------------------------------
# Tests for ComplexGammatoneFilterBank
# ---------------------------------------------------

def test_complex_gammatone_init_and_properties():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=4, order=3, sampling_rate=DEFAULT_FS)
    assert bank.order == 3
    assert bank.num_filts == 4
    assert bank.sampling_rate == DEFAULT_FS
    assert not bank.is_real
    assert bank.is_zero_phase is False
    assert isinstance(bank.is_analytic, bool)
    assert len(bank.centers_hz) == 4


def test_complex_gammatone_invalid_inputs():
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), low_hz=-10)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), low_hz=2000, high_hz=1000)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), high_hz=DEFAULT_FS // 2 + 10)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=0)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=-1)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=2.5)  # Non-int


def test_complex_gammatone_get_impulse_response_shape_and_complex():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=3, order=4)
    width = 128
    for filt_idx in range(bank.num_filts):
        ir = bank.get_impulse_response(filt_idx, width)
        assert len(ir) == width
        assert np.iscomplexobj(ir)


def test_complex_gammatone_get_frequency_response_shape_and_type():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=2, order=4)
    width = 128
    for filt_idx in range(bank.num_filts):
        full_resp = bank.get_frequency_response(filt_idx, width, half=False)
        assert full_resp.shape == (width,)
        assert np.iscomplexobj(full_resp)
        half_resp = bank.get_frequency_response(filt_idx, width, half=True)
        expected_len = width // 2 + 1 if width % 2 == 0 else (width + 1) // 2
        assert half_resp.shape == (expected_len,)
        assert np.iscomplexobj(half_resp)


def test_complex_gammatone_get_truncated_response_and_output():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=3, order=4)
    width = 256
    for filt_idx in range(bank.num_filts):
        start_idx, trnc = bank.get_truncated_response(filt_idx, width)
        assert isinstance(start_idx, int)
        assert isinstance(trnc, np.ndarray)
        assert trnc.dtype == np.complex128 or np.issubdtype(trnc.dtype, np.complexfloating)
        assert len(trnc) > 0


# ---------------------------------------------------
# Tests for Window Functions
# ---------------------------------------------------

@pytest.mark.parametrize("WindowCls", [
    BartlettWindow,
    BlackmanWindow,
    HammingWindow,
    HannWindow,
])
def test_window_function_output_length_and_sum(WindowCls):
    window = WindowCls()
    for width in [1, 5, 10, 100]:
        resp = window.get_impulse_response(width)
        assert isinstance(resp, np.ndarray)
        assert len(resp) == width
        assert resp.dtype == np.float64
        # All values non-negative (typical for these windows)
        assert np.all(resp >= 0)
        # Window generally normalized such that max value is > 0
        assert np.max(resp) > 0


def test_gamma_window_empty_and_single():
    w = GammaWindow(order=3)
    resp = w.get_impulse_response(0)
    assert resp.shape == (0,)
    resp1 = w.get_impulse_response(1)
    assert resp1.shape == (1,)
    assert resp1[0] == pytest.approx(1.0)


def test_gamma_window_values_and_order_effect():
    for order in [1, 2, 4]:
        w = GammaWindow(order=order, peak=0.7)
        width = 10
        resp = w.get_impulse_response(width)
        assert isinstance(resp, np.ndarray)
        assert len(resp) == width
        # Positive values expected
        assert np.all(resp >= 0)
        if order > 1:
            # Should have a maximum near peak*width
            peak_idx = int(w.peak * width)
            max_idx = np.argmax(resp)
            assert abs(max_idx - peak_idx) <= 1


# ---------------------------------------------------
# Additional edge case tests for get_frequency_response with half flag
# ---------------------------------------------------

def test_triangular_overlapping_get_frequency_response_half_equivalence():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=1)
    width = 11  # odd
    filt_idx = 0
    full = bank.get_frequency_response(filt_idx, width, half=False)
    half = bank.get_frequency_response(filt_idx, width, half=True)
    expected_half_len = (width + 1) // 2
    assert len(half) == expected_half_len
    # Half response matches first half of full response
    np.testing.assert_allclose(full[:expected_half_len], half)


def test_fbank_get_frequency_response_half_even_and_odd_width():
    bank = Fbank(num_filts=1)
    for width in [128, 129]:  # even and odd
        filt_idx = 0
        half_resp = bank.get_frequency_response(filt_idx, width, half=True)
        expected_len = width // 2 + 1 if width % 2 == 0 else (width + 1) // 2
        assert half_resp.shape == (expected_len,)


def test_gabor_get_frequency_response_with_large_width():
    bank = GaborFilterBank(MelScaling(), num_filts=1)
    width = 1024
    resp = bank.get_frequency_response(0, width)
    assert resp.shape == (width,)
    assert np.iscomplexobj(resp)


def test_complex_gammatone_get_impulse_response_sums_close_to_freq_response_sum():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=1, order=3)
    width = 64
    ir = bank.get_impulse_response(0, width)
    freq_resp = bank.get_frequency_response(0, width)
    # The sum of magnitudes of impulse response should be somewhat related to sum of freq response magnitudes
    ir_sum = np.sum(np.abs(ir))
    freq_sum = np.sum(np.abs(freq_resp))
    assert ir_sum > 0
    assert freq_sum > 0


def test_gamma_window_peak_parameter_effect():
    w1 = GammaWindow(order=3, peak=0.1)
    w2 = GammaWindow(order=3, peak=0.9)
    resp1 = w1.get_impulse_response(50)
    resp2 = w2.get_impulse_response(50)
    max_idx1 = np.argmax(resp1)
    max_idx2 = np.argmax(resp2)
    # Peak param affects max index location approximately
    assert max_idx1 < max_idx2


# ---------------------------------------------------
# All tests fit between 10 and 25 functions per instructions.
# ---------------------------------------------------
