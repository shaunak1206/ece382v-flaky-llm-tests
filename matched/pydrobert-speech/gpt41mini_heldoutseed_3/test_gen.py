import math
import numpy as np
import pytest

from pydrobert.speech.filters import (
    LinearFilterBank,
    TriangularOverlappingFilterBank,
    Fbank,
    GaborFilterBank,
    ComplexGammatoneFilterBank,
)


@pytest.mark.parametrize("alias", ["triangular", "tri"])
def test_triangularoverlappingfilterbank_aliases(alias):
    bank = TriangularOverlappingFilterBank(alias)
    assert isinstance(bank, TriangularOverlappingFilterBank)


def test_triangular_filterbank_basic_properties():
    bank = TriangularOverlappingFilterBank("mel", num_filts=3, low_hz=100, high_hz=4000, sampling_rate=16000)
    assert bank.num_filts == 3
    assert bank.is_real is True
    assert bank.is_analytic is False
    assert bank.is_zero_phase is True
    assert bank.sampling_rate == 16000
    # centers_hz in ascending order and inside [low_hz, high_hz]
    for c in bank.centers_hz:
        assert 100 <= c <= 4000
    assert len(bank.supports_hz) == 3
    assert len(bank.supports) == 3
    assert len(bank.supports_ms) == 3
    # supports in int pairs
    for support in bank.supports:
        assert isinstance(support, tuple)
        assert all(isinstance(x, int) for x in support)
    # supports_ms should be floats and scaled correctly
    for (start_ms, end_ms), (start_s, end_s) in zip(bank.supports_ms, bank.supports):
        assert math.isclose(start_ms, start_s * 1000 / bank.sampling_rate)
        assert math.isclose(end_ms, end_s * 1000 / bank.sampling_rate)


def test_triangular_filterbank_invalid_freq_range():
    with pytest.raises(ValueError):
        _ = TriangularOverlappingFilterBank("mel", low_hz=-10)
    with pytest.raises(ValueError):
        _ = TriangularOverlappingFilterBank("mel", high_hz=4000, low_hz=5000)
    with pytest.raises(ValueError):
        _ = TriangularOverlappingFilterBank("mel", high_hz=9000, sampling_rate=16000)


@pytest.mark.parametrize("analytic", [True, False])
def test_triangular_filterbank_get_impulse_response_shape_and_type(analytic):
    bank = TriangularOverlappingFilterBank("mel", num_filts=2, analytic=analytic)
    width = 50
    for filt_idx in range(bank.num_filts):
        resp = bank.get_impulse_response(filt_idx, width)
        assert resp.shape == (width,)
        # analytic filters are complex, non-analytic float
        if analytic:
            assert np.iscomplexobj(resp)
        else:
            assert resp.dtype == np.float64


@pytest.mark.parametrize("half", [True, False])
def test_triangular_filterbank_get_frequency_response_basic(half):
    bank = TriangularOverlappingFilterBank("mel", num_filts=3)
    width = 64
    for filt_idx in range(bank.num_filts):
        resp = bank.get_frequency_response(filt_idx, width, half=half)
        if half:
            if width % 2 == 0:
                expected_len = width // 2 + 1
            else:
                expected_len = (width + 1) // 2
            assert resp.shape == (expected_len,)
        else:
            assert resp.shape == (width,)
        assert resp.dtype == np.float64
        # Values between 0 and 1 as these are triangular filters
        assert np.all(resp >= 0)
        assert np.all(resp <= 1)


def test_triangular_filterbank_get_frequency_response_symmetry():
    bank = TriangularOverlappingFilterBank("mel", num_filts=1, analytic=False)
    width = 32
    resp_full = bank.get_frequency_response(0, width, half=False)
    # This filter is real and non-analytic, response should be symmetric across frequency bins 1..N-1
    # resp_full[0] is DC, resp_full[-0] == resp_full[0], check symmetry for indices 1 to end
    for i in range(1, width):
        assert math.isclose(resp_full[i], resp_full[-i], rel_tol=1e-12)


def test_triangular_filterbank_get_truncated_response_consistency():
    bank = TriangularOverlappingFilterBank("mel", num_filts=3)
    width = 128
    for filt_idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
        # bin_idx is integer and valid index in [0,width)
        assert isinstance(bin_idx, int)
        assert 0 <= bin_idx < width
        assert isinstance(trnc, np.ndarray)
        assert trnc.ndim == 1
        # truncated response values between 0 and 1
        assert np.all(trnc >= 0)
        assert np.all(trnc <= 1)
        # Length consistency with bin_idx and width
        assert bin_idx + len(trnc) <= width


@pytest.mark.parametrize("analytic", [True, False])
def test_fbank_basic_properties_and_response(analytic):
    bank = Fbank(num_filts=5, analytic=analytic)
    assert bank.num_filts == 5
    assert bank.is_real == (not analytic)
    assert bank.is_analytic == analytic
    assert bank.is_zero_phase is True
    assert bank.sampling_rate == 16000
    # centers_hz increasing
    assert all(bank.centers_hz[i] < bank.centers_hz[i + 1] for i in range(bank.num_filts - 1))
    # get_impulse_response and frequency response shapes & types
    width = 64
    for filt_idx in range(bank.num_filts):
        impulse = bank.get_impulse_response(filt_idx, width)
        if analytic:
            assert np.iscomplexobj(impulse)
            assert impulse.shape == (width,)
        else:
            assert impulse.shape == (width,)
            assert impulse.dtype == np.float64
        freq_resp_half = bank.get_frequency_response(filt_idx, width, half=True)
        assert freq_resp_half.ndim == 1
        freq_resp_full = bank.get_frequency_response(filt_idx, width, half=False)
        assert freq_resp_full.ndim == 1
        bin_idx, truncated = bank.get_truncated_response(filt_idx, width)
        assert isinstance(bin_idx, int)
        assert truncated.ndim == 1


def test_fbank_invalid_freq_range():
    with pytest.raises(ValueError):
        _ = Fbank(low_hz=-1)
    with pytest.raises(ValueError):
        _ = Fbank(high_hz=8000, low_hz=9000)
    with pytest.raises(ValueError):
        _ = Fbank(high_hz=20000, sampling_rate=16000)


def test_gabor_filterbank_properties_and_response():
    bank = GaborFilterBank("mel", num_filts=3, scale_l2_norm=True, erb=True)
    assert bank.is_real is False
    # is_analytic depends on wrap_below, difficult to predict, just verify boolean
    assert isinstance(bank.is_analytic, bool)
    assert bank.is_zero_phase is True
    assert bank.num_filts == 3
    # centers_hz sorted ascending
    assert all(bank.centers_hz[i] < bank.centers_hz[i + 1] for i in range(bank.num_filts - 1))
    # supports_hz and supports tuple lengths
    assert len(bank.supports_hz) == bank.num_filts
    assert len(bank.supports) == bank.num_filts
    # impulse response shape and type
    width = 80
    for idx in range(bank.num_filts):
        impulse = bank.get_impulse_response(idx, width)
        assert impulse.shape == (width,)
        assert np.iscomplexobj(impulse)
    # frequency response shape and truncated response
    for half in [True, False]:
        for idx in range(bank.num_filts):
            freq_resp = bank.get_frequency_response(idx, width, half=half)
            expected_len = (width + 1) // 2 if half and (width % 2) else (width // 2 + 1 if half else width)
            assert freq_resp.shape == (expected_len,)
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        assert isinstance(bin_idx, int)
        # trnc is ndarray float64
        assert isinstance(trnc, np.ndarray)
        assert np.issubdtype(trnc.dtype, np.floating)


def test_complexgammatone_filterbank_invalid_order():
    with pytest.raises(ValueError):
        _ = ComplexGammatoneFilterBank("mel", order=0)
    with pytest.raises(ValueError):
        _ = ComplexGammatoneFilterBank("mel", order=-1)
    with pytest.raises(ValueError):
        _ = ComplexGammatoneFilterBank("mel", order=2.5)


def test_complexgammatone_filterbank_properties_and_responses():
    bank = ComplexGammatoneFilterBank("mel", num_filts=3, order=3, max_centered=True, scale_l2_norm=True, erb=True)
    assert bank.is_real is False
    assert isinstance(bank.is_analytic, bool)
    assert bank.is_zero_phase is False
    assert bank.order == 3
    width = 50
    for idx in range(bank.num_filts):
        impulse = bank.get_impulse_response(idx, width)
        assert impulse.shape == (width,)
        assert np.iscomplexobj(impulse)
        freq_resp_half = bank.get_frequency_response(idx, width, half=True)
        freq_resp_full = bank.get_frequency_response(idx, width, half=False)
        # frequency responses should be complex128
        assert freq_resp_half.dtype == np.complex128
        assert freq_resp_full.dtype == np.complex128
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        # bin_idx within valid range
        assert 0 <= bin_idx < width
        # truncated response is complex ndarray
        assert isinstance(trnc, np.ndarray)
        assert trnc.dtype == np.complex128


@pytest.mark.parametrize("width", [0, 1, 10])
def test_gammawindow_impulse_response_basic_behavior(width):
    from pydrobert.speech.filters import GammaWindow
    # Default order=4, peak=0.75
    win = GammaWindow()
    resp = win.get_impulse_response(width)
    if width <= 0:
        assert resp.size == 0
    elif width == 1:
        assert resp.size == 1
        assert resp[0] == 1.0
    else:
        # For larger widths, response is a 1-D array of length width
        assert resp.shape == (width,)
        # Values are non-negative floats
        assert np.all(resp >= 0)
        assert resp.dtype == float


@pytest.mark.parametrize("order,peak", [(1, 0.5), (3, 0.9), (5, 0.3)])
def test_gammawindow_impulse_response_order_and_peak(order, peak):
    from pydrobert.speech.filters import GammaWindow
    width = 50
    win = GammaWindow(order=order, peak=peak)
    resp = win.get_impulse_response(width)
    assert resp.shape == (width,)
    # The maximal value should be near (peak*width) index (within +/-5 samples)
    peak_sample = int(peak * width)
    max_idx = np.argmax(resp)
    assert abs(max_idx - peak_sample) <= 5


def test_triangular_filterbank_get_impulse_response_values_nonzero():
    bank = TriangularOverlappingFilterBank("mel", num_filts=2)
    width = 15
    resp = bank.get_impulse_response(1, width)
    # Check that at least some values are nonzero
    assert np.any(np.abs(resp) > 1e-10)


def test_fbank_get_frequency_response_nonzero_inside_support():
    bank = Fbank(num_filts=3)
    width = 64
    for filt_idx in range(bank.num_filts):
        left, right = bank.supports_hz[filt_idx]
        freq_resp = bank.get_frequency_response(filt_idx, width, half=True)
        # Convert left/right Hz to indices
        start_idx = int(np.ceil(width * left / bank.sampling_rate))
        end_idx = int(width * right / bank.sampling_rate)
        # At support edges frequency response should be near zero or zero
        if 0 <= start_idx < len(freq_resp):
            assert freq_resp[start_idx] < 1.0
        if 0 <= end_idx < len(freq_resp):
            assert freq_resp[end_idx] < 1.0
        # At mid center frequency, response should be maximum >= near 1
        center = bank.centers_hz[filt_idx]
        center_idx = int(width * center / bank.sampling_rate)
        if 0 <= center_idx < len(freq_resp):
            assert freq_resp[center_idx] > 0.5


def test_gabor_filterbank_supports_length_and_wrap_below_flag():
    bank = GaborFilterBank("mel", num_filts=5)
    assert len(bank.supports) == 5
    assert len(bank.supports_hz) == 5
    assert isinstance(bank._wrap_below, bool)


def test_complexgammatone_filterbank_internal_support_calculation_behavior():
    bank = ComplexGammatoneFilterBank("mel", num_filts=2, order=2)
    # _calculate_temp_support is private, but test indirectly by calls during init
    supports = bank.supports
    # Each support is tuple of length 2 with int entries
    for support in supports:
        assert isinstance(support, tuple)
        assert len(support) == 2
        assert all(isinstance(x, int) for x in support)


def test_triangular_filterbank_supports_ms_scaling_correctness():
    bank = TriangularOverlappingFilterBank("mel", num_filts=2)
    for (start_samples, end_samples), (start_ms, end_ms) in zip(bank.supports, bank.supports_ms):
        # ms should be correct conversion of samples
        assert math.isclose(start_ms, start_samples * 1000 / bank.sampling_rate)
        assert math.isclose(end_ms, end_samples * 1000 / bank.sampling_rate)
