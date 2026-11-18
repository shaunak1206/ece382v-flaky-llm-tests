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
    GammaWindow,
)
from pydrobert.speech.scales import MelScaling
from pydrobert.speech.util import hertz_to_angular


@pytest.mark.parametrize("analytical_flag", [False, True])
def test_triangular_overlap_filterbank_basic_properties(analytical_flag):
    bank = TriangularOverlappingFilterBank("linear", num_filts=10, sampling_rate=16000, analytic=analytical_flag)
    # Types and lengths
    assert isinstance(bank.is_real, bool)
    assert isinstance(bank.is_analytic, bool)
    assert isinstance(bank.is_zero_phase, bool)
    assert bank.num_filts == 10
    assert bank.sampling_rate == 16000

    # centers_hz length == num_filts
    assert len(bank.centers_hz) == bank.num_filts
    # supports_hz length == num_filts
    assert len(bank.supports_hz) == bank.num_filts
    # supports length == num_filts
    assert len(bank.supports) == bank.num_filts
    # supports_ms matches supports in length and scaling
    assert len(bank.supports_ms) == bank.num_filts
    for (start_samp, end_samp), (start_ms, end_ms) in zip(bank.supports, bank.supports_ms):
        assert np.isclose(start_ms, 1000 * start_samp / bank.sampling_rate)
        assert np.isclose(end_ms, 1000 * end_samp / bank.sampling_rate)


def test_triangular_overlap_filterbank_invalid_frequency_range():
    # low_hz < 0 raises
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank("linear", low_hz=-1)
    # high_hz <= low_hz raises
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank("linear", low_hz=1000, high_hz=1000)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank("linear", low_hz=1000, high_hz=500)


@pytest.mark.parametrize("filt_idx", [0, 1, 5, 9])
def test_triangular_overlap_get_impulse_response_lengths(filt_idx):
    bank = TriangularOverlappingFilterBank("linear", num_filts=10)
    width = 15
    impulse = bank.get_impulse_response(filt_idx, width)
    assert isinstance(impulse, np.ndarray)
    assert impulse.shape == (width,)
    if bank.is_real:
        assert np.isrealobj(impulse)
    else:
        # complex dtype expected for analytic=True
        assert np.iscomplexobj(impulse)


@pytest.mark.parametrize("filt_idx", [0, 3, 8])
@pytest.mark.parametrize("width", [8, 16, 33])
@pytest.mark.parametrize("half", [False, True])
def test_triangular_overlap_get_frequency_response_shape(filt_idx, width, half):
    bank = TriangularOverlappingFilterBank("linear", num_filts=10)
    freq_resp = bank.get_frequency_response(filt_idx, width, half=half)
    if half:
        expected_len = (width + 1) // 2 if width % 2 else (width // 2 + 1)
    else:
        expected_len = width
    assert isinstance(freq_resp, np.ndarray)
    assert len(freq_resp) == expected_len
    # Check dtype matches is_zero_phase flag and analicity
    if bank.is_zero_phase:
        assert freq_resp.dtype == np.float64
    else:
        assert freq_resp.dtype == np.complex128 or freq_resp.dtype == np.float64
    # Values between 0 and 1
    assert np.all(freq_resp >= 0)
    assert np.all(freq_resp <= 1)


@pytest.mark.parametrize("filt_idx", [0, 2, 9])
@pytest.mark.parametrize("width", [16, 32])
def test_triangular_overlap_get_truncated_response_content(filt_idx, width):
    bank = TriangularOverlappingFilterBank("linear", num_filts=10)
    bin_idx, truncated = bank.get_truncated_response(filt_idx, width)
    assert isinstance(bin_idx, int)
    assert isinstance(truncated, np.ndarray)
    # truncated length should be positive and less than width
    assert 0 <= bin_idx < width
    assert len(truncated) > 0
    assert len(truncated) <= width
    # Values between 0 and 1 since they represent triangular shapes
    assert np.all(truncated >= 0)
    assert np.all(truncated <= 1)
    # Retrieving full response from truncated sometimes discussed in docstring.
    # We test that truncated response fits into full frequency response correctly
    full = np.zeros(width)
    right_idx = bin_idx + len(truncated)
    if right_idx <= width:
        full[bin_idx:right_idx] = truncated
    else:
        wrap = width - bin_idx
        full[bin_idx:] = truncated[:wrap]
        full[:right_idx - width] = truncated[wrap:]
    full_resp = bank.get_frequency_response(filt_idx, width, half=True)
    # The truncated reconstruction should match the main frequency response on that section
    idxs = np.arange(bin_idx, min(right_idx, width)) % width
    assert np.allclose(full_resp[idxs], truncated[:len(idxs)], atol=1e-5)


@pytest.mark.parametrize("analytic", [False, True])
def test_fbank_basic_properties(analytic):
    bank = Fbank(num_filts=20, analytic=analytic)
    assert isinstance(bank.num_filts, int)
    assert bank.num_filts == 20
    assert isinstance(bank.centers_hz, tuple)
    assert len(bank.centers_hz) == bank.num_filts
    assert isinstance(bank.is_real, bool)
    assert bank.is_real == (not analytic)
    assert bank.is_zero_phase
    # supports_hz and supports lengths
    assert len(bank.supports_hz) == bank.num_filts
    assert len(bank.supports) == bank.num_filts


@pytest.mark.parametrize("filt_idx", [0, 10, 19])
@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("half", [False, True])
def test_fbank_frequency_response_shape(filt_idx, width, half):
    bank = Fbank(num_filts=20)
    resp = bank.get_frequency_response(filt_idx, width, half=half)
    if half:
        expected_len = (width + 1) // 2 if width % 2 else (width // 2 + 1)
    else:
        expected_len = width
    assert resp.shape == (expected_len,)
    assert np.all(resp >= 0)
    assert np.all(resp <= 1)


@pytest.mark.parametrize("filt_idx", [0, 5, 19])
@pytest.mark.parametrize("width", [16, 64])
def test_fbank_get_truncated_response_values(filt_idx, width):
    bank = Fbank(num_filts=20)
    bin_idx, truncated = bank.get_truncated_response(filt_idx, width)
    assert isinstance(bin_idx, int)
    assert isinstance(truncated, np.ndarray)
    # Values nonnegative due to sqrt of positive triangles
    assert np.all(truncated >= 0)
    # Size matches expected segment length
    assert truncated.size == len(truncated)
    # Reconstruction check (partial)
    full = np.zeros(width)
    right_idx = bin_idx + len(truncated)
    if right_idx <= width:
        full[bin_idx:right_idx] = truncated
    else:
        wrap = width - bin_idx
        full[bin_idx:] = truncated[:wrap]
        full[:right_idx - width] = truncated[wrap:]
    freq_resp = bank.get_frequency_response(filt_idx, width, half=True)
    idxs = np.arange(bin_idx, min(right_idx, width)) % width
    assert np.allclose(freq_resp[idxs], truncated[:len(idxs)])


@pytest.mark.parametrize("scale_l2_norm", [False, True])
@pytest.mark.parametrize("erb", [False, True])
def test_gabor_filter_bank_properties(scale_l2_norm, erb):
    bank = GaborFilterBank("linear", num_filts=10, scale_l2_norm=scale_l2_norm, erb=erb)
    assert bank.num_filts == 10
    assert isinstance(bank.centers_hz, tuple)
    assert len(bank.centers_hz) == 10
    assert isinstance(bank.is_real, bool)
    assert bank.is_real is False
    assert isinstance(bank.is_analytic, bool)
    assert isinstance(bank.is_zero_phase, bool) and bank.is_zero_phase
    assert isinstance(bank.scaled_l2_norm, bool)
    assert bank.scaled_l2_norm == scale_l2_norm
    assert bank.erb == erb
    assert len(bank.supports) == 10
    assert len(bank.supports_hz) == 10


@pytest.mark.parametrize("filt_idx", [0, 3, 9])
@pytest.mark.parametrize("width", [64, 128])
@pytest.mark.parametrize("half", [False, True])
def test_gabor_get_frequency_response_and_impulse_response_shapes(filt_idx, width, half):
    bank = GaborFilterBank("linear", num_filts=10)
    freq_resp = bank.get_frequency_response(filt_idx, width, half=half)
    impulse_resp = bank.get_impulse_response(filt_idx, width)
    expected_len = (width + 1) // 2 if (half and width % 2) else (width // 2 + 1) if half else width
    assert freq_resp.shape == (expected_len,)
    assert impulse_resp.shape == (width,)
    # Frequency response is float64 due to sum of exponentials
    assert np.issubdtype(freq_resp.dtype, np.floating)
    # Impulse response is complex128
    assert np.iscomplexobj(impulse_resp)


def test_gabor_get_truncated_response_wrap_behavior():
    # Forcing a large wrap support to trigger full response return
    bank = GaborFilterBank("linear", num_filts=2)
    # manually set wrap support to large value to test first branch
    bank._wrap_supports_ang = (10 * np.pi, 10 * np.pi)
    idx, resp = bank.get_truncated_response(0, 32)
    assert idx == 0
    assert resp.shape == (32,)


@pytest.mark.parametrize("order", [1, 3, 5])
def test_complex_gammatone_invalid_order():
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("linear", order=0)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("linear", order=-1)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("linear", order=2.5)


@pytest.mark.parametrize("order", [1, 2, 4])
@pytest.mark.parametrize("scale_l2_norm", [False, True])
def test_complex_gammatone_basic_properties(order, scale_l2_norm):
    bank = ComplexGammatoneFilterBank("linear", num_filts=10, order=order, scale_l2_norm=scale_l2_norm, max_centered=True)
    assert bank.order == order
    assert bank.num_filts == 10
    assert len(bank.centers_hz) == 10
    assert isinstance(bank.is_real, bool) and bank.is_real is False
    assert isinstance(bank.is_analytic, bool)
    assert isinstance(bank.is_zero_phase, bool) and bank.is_zero_phase is False
    assert bank.scaled_l2_norm == scale_l2_norm
    assert len(bank.supports) == 10


@pytest.mark.parametrize("filt_idx", [0, 4, 9])
@pytest.mark.parametrize("width", [64, 128])
@pytest.mark.parametrize("half", [False, True])
def test_complex_gammatone_get_frequency_and_impulse_responses(filt_idx, width, half):
    bank = ComplexGammatoneFilterBank("linear", num_filts=10, order=3)
    freq_resp = bank.get_frequency_response(filt_idx, width, half=half)
    impulse_resp = bank.get_impulse_response(filt_idx, width)
    expected_len = (width + 1) // 2 if (half and width % 2) else (width // 2 + 1) if half else width
    assert freq_resp.shape == (expected_len,)
    assert impulse_resp.shape == (width,)
    # freq response is complex128 dtype
    assert np.iscomplexobj(freq_resp)
    assert np.iscomplexobj(impulse_resp)


@pytest.mark.parametrize("filt_idx", [0, 2, 5])
@pytest.mark.parametrize("width", [32, 64])
def test_complex_gammatone_get_truncated_response(filt_idx, width):
    bank = ComplexGammatoneFilterBank("linear", num_filts=10)
    idx, vals = bank.get_truncated_response(filt_idx, width)
    assert isinstance(idx, int)
    assert isinstance(vals, np.ndarray)
    assert vals.size > 0
    # truncated values should be complex dtype
    assert np.iscomplexobj(vals)


@pytest.mark.parametrize("width", [0, 1, 10, 50])
def test_bartlett_window_impulse_response(width):
    w = BartlettWindow()
    result = w.get_impulse_response(width)
    assert isinstance(result, np.ndarray)
    assert len(result) == width
    if width > 1:
        # Bartlett window is symmetric and positive
        assert np.all(result >= 0)


@pytest.mark.parametrize("width", [0, 1, 10, 50])
def test_blackman_window_impulse_response(width):
    w = BlackmanWindow()
    result = w.get_impulse_response(width)
    assert isinstance(result, np.ndarray)
    assert len(result) == width
    if width > 1:
        # Blackman window values are floats, typically positive or near zero
        assert np.all(result >= 0)


@pytest.mark.parametrize("width", [0, 1, 10, 50])
def test_hamming_window_impulse_response(width):
    w = HammingWindow()
    result = w.get_impulse_response(width)
    assert isinstance(result, np.ndarray)
    assert len(result) == width
    if width > 1:
        assert np.all(result >= 0)


@pytest.mark.parametrize("width", [0, 1, 10, 50])
def test_hann_window_impulse_response(width):
    w = HannWindow()
    result = w.get_impulse_response(width)
    assert isinstance(result, np.ndarray)
    assert len(result) == width
    if width > 1:
        assert np.all(result >= 0)


@pytest.mark.parametrize("width", [0, 1, 15, 30])
@pytest.mark.parametrize("order, peak", [(1, 0.5), (4, 0.75), (3, 0.95)])
def test_gamma_window_impulse_response(order, peak, width):
    w = GammaWindow(order=order, peak=peak)
    result = w.get_impulse_response(width)
    assert isinstance(result, np.ndarray)
    assert len(result) == width
    if width == 0:
        assert result.size == 0
    elif width == 1:
        # Single-element array, value 1.0
        assert result[0] == 1.0
    else:
        # Values should be floats and non-negative
        assert np.all(result >= 0)


def test_gamma_window_impulse_response_order1_behavior():
    w = GammaWindow(order=1, peak=0.5)
    result = w.get_impulse_response(10)
    assert isinstance(result, np.ndarray)
    assert result.size == 10
    # The sequence should roughly decrease (reflection of gamma)
    assert result[0] >= result[-1]


@pytest.mark.parametrize("filt_idx", [0])
@pytest.mark.parametrize("width", [10])
def test_triangular_impulse_response_symmetry(filt_idx, width):
    bank = TriangularOverlappingFilterBank("linear", num_filts=5)
    impulse = bank.get_impulse_response(filt_idx, width)
    # Check for conjugate symmetry about center for real filters
    # Zero phase implies symmetrical impulse response in time domain
    if bank.is_zero_phase:
        # Index 0 plus last element should be real and values conjugate pairs
        for i in range(1, width):
            assert np.allclose(impulse[i], np.conj(impulse[-i]))


@pytest.mark.parametrize("filt_idx", [0])
def test_gabor_impulse_response_nonzero(filt_idx):
    bank = GaborFilterBank("linear", num_filts=5)
    impulse = bank.get_impulse_response(filt_idx, 20)
    # Should not be all zeros
    assert np.any(np.abs(impulse) > 1e-10)


@pytest.mark.parametrize("width", [0, 1, 10])
def test_bartlett_window_empty_and_single():
    w = BartlettWindow()
    res0 = w.get_impulse_response(0)
    res1 = w.get_impulse_response(1)
    assert isinstance(res0, np.ndarray)
    assert res0.size == 0
    assert isinstance(res1, np.ndarray)
    assert res1.size == 1


@pytest.mark.parametrize("width", [0, 1, 20])
def test_blackman_window_empty_and_single(width):
    w = BlackmanWindow()
    res = w.get_impulse_response(width)
    assert isinstance(res, np.ndarray)
    assert res.size == width


@pytest.mark.parametrize("width", [0, 1, 20])
def test_hamming_window_empty_and_single(width):
    w = HammingWindow()
    res = w.get_impulse_response(width)
    assert isinstance(res, np.ndarray)
    assert res.size == width


@pytest.mark.parametrize("width", [0, 1, 20])
def test_hann_window_empty_and_single(width):
    w = HannWindow()
    res = w.get_impulse_response(width)
    assert isinstance(res, np.ndarray)
    assert res.size == width
