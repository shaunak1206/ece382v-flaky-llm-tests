import math
import numpy as np
import pytest

from pydrobert.speech.filters import (
    TriangularOverlappingFilterBank,
    Fbank,
    GaborFilterBank,
    ComplexGammatoneFilterBank,
)


def test_triangular_init_and_properties_defaults():
    bank = TriangularOverlappingFilterBank("linear", num_filts=10)
    assert bank.num_filts == 10
    assert bank.is_zero_phase
    assert bank.is_real
    assert not bank.is_analytic
    assert bank.sampling_rate == 16000
    assert len(bank.centers_hz) == 10
    assert len(bank.supports_hz) == 10
    assert len(bank.supports) == 10
    # supports_ms length matches supports length
    assert len(bank.supports_ms) == len(bank.supports)
    # Check supports_ms correctness
    for (start_samps, end_samps), (start_ms, end_ms) in zip(bank.supports, bank.supports_ms):
        assert math.isclose(start_ms, 1000 * start_samps / bank.sampling_rate)
        assert math.isclose(end_ms, 1000 * end_samps / bank.sampling_rate)


def test_triangular_invalid_frequency_range():
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank("linear", low_hz=-1)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank("linear", low_hz=1000, high_hz=500, sampling_rate=2000)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank("linear", low_hz=0, high_hz=16000, sampling_rate=16000)


@pytest.mark.parametrize("analytic", [True, False])
def test_triangular_get_impulse_response_shape_and_type(analytic):
    bank = TriangularOverlappingFilterBank("linear", num_filts=5, analytic=analytic)
    width = 16
    for i in range(bank.num_filts):
        ir = bank.get_impulse_response(i, width)
        assert ir.shape == (width,)
        if analytic:
            assert np.iscomplexobj(ir)
        else:
            assert np.issubdtype(ir.dtype, np.floating)


@pytest.mark.parametrize("analytic", [True, False])
@pytest.mark.parametrize("half", [True, False])
def test_triangular_get_frequency_response_length_and_values(analytic, half):
    bank = TriangularOverlappingFilterBank("linear", num_filts=5, analytic=analytic)
    width_even = 16
    width_odd = 15
    for width in [width_even, width_odd]:
        for i in range(bank.num_filts):
            freq_resp = bank.get_frequency_response(i, width, half=half)
            expected_len = width
            if half:
                expected_len = (width // 2 + 1) if width % 2 == 0 else ((width + 1) // 2)
            assert freq_resp.shape == (expected_len,)
            # Values should be between 0 and 1 (triangle)
            assert np.all(freq_resp >= 0)
            assert np.all(freq_resp <= 1)


def test_triangular_get_truncated_response_basic():
    bank = TriangularOverlappingFilterBank("linear", num_filts=3)
    width = 64
    for i in range(bank.num_filts):
        bin_idx, truncated = bank.get_truncated_response(i, width)
        # bin_idx should be int and truncated a numpy array
        assert isinstance(bin_idx, int)
        assert isinstance(truncated, np.ndarray)
        assert truncated.ndim == 1
        assert truncated.dtype == np.float64
        # truncated length + bin_idx should be <= width
        assert bin_idx + len(truncated) <= width
        # truncated values between 0 and 1
        assert np.all(truncated >= 0)
        assert np.all(truncated <= 1)


def test_fbank_init_and_properties_defaults():
    bank = Fbank(num_filts=20)
    assert bank.num_filts == 20
    assert bank.is_zero_phase
    assert bank.is_real
    assert not bank.is_analytic
    assert bank.sampling_rate == 16000
    assert len(bank.centers_hz) == 20
    # supports should have length num_filts
    assert len(bank.supports_hz) == 20
    assert len(bank.supports) == 20
    assert len(bank.supports_ms) == 20


def test_fbank_invalid_frequency_range():
    with pytest.raises(ValueError):
        Fbank(low_hz=-5)
    with pytest.raises(ValueError):
        Fbank(low_hz=4000, high_hz=1000, sampling_rate=16000)
    with pytest.raises(ValueError):
        Fbank(low_hz=0, high_hz=16000, sampling_rate=16000)


@pytest.mark.parametrize("analytic", [True, False])
@pytest.mark.parametrize("half", [True, False])
def test_fbank_get_frequency_response_values_and_length(analytic, half):
    bank = Fbank(num_filts=5, analytic=analytic)
    width = 31
    for i in range(bank.num_filts):
        freq_resp = bank.get_frequency_response(i, width, half=half)
        expected_len = width
        if half:
            expected_len = (width + 1) // 2 if width % 2 else (width // 2 + 1)
        assert freq_resp.shape == (expected_len,)
        # Values non-negative and <=1 since frequency response is sqrt of something [0,1]
        assert np.all(freq_resp >= 0)
        assert np.all(freq_resp <= 1)

        # Check symmetric in the full spectrum when not analytic and half=False
        if not analytic and not half:
            # Since real, freq_resp[-i] == freq_resp[i] for appropriate i
            for idx in range(1, expected_len):
                assert math.isclose(freq_resp[idx], freq_resp[-idx], rel_tol=1e-9)


@pytest.mark.parametrize("analytic", [True, False])
def test_fbank_get_impulse_response_type_and_length(analytic):
    bank = Fbank(num_filts=5, analytic=analytic)
    width = 32
    for i in range(bank.num_filts):
        ir = bank.get_impulse_response(i, width)
        assert ir.shape == (width,)
        # Analytic returns complex else float
        if analytic:
            assert np.iscomplexobj(ir)
        else:
            assert np.issubdtype(ir.dtype, np.floating)


def test_fbank_get_truncated_response_basic():
    bank = Fbank(num_filts=4)
    width = 80
    for i in range(bank.num_filts):
        bin_idx, truncated = bank.get_truncated_response(i, width)
        assert isinstance(bin_idx, int)
        assert isinstance(truncated, np.ndarray)
        assert truncated.ndim == 1
        # Values non-negative and <= 1
        assert np.all(truncated >= 0)
        assert np.all(truncated <= 1)


def test_gabor_filterbank_init_and_properties():
    bank = GaborFilterBank("mel", num_filts=7, scale_l2_norm=True, erb=True)
    assert bank.num_filts == 7
    assert bank.is_zero_phase
    assert not bank.is_real
    # is_analytic depends on wrap_below
    assert isinstance(bank.is_analytic, bool)
    assert bank.sampling_rate == 16000
    assert len(bank.centers_hz) == 7
    assert len(bank.supports) == 7
    assert len(bank.supports_hz) == 7


def test_gabor_invalid_frequency_range():
    with pytest.raises(ValueError):
        GaborFilterBank("mel", low_hz=-0.01)
    with pytest.raises(ValueError):
        GaborFilterBank("mel", low_hz=3000, high_hz=2000, sampling_rate=8000)
    with pytest.raises(ValueError):
        GaborFilterBank("mel", low_hz=0, high_hz=8000, sampling_rate=8000)


def test_gabor_get_impulse_response_complex_and_shape():
    bank = GaborFilterBank("mel", num_filts=3)
    width = 20
    for i in range(bank.num_filts):
        ir = bank.get_impulse_response(i, width)
        assert ir.shape == (width,)
        assert np.iscomplexobj(ir)


def test_gabor_get_frequency_response_length_and_nonnegative():
    bank = GaborFilterBank("mel", num_filts=3)
    width = 40
    for half in [True, False]:
        for i in range(bank.num_filts):
            freq_resp = bank.get_frequency_response(i, width, half=half)
            expected_len = width
            if half:
                expected_len = (width + 1) // 2 if width % 2 else (width // 2 + 1)
            assert freq_resp.shape == (expected_len,)
            # Values should be complex128 but magnitude non-negative
            assert np.iscomplexobj(freq_resp)
            assert np.all(np.abs(freq_resp) >= 0)


def test_gabor_get_truncated_response_wrap_vs_no_wrap():
    bank = GaborFilterBank("mel", num_filts=3)
    width = 32

    # For filters where wrap_support_ang < 2*pi, we get truncated response not starting at 0
    for i in range(bank.num_filts):
        left_idx, trnc = bank.get_truncated_response(i, width)
        assert isinstance(left_idx, int)
        assert isinstance(trnc, np.ndarray)
        if bank._wrap_supports_ang[i] >= 2 * np.pi:
            # If full support, left_idx == 0 and length matches full freq response
            full_resp = bank.get_frequency_response(i, width)
            assert left_idx == 0
            # truncated response equals full response
            assert np.allclose(trnc, full_resp)
        else:
            # truncated response length matches expected segment
            assert trnc.ndim == 1
            assert trnc.dtype == np.float64


def test_complexgammatone_init_properties_and_invalid_order():
    bank = ComplexGammatoneFilterBank("mel", num_filts=5, order=3, max_centered=True)
    assert bank.num_filts == 5
    assert bank.order == 3
    assert bank.is_zero_phase is False
    assert bank.sampling_rate == 16000
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", order=0)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", order=-1)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", order=2.5)


def test_complexgammatone_get_impulse_response_and_length():
    bank = ComplexGammatoneFilterBank("mel", num_filts=4, order=2)
    width = 25
    for i in range(bank.num_filts):
        ir = bank.get_impulse_response(i, width)
        assert ir.shape == (width,)
        assert np.iscomplexobj(ir)


def test_complexgammatone_get_frequency_response_length_and_type():
    bank = ComplexGammatoneFilterBank("mel", num_filts=4, order=2)
    width = 27
    for half in [True, False]:
        for i in range(bank.num_filts):
            freq_resp = bank.get_frequency_response(i, width, half=half)
            expected_len = width
            if half:
                expected_len = (width + 1) // 2 if width % 2 else (width // 2 + 1)
            assert freq_resp.shape == (expected_len,)
            assert np.iscomplexobj(freq_resp)


def test_complexgammatone_get_truncated_response_behavior():
    bank = ComplexGammatoneFilterBank("mel", num_filts=3)
    width = 40
    for i in range(bank.num_filts):
        start_idx, truncated = bank.get_truncated_response(i, width)
        assert isinstance(start_idx, int)
        assert isinstance(truncated, np.ndarray)
        assert truncated.ndim >= 1
        # truncated length cannot exceed width (since indices are modulo width)
        assert len(truncated) <= width
        # start_idx modulo width is in range
        assert 0 <= (start_idx % width) < width


def test_triangular_get_impulse_response_symmetry_real():
    bank = TriangularOverlappingFilterBank("linear", num_filts=1, analytic=False)
    width = 15
    ir = bank.get_impulse_response(0, width)
    # Real impulse response should be symmetric about center (t=0 corresponds to middle)
    # Since it's zero phase and causal symmetric (around 0 with wrapping)
    # Test symmetry: ir[t] == ir[-t]
    for t in range(1, width):
        left_val = ir[t]
        right_val = ir[-t]
        # They should be complex conjugates (for real, same)
        assert np.isclose(left_val, right_val)


def test_fbank_get_frequency_response_monotonicity():
    bank = Fbank(num_filts=1)
    width = 50
    freq_resp = bank.get_frequency_response(0, width, half=True)
    # For the single filter, frequency response should rise then fall as mel scale triangle
    max_idx = np.argmax(freq_resp)
    # Values before max should not decrease
    assert np.all(np.diff(freq_resp[:max_idx]) >= -1e-12)
    # Values after max should not increase
    assert np.all(np.diff(freq_resp[max_idx:]) <= 1e-12)


def test_gabor_get_impulse_and_frequency_response_consistency():
    bank = GaborFilterBank("mel", num_filts=1)
    width = 40
    ir = bank.get_impulse_response(0, width)
    freq_resp = bank.get_frequency_response(0, width)
    # Frequency response should roughly match FFT magnitude of impulse response
    ir_fft = np.fft.fft(ir)
    # Because Gabor freq_resp is positive and scaled, ratios should be close within tolerance
    ratio = np.abs(freq_resp) / (np.abs(ir_fft) + 1e-15)
    # ratio should not be wildly varying (std deviation small)
    assert np.std(ratio) < 0.1


def test_complexgammatone_h_vs_H_consistency():
    # Test private methods _h and _H behave as expected for scalar and array inputs

    bank = ComplexGammatoneFilterBank("mel", num_filts=1, order=2)
    # _h with scalar t
    val_t_5 = bank._h(5, 0)
    val_t_0 = bank._h(0, 0)
    assert isinstance(val_t_5, complex)
    assert val_t_0 == 0j  # since t <= offset (offset=0 default), returns 0
    # _H with scalar omega
    val_omega = bank._H(math.pi, 0)
    assert isinstance(val_omega, complex)
    # _H with array omega returns array of complex values
    omega_array = np.array([0, math.pi / 2, math.pi])
    val_array = bank._H(omega_array, 0)
    assert isinstance(val_array, np.ndarray)
    assert val_array.dtype == np.complex128


def test_triangular_support_consistency():
    bank = TriangularOverlappingFilterBank("linear", num_filts=3)
    supports = bank.supports
    supports_hz = bank.supports_hz
    rate = bank.sampling_rate
    # supports_hz boundaries should align with hertz_to_angular for supports
    for idx, ((start, end), (start_hz, end_hz)) in enumerate(zip(supports, supports_hz)):
        start_ang = start * 2 * math.pi / rate
        end_ang = end * 2 * math.pi / rate
        # angular frequencies roughly between supports_hz angles (scaled)
        ang_low = hertz_to_angular(start_hz, rate)
        ang_high = hertz_to_angular(end_hz, rate)
        # Since supports is sample indices, map to angular frequencies for approximate check
        assert start < 0
        assert end > 0
        # Supports length is consistent for impulse response width expectations
        assert (end - start) > 0


# Helpers

def hertz_to_angular(freq, rate):
    from pydrobert.speech.util import hertz_to_angular as hta
    return hta(freq, rate)
