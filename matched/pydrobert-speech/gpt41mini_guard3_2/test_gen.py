import numpy as np
import pytest
import random

from pydrobert.speech.filters import (
    TriangularOverlappingFilterBank,
    Fbank,
    GaborFilterBank,
    ComplexGammatoneFilterBank,
)


def approx_equal(a, b, tol=1e-7):
    """Helper: assert arrays a and b are almost equal within tol."""
    assert a.shape == b.shape
    assert np.allclose(a, b, atol=tol, rtol=tol)


def test_triangular_init_valid_ranges():
    # seed numpy to guard against any internal randomness (likely unused here)
    np.random.seed(123)
    random.seed(123)
    # Normal range
    bank = TriangularOverlappingFilterBank("mel", num_filts=5, low_hz=20.0, high_hz=8000.0, sampling_rate=16000)
    assert bank.num_filts == 5
    assert bank.sampling_rate == 16000
    assert bank.is_zero_phase is True
    # high_hz None defaults to Nyquist
    bank2 = TriangularOverlappingFilterBank("mel", num_filts=5, low_hz=10.0, high_hz=None, sampling_rate=16000)
    assert bank2.sampling_rate == 16000
    assert bank2._vertices[-1] == 16000 // 2
    # analytic True sets is_analytic True and is_real False
    bank3 = TriangularOverlappingFilterBank("mel", analytic=True)
    assert bank3.is_analytic is True
    assert bank3.is_real is False


@pytest.mark.parametrize(
    "low_hz, high_hz",
    [
        (-1, 8000),           # low_hz < 0 invalid
        (100, 50),            # high_hz <= low_hz invalid
        (100, 16000),         # high_hz > Nyquist invalid (for 16000sr Nyquist=8000)
    ],
)
def test_triangular_init_invalid_ranges(low_hz, high_hz):
    np.random.seed(123)
    random.seed(123)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank("mel", low_hz=low_hz, high_hz=high_hz, sampling_rate=16000)


def test_triangular_supports_properties_consistency():
    np.random.seed(42)
    random.seed(42)
    bank = TriangularOverlappingFilterBank("mel", num_filts=3, low_hz=20., high_hz=1000., sampling_rate=4000)
    assert isinstance(bank.supports_hz, tuple)
    assert len(bank.supports_hz) == bank.num_filts
    for low, high in bank.supports_hz:
        assert low <= high
        assert low >= 0
        assert high <= bank.sampling_rate / 2
    assert len(bank.supports) == bank.num_filts
    for s in bank.supports:
        assert isinstance(s, tuple)
        assert len(s) == 2
        assert s[0] < s[1]
    # supports_ms matches supports scaled by 1000/sampling_rate
    supports_ms_calc = tuple((
        s[0] * 1000 / bank.sampling_rate,
        s[1] * 1000 / bank.sampling_rate,
    ) for s in bank.supports)
    for a, b in zip(bank.supports_ms, supports_ms_calc):
        approx_equal(np.array(a), np.array(b))


@pytest.mark.parametrize("width", [5, 10, 15])
def test_triangular_get_impulse_response_symmetry(width):
    np.random.seed(123)
    random.seed(123)
    bank = TriangularOverlappingFilterBank("mel", num_filts=2, low_hz=20., high_hz=8000., sampling_rate=16000, analytic=False)
    for idx in range(bank.num_filts):
        h = bank.get_impulse_response(idx, width)
        # Impulse response length equals width
        assert isinstance(h, np.ndarray)
        assert h.shape == (width,)
        # type is float64 when analytic False
        assert h.dtype == np.float64
        # Since zero phase, impulse response should be Hermitian symmetric: h[t] == h[-t]
        for t in range(1, width):
            assert np.isclose(h[t], h[-t], atol=1e-7)
        # Value at 0 is real
        assert np.isclose(h[0].imag, 0., atol=1e-10)
    # analytic True returns complex128 dtype output
    bank_a = TriangularOverlappingFilterBank("mel", num_filts=2, analytic=True)
    for idx in range(bank_a.num_filts):
        ha = bank_a.get_impulse_response(idx, width)
        assert ha.dtype == np.complex128
        assert ha.shape == (width,)


@pytest.mark.parametrize("width", [4, 6, 7])
@pytest.mark.parametrize("half", [False, True])
def test_triangular_get_frequency_response_properties(width, half):
    np.random.seed(50)
    random.seed(50)
    bank = TriangularOverlappingFilterBank("mel", num_filts=3, low_hz=20., high_hz=8000., sampling_rate=16000, analytic=False)
    for idx in range(bank.num_filts):
        freq_resp = bank.get_frequency_response(idx, width, half=half)
        # dtype float64 expected
        assert freq_resp.dtype == np.float64
        if half:
            expected_len = width // 2 + 1 if width % 2 == 0 else (width + 1) // 2
        else:
            expected_len = width
        assert freq_resp.shape == (expected_len,)
        # Values should be all >= 0 and <= 1 (triangular)
        assert np.all(freq_resp >= 0)
        assert np.all(freq_resp <= 1)
        # If not analytic and full spectrum, symmetric values at mirrored bins
        if not half and not bank.is_analytic:
            for i in range(1, expected_len):
                # freq_resp[i] == freq_resp[-i]
                assert np.isclose(freq_resp[i], freq_resp[-i], atol=1e-7)


@pytest.mark.parametrize("width", [8, 16, 32])
def test_triangular_get_truncated_response_valid(width):
    np.random.seed(10)
    random.seed(10)
    bank = TriangularOverlappingFilterBank("mel", num_filts=4, low_hz=20, high_hz=8000, sampling_rate=16000, analytic=False)
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        assert isinstance(bin_idx, int)
        assert isinstance(trnc, np.ndarray)
        # truncated response length > 0
        assert trnc.size > 0
        # All values in truncated response are between 0 and 1
        assert np.all(trnc >= 0)
        assert np.all(trnc <= 1)
        # bin_idx within valid range for width, but can be zero or more
        assert 0 <= bin_idx < width


def test_fbank_properties_and_methods_consistency():
    np.random.seed(7)
    random.seed(7)
    bank = Fbank(num_filts=5, low_hz=20., high_hz=8000., sampling_rate=16000, analytic=False)
    assert bank.num_filts == 5
    # centers in Hz count equals num_filts
    assert len(bank.centers_hz) == bank.num_filts
    # supports_hz same length as num_filts
    assert len(bank.supports_hz) == bank.num_filts
    # supports boundaries valid: low < high and within range
    for low, high in bank.supports_hz:
        assert 0 <= low <= high <= bank.sampling_rate / 2
    # supports: length num_filts, tuples of two ints
    supports = bank.supports
    assert len(supports) == bank.num_filts
    for s in supports:
        assert isinstance(s, tuple)
        assert len(s) == 2
    # impulse response shape and dtype tests
    width = 20
    for idx in range(bank.num_filts):
        imp = bank.get_impulse_response(idx, width)
        # fbank analytic False produces real output of length width
        assert imp.shape == (width,)
        assert np.issubdtype(imp.dtype, np.floating)
        # frequency response returns float64 real values, half True or False
        freq_full = bank.get_frequency_response(idx, width, half=False)
        freq_half = bank.get_frequency_response(idx, width, half=True)
        assert freq_full.dtype == np.float64
        assert freq_half.dtype == np.float64
        expected_len_full = width
        expected_len_half = width // 2 + 1 if width % 2 == 0 else (width + 1) // 2
        assert freq_full.shape == (expected_len_full,)
        assert freq_half.shape == (expected_len_half,)
        # truncated response returns valid index and array, values positive and <=1
        bin_idx, truncated = bank.get_truncated_response(idx, width)
        assert isinstance(bin_idx, int)
        assert isinstance(truncated, np.ndarray)
        assert truncated.size > 0
        assert np.all(truncated >= 0)
        assert np.all(truncated <= 1)


def test_gabor_filterbank_basic_properties():
    # seed numpy and random as required
    np.random.seed(2023)
    random.seed(2023)
    bank = GaborFilterBank("erb", num_filts=4, low_hz=20., high_hz=8000., sampling_rate=16000,
                          scale_l2_norm=False, erb=True)

    assert bank.num_filts == 4
    assert bank.is_real is False
    # is_analytic depends on internal wrap_below flag, bool
    assert isinstance(bank.is_analytic, bool)
    assert bank.is_zero_phase is True
    # centers_hz length matches num_filts
    assert len(bank.centers_hz) == bank.num_filts
    # supports_hz and supports length match num_filts
    assert len(bank.supports_hz) == bank.num_filts
    assert len(bank.supports) == bank.num_filts
    # values inside supports are float or int tuples (2 items)
    for low, high in bank.supports_hz:
        assert low <= high
        assert low >= 0
    for s in bank.supports:
        assert isinstance(s, tuple)
        assert len(s) == 2
    # scaled_l2_norm and erb properties matching constructor params
    assert bank.scaled_l2_norm is False
    assert bank.erb is True


def test_gabor_get_impulse_response_values_and_type():
    np.random.seed(100)
    random.seed(100)
    bank = GaborFilterBank("erb", num_filts=2, low_hz=20., high_hz=8000., sampling_rate=16000)
    width = 16
    for idx in range(bank.num_filts):
        imp = bank.get_impulse_response(idx, width)
        assert isinstance(imp, np.ndarray)
        assert imp.shape == (width,)
        assert np.issubdtype(imp.dtype, np.complexfloating)
        # Values should not be all zero
        assert np.any(np.abs(imp) > 0)
        # Values should be finite
        assert np.all(np.isfinite(imp))


@pytest.mark.parametrize("width", [16, 32])
@pytest.mark.parametrize("half", [True, False])
def test_gabor_get_frequency_response_values(width, half):
    np.random.seed(42)
    random.seed(42)
    bank = GaborFilterBank("erb", num_filts=3, low_hz=20., high_hz=8000., sampling_rate=16000)
    for idx in range(bank.num_filts):
        resp = bank.get_frequency_response(idx, width, half=half)
        # Returned array shape correct (half or full)
        expected_len = width
        if half:
            expected_len = width // 2 + 1 if width % 2 == 0 else (width + 1) // 2
        assert resp.shape == (expected_len,)
        # dtype complex128 expected
        assert np.issubdtype(resp.dtype, np.complexfloating)
        # Not all values are zero
        assert np.any(np.abs(resp) > 0)
        # finite values
        assert np.all(np.isfinite(resp))


def test_gabor_get_truncated_response_and_wrap(monkeypatch):
    np.random.seed(123)
    random.seed(123)
    bank = GaborFilterBank("erb", num_filts=2, low_hz=20., high_hz=8000., sampling_rate=16000)
    width = 32

    # patch _wrap_supports_ang to test both branches
    monkeypatch.setattr(bank, "_wrap_supports_ang", (4.0, 10.0))  # first < 2pi, second > 2pi approx 6.283

    # For first filter: wrap_supports_ang < 2pi follows normal truncated response
    left_idx, trnc = bank.get_truncated_response(0, width)
    assert isinstance(left_idx, int)
    assert isinstance(trnc, np.ndarray)
    assert trnc.size > 0
    assert np.all(np.isfinite(trnc))

    # For second filter: forcibly exceed 2pi triggers full frequency response return
    # monkeypatch _wrap_supports_ang element to > 2pi triggers alternative branch.
    # Override to larger than 2*pi ( ~6.283)
    monkeypatch.setattr(bank._wrap_supports_ang, 1, 10.0)
    # Actually, cannot patch tuple elements, recreate tuple attribute:
    new_wraps = (bank._wrap_supports_ang[0], 10.0)
    monkeypatch.setattr(bank, "_wrap_supports_ang", new_wraps)
    left_idx2, trnc2 = bank.get_truncated_response(1, width)
    assert left_idx2 == 0
    assert trnc2.shape == (width,)
    # Should be equal to get_frequency_response for that index
    full_resp = bank.get_frequency_response(1, width)
    approx_equal(trnc2, full_resp)


def test_complex_gammatone_init_invalid_order():
    np.random.seed(1)
    random.seed(1)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", order=0)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", order=-3)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", order="four")


def test_complex_gammatone_properties_and_methods():
    np.random.seed(24)
    random.seed(24)
    bank = ComplexGammatoneFilterBank("mel", num_filts=3, low_hz=20., high_hz=8000., sampling_rate=16000, order=3)
    assert bank.num_filts == 3
    assert bank.order == 3
    assert bank.is_real is False
    assert bank.is_zero_phase is False
    assert isinstance(bank.is_analytic, bool)
    assert len(bank.centers_hz) == bank.num_filts
    assert len(bank.supports) == bank.num_filts
    assert len(bank.supports_hz) == bank.num_filts

    width = 16
    for idx in range(bank.num_filts):
        # get_impulse_response returns complex128 array
        imp = bank.get_impulse_response(idx, width)
        assert isinstance(imp, np.ndarray)
        assert imp.shape == (width,)
        assert np.issubdtype(imp.dtype, np.complexfloating)
        # values finite and nonzero
        assert np.all(np.isfinite(imp))
        assert np.any(np.abs(imp) > 0)

        # frequency response returns complex128 array, shape depends on half
        freq_resp = bank.get_frequency_response(idx, width, half=False)
        assert freq_resp.shape == (width,)
        freq_resp_h = bank.get_frequency_response(idx, width, half=True)
        expected_len_h = width // 2 + 1 if width % 2 == 0 else (width + 1) // 2
        assert freq_resp_h.shape == (expected_len_h,)
        assert np.issubdtype(freq_resp.dtype, np.complexfloating)
        assert np.issubdtype(freq_resp_h.dtype, np.complexfloating)

        # truncated response returns int and np.ndarray of floats
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        # Because _H returns complex, expect complex result here too
        assert isinstance(bin_idx, int)
        assert isinstance(trnc, np.ndarray)
        assert trnc.size > 0
        assert np.all(np.isfinite(trnc))


def test_complex_gammatone__h_and__H_consistency():
    np.random.seed(11)
    random.seed(11)
    bank = ComplexGammatoneFilterBank("mel", num_filts=2, low_hz=20., high_hz=8000., sampling_rate=16000, order=2)
    # Test internal _h method against known properties
    t = 1.5
    idx = 0
    val = bank._h(t, idx)
    assert isinstance(val, complex)
    # For t <= offset, method returns 0j
    offset = bank._offsets[idx]
    assert bank._h(offset - 0.5, idx) == 0j
    # _H returns complex value for scalar omega, vector omega
    omega_scalar = np.pi / 2
    resp_scalar = bank._H(omega_scalar, idx)
    assert isinstance(resp_scalar, complex)
    omega_vec = np.linspace(0, 2 * np.pi, 5)
    resp_vec = bank._H(omega_vec, idx)
    assert isinstance(resp_vec, np.ndarray)
    assert resp_vec.shape == omega_vec.shape
    # Values finite and nonzero
    assert np.all(np.isfinite(resp_vec))
    assert np.any(np.abs(resp_vec) > 0)

