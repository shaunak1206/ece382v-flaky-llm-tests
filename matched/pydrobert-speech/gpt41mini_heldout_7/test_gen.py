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


@pytest.mark.parametrize("klass", [
    TriangularOverlappingFilterBank,
    Fbank,
    GaborFilterBank,
    ComplexGammatoneFilterBank,
])
def test_num_filts_and_centers_are_consistent(klass):
    bank = klass("mel", num_filts=10, sampling_rate=16000)
    assert bank.num_filts == 10
    assert isinstance(bank.centers_hz, tuple)
    assert len(bank.centers_hz) == bank.num_filts
    # centers are strictly increasing
    assert all(x < y for x, y in zip(bank.centers_hz, bank.centers_hz[1:]))


@pytest.mark.parametrize("klass", [
    TriangularOverlappingFilterBank,
    Fbank,
    GaborFilterBank,
    ComplexGammatoneFilterBank,
])
def test_sampling_rate_property(klass):
    sr = 22050
    bank = klass("mel", sampling_rate=sr)
    assert bank.sampling_rate == sr


def test_triangular_invalid_freq_range():
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank("mel", low_hz=-1)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank("mel", low_hz=5000, high_hz=4000)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank("mel", high_hz=20000, sampling_rate=32000)


def test_fbank_invalid_freq_range():
    with pytest.raises(ValueError):
        Fbank(low_hz=-5)
    with pytest.raises(ValueError):
        Fbank(low_hz=4000, high_hz=3999)
    with pytest.raises(ValueError):
        Fbank(high_hz=20001, sampling_rate=40000)


def test_gabor_invalid_freq_range():
    with pytest.raises(ValueError):
        GaborFilterBank("mel", low_hz=-0.1)
    with pytest.raises(ValueError):
        GaborFilterBank("mel", low_hz=8000, high_hz=4000)
    with pytest.raises(ValueError):
        GaborFilterBank("mel", high_hz=9000, sampling_rate=16000)


def test_gammatone_invalid_freq_range_and_order():
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", low_hz=-0.1)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", low_hz=8000, high_hz=4000)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", high_hz=9000, sampling_rate=16000)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", order=0)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", order=-5)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", order=2.5)


def test_triangular_get_impulse_response_shape_and_dtype():
    bank = TriangularOverlappingFilterBank("mel", num_filts=3)
    width = 50
    for idx in range(bank.num_filts):
        out = bank.get_impulse_response(idx, width)
        assert isinstance(out, np.ndarray)
        assert out.shape == (width,)
        if bank.is_analytic:
            assert np.iscomplexobj(out)
        else:
            assert np.issubdtype(out.dtype, np.floating)


def test_fbank_get_impulse_response_shape_and_dtype():
    bank = Fbank(num_filts=3)
    width = 64
    for idx in range(bank.num_filts):
        out = bank.get_impulse_response(idx, width)
        assert isinstance(out, np.ndarray)
        assert out.shape == (width,)
        assert np.issubdtype(out.dtype, np.floating) or np.iscomplexobj(out)


def test_gabor_get_impulse_response_shape_and_dtype():
    bank = GaborFilterBank("mel", num_filts=4)
    width = 128
    for idx in range(bank.num_filts):
        out = bank.get_impulse_response(idx, width)
        assert isinstance(out, np.ndarray)
        assert out.shape == (width,)
        # Gabor filters are complex
        assert np.iscomplexobj(out)


def test_gammatone_get_impulse_response_shape_and_dtype():
    bank = ComplexGammatoneFilterBank("mel", num_filts=4, order=3)
    width = 40
    for idx in range(bank.num_filts):
        out = bank.get_impulse_response(idx, width)
        assert isinstance(out, np.ndarray)
        assert out.shape == (width,)
        assert np.iscomplexobj(out)


@pytest.mark.parametrize("klass", [
    TriangularOverlappingFilterBank,
    Fbank,
    GaborFilterBank,
    ComplexGammatoneFilterBank,
])
def test_frequency_response_half_and_full_lengths(klass):
    bank = klass("mel", num_filts=5)
    width_even = 64
    width_odd = 65

    # test half=False returns length width
    for idx in range(bank.num_filts):
        freq_full = bank.get_frequency_response(idx, width_even, half=False)
        assert len(freq_full) == width_even

        freq_half = bank.get_frequency_response(idx, width_even, half=True)
        expected_half_len_even = width_even // 2 + 1
        assert len(freq_half) == expected_half_len_even

        freq_full_odd = bank.get_frequency_response(idx, width_odd, half=False)
        assert len(freq_full_odd) == width_odd

        freq_half_odd = bank.get_frequency_response(idx, width_odd, half=True)
        expected_half_len_odd = (width_odd + 1) // 2
        assert len(freq_half_odd) == expected_half_len_odd


@pytest.mark.parametrize("klass", [
    TriangularOverlappingFilterBank,
    Fbank,
    GaborFilterBank,
    ComplexGammatoneFilterBank,
])
def test_get_truncated_response_length_and_type(klass):
    bank = klass("mel", num_filts=5)
    width = 100
    for idx in range(bank.num_filts):
        bin_idx, buf = bank.get_truncated_response(idx, width)
        assert isinstance(bin_idx, int)
        assert isinstance(buf, np.ndarray)
        # The buffer length plus bin idx should be less or equal width for real filters;
        # For complex filters wrapping can occur. We test basic conditions only.
        assert buf.ndim == 1
        assert len(buf) > 0


def test_triangular_supports_hz_are_valid_intervals():
    bank = TriangularOverlappingFilterBank("mel", num_filts=6, sampling_rate=16000)
    for low, high in bank.supports_hz:
        assert low < high
        assert 0 <= low < bank.sampling_rate / 2
        assert 0 < high <= bank.sampling_rate / 2


def test_triangle_is_real_analytic_zero_phase_properties():
    bank_real = TriangularOverlappingFilterBank("mel", analytic=False)
    bank_analytic = TriangularOverlappingFilterBank("mel", analytic=True)
    assert bank_real.is_real
    assert not bank_real.is_analytic
    assert bank_real.is_zero_phase

    assert not bank_analytic.is_real
    assert bank_analytic.is_analytic
    assert bank_analytic.is_zero_phase


def test_gabor_properties():
    bank = GaborFilterBank("mel", num_filts=7, scale_l2_norm=True, erb=True)
    assert bank.is_real is False
    # is_analytic depends on _wrap_below, so just bool check
    assert isinstance(bank.is_analytic, bool)
    assert bank.is_zero_phase
    assert bank.scaled_l2_norm is True
    assert bank.erb is True


def test_complex_gammatone_properties_and_order():
    bank = ComplexGammatoneFilterBank("mel", num_filts=5, order=4,
                                      max_centered=True, scale_l2_norm=True, erb=True)
    assert bank.is_real is False
    assert isinstance(bank.is_analytic, bool)
    assert bank.is_zero_phase is False
    assert bank.order == 4
    assert bank.scaled_l2_norm is True
    assert bank.erb is True


def test_complex_gammatone_get_h_and_H_consistency():
    bank = ComplexGammatoneFilterBank("mel", num_filts=1, order=2)
    h_val = bank._h(1, 0)
    H_val = bank._H(np.array([0.5]), 0)
    # h_val should be complex
    assert isinstance(h_val, complex)
    assert isinstance(H_val, np.ndarray)
    assert np.iscomplexobj(H_val)
    # Check broadcasting and shape
    H_val_scalar = bank._H(0.5, 0)
    assert np.iscomplexobj(H_val_scalar)


def test_triangular_supports_property_matches_supports_ms():
    bank = TriangularOverlappingFilterBank("mel", num_filts=3, sampling_rate=16000)
    supports = bank.supports
    supports_ms = bank.supports_ms
    for (start, end), (start_ms, end_ms) in zip(supports, supports_ms):
        expected_start_ms = start * 1000 / bank.sampling_rate
        expected_end_ms = end * 1000 / bank.sampling_rate
        assert math.isclose(start_ms, expected_start_ms, rel_tol=1e-9)
        assert math.isclose(end_ms, expected_end_ms, rel_tol=1e-9)


def test_triangular_get_frequency_response_values_basic():
    # We'll test that frequency response values are in [0,1] and are zero outside support
    bank = TriangularOverlappingFilterBank("mel", num_filts=2, sampling_rate=16000)
    width = 128
    for idx in range(bank.num_filts):
        left, right = bank.supports_hz[idx]
        freq_resp = bank.get_frequency_response(idx, width)
        assert np.all(freq_resp >= 0)
        assert np.all(freq_resp <= 1)
        # Values at indices below left_idx should be zero or close
        left_idx = int(np.ceil(width * left / bank.sampling_rate))
        right_idx = int(width * right / bank.sampling_rate)
        before_left = freq_resp[:left_idx - 1] if left_idx > 1 else []
        after_right = freq_resp[right_idx + 1:] if right_idx + 1 < len(freq_resp) else []
        if len(before_left) > 0:
            assert np.allclose(before_left, 0, atol=1e-5)
        if len(after_right) > 0:
            assert np.allclose(after_right, 0, atol=1e-5)


def test_fbank_get_frequency_response_values_basic():
    # Similar test for Fbank
    bank = Fbank(num_filts=3, sampling_rate=16000)
    width = 128
    for idx in range(bank.num_filts):
        left, right = bank.supports_hz[idx]
        freq_resp = bank.get_frequency_response(idx, width)
        assert np.all(freq_resp >= 0)
        assert np.all(freq_resp <= 1)
        left_idx = int(np.ceil(width * left / bank.sampling_rate))
        right_idx = int(width * right / bank.sampling_rate)
        before_left = freq_resp[:left_idx - 1] if left_idx > 1 else []
        after_right = freq_resp[right_idx + 1:] if right_idx + 1 < len(freq_resp) else []
        if len(before_left) > 0:
            assert np.allclose(before_left, 0, atol=1e-5)
        if len(after_right) > 0:
            assert np.allclose(after_right, 0, atol=1e-5)


def test_get_truncated_response_reconstruction_triangular():
    bank = TriangularOverlappingFilterBank("mel", num_filts=3, sampling_rate=16000)
    width = 64
    for filt_idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
        full = np.zeros(width, dtype=trnc.dtype)
        wrap = min(bin_idx + len(trnc), width) - bin_idx
        full[bin_idx:bin_idx + wrap] = trnc[:wrap]
        if not bank.is_real:
            full[:len(trnc) - wrap] = trnc[wrap:]
        # The truncated version should be close to the full response in the support window
        freq_full = bank.get_frequency_response(filt_idx, width)
        # values within truncated region should be close
        left = bin_idx
        right = bin_idx + len(trnc)
        np.testing.assert_allclose(full[left:right], freq_full[left:right], atol=1e-5)


def test_get_truncated_response_reconstruction_fbank():
    bank = Fbank(num_filts=4)
    width = 64
    for filt_idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
        full = np.zeros(width, dtype=trnc.dtype)
        wrap = min(bin_idx + len(trnc), width) - bin_idx
        full[bin_idx:bin_idx + wrap] = trnc[:wrap]
        if not bank.is_real:
            full[:len(trnc) - wrap] = trnc[wrap:]
        freq_full = bank.get_frequency_response(filt_idx, width)
        left = bin_idx
        right = bin_idx + len(trnc)
        np.testing.assert_allclose(full[left:right], freq_full[left:right], atol=1e-5)
