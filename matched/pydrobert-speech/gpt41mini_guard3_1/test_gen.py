import numpy as np
import pytest
import random

from pydrobert.speech.filters import (
    TriangularOverlappingFilterBank,
    Fbank,
    GaborFilterBank,
    ComplexGammatoneFilterBank,
)


# Common parameters for testing
NUM_FILTS = 10
SAMPLING_RATE = 8000
LOW_HZ = 50.0
HIGH_HZ = 3500.0
WIDTH = 256


def seed_all():
    """Seed all RNGs that could be used by the code under test."""
    random.seed(1234)
    np.random.seed(1234)


@pytest.mark.parametrize("analytic", [False, True])
def test_triangular_filter_bank_basic_properties(analytic):
    seed_all()
    bank = TriangularOverlappingFilterBank(
        scaling_function="mel", num_filts=NUM_FILTS,
        high_hz=HIGH_HZ, low_hz=LOW_HZ,
        sampling_rate=SAMPLING_RATE, analytic=analytic,
    )

    # num_filts
    assert bank.num_filts == NUM_FILTS
    # sampling rate
    assert bank.sampling_rate == SAMPLING_RATE
    # vertices define centers_hz
    assert len(bank.centers_hz) == NUM_FILTS
    # supports_hz length matches num_filts
    assert len(bank.supports_hz) == NUM_FILTS
    # supports length matches num_filts
    assert len(bank.supports) == NUM_FILTS
    # supports_ms matches supports length and units conversion
    assert len(bank.supports_ms) == NUM_FILTS
    for (start_samp, end_samp), (start_ms, end_ms) in zip(bank.supports, bank.supports_ms):
        assert np.isclose(start_ms, start_samp * 1000 / SAMPLING_RATE)
        assert np.isclose(end_ms, end_samp * 1000 / SAMPLING_RATE)

    assert bank.is_real != bank.is_analytic
    assert bank.is_zero_phase is True
    # centers_hz should be sorted ascending
    assert all(x < y for x, y in zip(bank.centers_hz, bank.centers_hz[1:]))


@pytest.mark.parametrize("filt_idx", [0, NUM_FILTS // 2, NUM_FILTS - 1])
@pytest.mark.parametrize("width", [64, 128, 256])
@pytest.mark.parametrize("analytic", [False, True])
def test_triangular_impulse_and_frequency_response_consistency(filt_idx, width, analytic):
    seed_all()
    bank = TriangularOverlappingFilterBank(
        scaling_function="mel", num_filts=NUM_FILTS,
        low_hz=LOW_HZ, high_hz=HIGH_HZ,
        sampling_rate=SAMPLING_RATE, analytic=analytic,
    )

    impulse_response = bank.get_impulse_response(filt_idx, width)
    freq_response = bank.get_frequency_response(filt_idx, width, half=False)
    # Impulse response length must match width
    assert impulse_response.shape == (width,)
    # Frequency response length must match width
    assert freq_response.shape == (width,)
    # Check types depending on analytic flag
    if analytic:
        assert np.iscomplexobj(impulse_response)
        assert freq_response.dtype == np.float64
    else:
        assert np.isrealobj(impulse_response)
        # Frequency response should be real
        assert freq_response.dtype == np.float64
    # Magnitude of freq response should be non-negative
    assert np.all(freq_response >= 0.0)
    # Supports boundaries enclose nonzero bins
    left_freq, right_freq = bank.supports_hz[filt_idx]
    left_idx, _ = bank.get_truncated_response(filt_idx, width)
    assert left_idx >= 0
    # truncated response shape > 0 and matches expected length
    left_idx, truncated = bank.get_truncated_response(filt_idx, width)
    assert truncated.ndim == 1
    assert truncated.size > 0
    assert left_idx + truncated.size <= width


def test_triangular_filterbank_invalid_freq_range():
    seed_all()
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank("mel", low_hz=-1)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank("mel", low_hz=500, high_hz=200)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank("mel", high_hz=SAMPLING_RATE)


@pytest.mark.parametrize("analytic", [False, True])
def test_fbank_basic_properties(analytic):
    seed_all()
    bank = Fbank(
        num_filts=NUM_FILTS, high_hz=HIGH_HZ,
        low_hz=LOW_HZ, sampling_rate=SAMPLING_RATE,
        analytic=analytic,
    )
    assert bank.num_filts == NUM_FILTS
    assert bank.sampling_rate == SAMPLING_RATE
    assert len(bank.centers_hz) == NUM_FILTS
    assert len(bank.supports_hz) == NUM_FILTS
    assert len(bank.supports) == NUM_FILTS
    assert len(bank.supports_ms) == NUM_FILTS
    # Check meta properties
    assert bank.is_real != bank.is_analytic
    assert bank.is_zero_phase is True


@pytest.mark.parametrize("filt_idx", [0, NUM_FILTS//2, NUM_FILTS-1])
@pytest.mark.parametrize("width", [64, 128, 256])
@pytest.mark.parametrize("analytic", [False, True])
@pytest.mark.parametrize("half", [False, True])
def test_fbank_frequency_and_truncated_response(filt_idx, width, analytic, half):
    seed_all()
    bank = Fbank(
        num_filts=NUM_FILTS, low_hz=LOW_HZ, high_hz=HIGH_HZ,
        sampling_rate=SAMPLING_RATE, analytic=analytic,
    )

    freq = bank.get_frequency_response(filt_idx, width, half=half)
    left_idx, truncated = bank.get_truncated_response(filt_idx, width)

    # Frequency response shape depends on half flag and width parity
    if half:
        expected_len = width // 2 + 1 if width % 2 == 0 else (width + 1) // 2
        assert freq.shape == (expected_len,)
    else:
        assert freq.shape == (width,)

    # truncated response should be 1D, nonempty
    assert truncated.ndim == 1
    assert truncated.size > 0
    # left_idx should be within frequency response range
    assert 0 <= left_idx < freq.size
    # truncated length fits into frequency response from left_idx
    assert left_idx + truncated.size <= freq.size
    # Values in truncated should be between 0 and 1 (mel sqrt of triangle)
    assert np.all(truncated >= 0) and np.all(truncated <= 1)


@pytest.mark.parametrize("scale_l2_norm", [False, True])
@pytest.mark.parametrize("erb", [False, True])
def test_gabor_filter_bank_basic_properties(scale_l2_norm, erb):
    seed_all()
    bank = GaborFilterBank(
        "mel", num_filts=NUM_FILTS, low_hz=LOW_HZ, high_hz=HIGH_HZ,
        sampling_rate=SAMPLING_RATE, scale_l2_norm=scale_l2_norm, erb=erb
    )
    assert bank.num_filts == NUM_FILTS
    assert bank.sampling_rate == SAMPLING_RATE
    assert len(bank.centers_hz) == NUM_FILTS
    assert bank.is_real is False
    assert bank.is_zero_phase is True
    # is_analytic depends on wrap_below, but at least type bool
    assert isinstance(bank.is_analytic, bool)
    # supports length matches num_filts
    assert len(bank.supports_hz) == NUM_FILTS
    assert len(bank.supports) == NUM_FILTS
    # scaled_l2_norm and erb props reflect parameters
    assert bank.scaled_l2_norm == scale_l2_norm
    assert bank.erb == erb


@pytest.mark.parametrize("filt_idx", [0, NUM_FILTS // 2, NUM_FILTS - 1])
@pytest.mark.parametrize("width", [64, 128])
def test_gabor_impulse_and_frequency_response_shapes(filt_idx, width):
    seed_all()
    bank = GaborFilterBank("mel", num_filts=NUM_FILTS, low_hz=LOW_HZ, high_hz=HIGH_HZ)

    impulse = bank.get_impulse_response(filt_idx, width)
    freq = bank.get_frequency_response(filt_idx, width, half=False)

    assert impulse.shape == (width,)
    assert freq.shape == (width,)
    assert np.iscomplexobj(impulse)
    assert np.iscomplexobj(freq)

    # truncated response shape and left_idx within bounds
    left_idx, truncated = bank.get_truncated_response(filt_idx, width)
    assert 0 <= left_idx < width
    assert truncated.ndim == 1
    assert truncated.size > 0
    assert truncated.dtype == np.float64


@pytest.mark.parametrize("order", [1, 4, 10])
@pytest.mark.parametrize("scale_l2_norm", [False, True])
@pytest.mark.parametrize("erb", [False, True])
@pytest.mark.parametrize("max_centered", [False, True])
def test_complex_gammatone_filter_bank_basic_properties(order, scale_l2_norm, erb, max_centered):
    seed_all()
    bank = ComplexGammatoneFilterBank(
        "mel", num_filts=NUM_FILTS, low_hz=LOW_HZ, high_hz=HIGH_HZ,
        sampling_rate=SAMPLING_RATE, order=order, max_centered=max_centered,
        scale_l2_norm=scale_l2_norm, erb=erb
    )
    assert bank.num_filts == NUM_FILTS
    assert bank.order == order
    assert bank.sampling_rate == SAMPLING_RATE
    assert len(bank.centers_hz) == NUM_FILTS
    assert bank.is_real is False
    assert bank.is_zero_phase is False
    assert isinstance(bank.is_analytic, bool)
    assert len(bank.supports_hz) == NUM_FILTS
    assert bank.scaled_l2_norm == scale_l2_norm
    assert bank.erb == erb


@pytest.mark.parametrize("filt_idx", [0, NUM_FILTS // 2, NUM_FILTS - 1])
@pytest.mark.parametrize("width", [64, 128])
def test_complex_gammatone_impulse_and_frequency_response(filt_idx, width):
    seed_all()
    bank = ComplexGammatoneFilterBank("mel", num_filts=NUM_FILTS, low_hz=LOW_HZ, high_hz=HIGH_HZ)

    impulse = bank.get_impulse_response(filt_idx, width)
    freq = bank.get_frequency_response(filt_idx, width, half=False)

    assert impulse.shape == (width,)
    assert freq.shape in [(width,), (width // 2 + 1), ((width + 1) // 2)]
    assert np.iscomplexobj(impulse)
    assert np.iscomplexobj(freq)

    # truncated response shape and left_idx within bounds
    left_idx, truncated = bank.get_truncated_response(filt_idx, width)
    assert 0 <= left_idx < width
    assert truncated.ndim == 1
    assert truncated.size > 0


def test_complex_gammatone_invalid_order():
    seed_all()
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", order=0)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", order=-1)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", order=2.5)


def test_frequency_response_half_and_full_shape_consistency():
    seed_all()
    bank = TriangularOverlappingFilterBank("mel", num_filts=NUM_FILTS, low_hz=LOW_HZ, high_hz=HIGH_HZ)

    width_even = 128
    width_odd = 129

    # half=True responses size
    freq_half_even = bank.get_frequency_response(0, width_even, half=True)
    expected_len_even = width_even // 2 + 1
    assert freq_half_even.shape == (expected_len_even,)

    freq_half_odd = bank.get_frequency_response(0, width_odd, half=True)
    expected_len_odd = (width_odd + 1) // 2
    assert freq_half_odd.shape == (expected_len_odd,)

    freq_full_even = bank.get_frequency_response(0, width_even, half=False)
    assert freq_full_even.shape == (width_even,)

    freq_full_odd = bank.get_frequency_response(0, width_odd, half=False)
    assert freq_full_odd.shape == (width_odd,)


@pytest.mark.parametrize("filt_idx", [0, NUM_FILTS // 2, NUM_FILTS - 1])
def test_get_truncated_response_consistent_with_frequency_response(filt_idx):
    seed_all()
    bank = TriangularOverlappingFilterBank("mel", num_filts=NUM_FILTS, low_hz=LOW_HZ, high_hz=HIGH_HZ)
    width = 256

    left_idx, truncated = bank.get_truncated_response(filt_idx, width)
    freq_response = bank.get_frequency_response(filt_idx, width, half=False)
    full_reconstructed = np.zeros_like(freq_response)
    wrap_len = min(left_idx + truncated.size, width) - left_idx
    full_reconstructed[left_idx:left_idx + wrap_len] = truncated[:wrap_len]
    if wrap_len < truncated.size:
        full_reconstructed[:truncated.size - wrap_len] = truncated[wrap_len:]
    # Assert that reconstructed response is close to truncated values in their place
    assert np.allclose(full_reconstructed[left_idx:left_idx + truncated.size], truncated, atol=1e-12)
    # Values outside truncated region should be zero or near zero within threshold
    outside_mask = np.ones_like(freq_response, dtype=bool)
    outside_mask[left_idx:left_idx + truncated.size] = False
    assert np.all(np.abs(full_reconstructed[outside_mask]) <= 1e-12)

