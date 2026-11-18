import math
import numpy as np
import pytest

from pydrobert.speech.filters import (
    TriangularOverlappingFilterBank,
    Fbank,
    GaborFilterBank,
    ComplexGammatoneFilterBank,
)

# Use a simple linear scaling function for TriangularOverlappingFilterBank
from pydrobert.speech.scales import LinearScaling


@pytest.mark.parametrize("num_filts", [1, 5, 10])
def test_triangular_num_filters_and_centers(num_filts):
    bank = TriangularOverlappingFilterBank(
        scaling_function=LinearScaling(),
        num_filts=num_filts,
        low_hz=20,
        high_hz=8000,
        sampling_rate=16000,
        analytic=False,
    )
    # num_filts property should match input
    assert bank.num_filts == num_filts
    assert len(bank.centers_hz) == num_filts
    assert isinstance(bank.centers_hz, tuple)


def test_triangular_invalid_frequency_range():
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(
            scaling_function=LinearScaling(),
            low_hz=-1,
        )
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(
            scaling_function=LinearScaling(),
            low_hz=500,
            high_hz=400,
        )
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(
            scaling_function=LinearScaling(),
            low_hz=0,
            high_hz=100000,  # above Nyquist at 16000//2=8000
            sampling_rate=16000,
        )


def test_triangular_is_real_analytic_zero_phase():
    bank = TriangularOverlappingFilterBank(
        scaling_function=LinearScaling(),
        analytic=True,
    )
    assert bank.is_analytic
    assert not bank.is_real
    assert bank.is_zero_phase

    bank2 = TriangularOverlappingFilterBank(
        scaling_function=LinearScaling(),
        analytic=False,
    )
    assert not bank2.is_analytic
    assert bank2.is_real
    assert bank2.is_zero_phase


@pytest.mark.parametrize("width", [16, 32])
def test_triangular_get_impulse_response_real_and_analytic(width):
    bank_real = TriangularOverlappingFilterBank(LinearScaling(), analytic=False)
    res_real = bank_real.get_impulse_response(0, width)
    assert isinstance(res_real, np.ndarray)
    assert res_real.dtype == np.float64
    assert res_real.shape == (width,)

    bank_analytic = TriangularOverlappingFilterBank(LinearScaling(), analytic=True)
    res_analytic = bank_analytic.get_impulse_response(0, width)
    assert isinstance(res_analytic, np.ndarray)
    assert res_analytic.dtype == np.complex128
    assert res_analytic.shape == (width,)


@pytest.mark.parametrize("width", [16, 32])
@pytest.mark.parametrize("half", [False, True])
def test_triangular_get_frequency_response_properties(width, half):
    bank = TriangularOverlappingFilterBank(LinearScaling(), analytic=False)
    for filt_idx in [0, max(0, bank.num_filts - 1)]:
        freq_resp = bank.get_frequency_response(filt_idx, width, half=half)
        assert isinstance(freq_resp, np.ndarray)
        # dtype depends on zero phase
        assert freq_resp.dtype == np.float64
        if half:
            expected_len = width // 2 + 1 if width % 2 == 0 else (width + 1) // 2
        else:
            expected_len = width
        assert freq_resp.shape == (expected_len,)
        # values are between 0 and 1 roughly
        assert np.all(freq_resp >= 0)
        assert np.all(freq_resp <= 1)


@pytest.mark.parametrize("width", [32, 64])
def test_triangular_get_truncated_response(width):
    bank = TriangularOverlappingFilterBank(LinearScaling(), analytic=False)
    filt_idx = 0
    start_bin, truncated = bank.get_truncated_response(filt_idx, width)
    assert isinstance(start_bin, int)
    assert isinstance(truncated, np.ndarray)
    # truncated length should be positive and consistent with truncated values
    assert truncated.size > 0
    # check values in expected range [0,1]
    assert np.all(truncated >= 0)
    assert np.all(truncated <= 1)


def test_fbank_num_filts_and_centers():
    bank = Fbank(num_filts=20, low_hz=50, high_hz=7000, sampling_rate=16000)
    assert bank.num_filts == 20
    assert len(bank.centers_hz) == 20
    for hz in bank.centers_hz:
        # values are within the frequency range
        assert 50 <= hz <= 7000


def test_fbank_invalid_frequency_range():
    with pytest.raises(ValueError):
        Fbank(low_hz=-0.1)
    with pytest.raises(ValueError):
        Fbank(low_hz=5000, high_hz=1000)
    with pytest.raises(ValueError):
        Fbank(low_hz=0, high_hz=9000, sampling_rate=16000)


@pytest.mark.parametrize("width", [32, 64])
def test_fbank_get_impulse_response_and_frequency_response(width):
    bank = Fbank(num_filts=10)
    ir = bank.get_impulse_response(0, width)
    assert ir.shape == (width,)
    # is real valued array (float64)
    assert ir.dtype == np.float64

    freq_resp = bank.get_frequency_response(0, width, half=True)
    assert freq_resp.dtype == np.float64
    expected_len = width // 2 + 1 if width % 2 == 0 else (width + 1) // 2
    assert freq_resp.shape == (expected_len,)
    # values nonnegative and less than or equal to 1 (square root function)
    assert np.all(freq_resp >= 0)
    assert np.all(freq_resp <= 1)


@pytest.mark.parametrize("width", [32, 64])
def test_fbank_get_truncated_response(width):
    bank = Fbank(num_filts=10)
    start_idx, truncated = bank.get_truncated_response(0, width)
    assert isinstance(start_idx, int)
    assert isinstance(truncated, np.ndarray)
    assert truncated.size > 0
    assert np.all(truncated >= 0)


def test_gabor_filter_bank_initialization_and_attributes():
    bank = GaborFilterBank(
        scaling_function=LinearScaling(),
        num_filts=5,
        low_hz=20,
        high_hz=8000,
        sampling_rate=16000,
        scale_l2_norm=True,
        erb=True,
    )
    assert bank.num_filts == 5
    assert isinstance(bank.centers_hz, tuple)
    assert isinstance(bank.supports, tuple)
    assert isinstance(bank.supports_hz, tuple)
    assert bank.is_real is False
    # When supports_ang have values below 0, is_analytic is False
    assert isinstance(bank.is_analytic, bool)
    assert bank.is_zero_phase


@pytest.mark.parametrize("width", [64, 128])
def test_gabor_get_impulse_response_and_frequency_response(width):
    bank = GaborFilterBank(LinearScaling(), num_filts=3)
    for filt_idx in range(bank.num_filts):
        ir = bank.get_impulse_response(filt_idx, width)
        # impulse response dtype complex128, length=width
        assert isinstance(ir, np.ndarray)
        assert ir.shape == (width,)
        assert ir.dtype == np.complex128

        freq_resp = bank.get_frequency_response(filt_idx, width, half=True)
        # frequency response dtype float64, length <= half spectrum
        assert isinstance(freq_resp, np.ndarray)
        expected_len = width // 2 + 1 if width % 2 == 0 else (width + 1) // 2
        assert freq_resp.shape == (expected_len,)
        assert freq_resp.dtype == np.float64


def test_gabor_get_truncated_response_wrap_covered_and_partial():
    bank = GaborFilterBank(LinearScaling(), num_filts=2)
    width = 64

    # Forces wrap_supports_ang < 2pi
    idx = 0
    start_idx, truncated = bank.get_truncated_response(idx, width)
    assert isinstance(start_idx, int)
    assert isinstance(truncated, np.ndarray)
    assert truncated.size > 0

    # Force wrap_supports_ang >= 2pi by monkeypatching
    bank._wrap_supports_ang = (2 * np.pi + 1,) * bank.num_filts
    start_idx2, truncated2 = bank.get_truncated_response(0, width)
    # In this case start idx is 0 and truncated response is full frequency response
    assert start_idx2 == 0
    assert truncated2.shape == (width,)
    assert truncated2.dtype == np.float64


def test_complex_gammatone_filter_bank_initialization_and_errors():
    # Valid init
    bank = ComplexGammatoneFilterBank(LinearScaling(), num_filts=4, order=3)

    assert bank.num_filts == 4
    assert bank.order == 3
    assert bank.is_zero_phase is False
    assert not bank.is_real
    assert isinstance(bank.centers_hz, tuple)

    # Invalid order
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(LinearScaling(), order=-1)

    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(LinearScaling(), order=0)

    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(LinearScaling(), order=2.5)


@pytest.mark.parametrize("width", [32, 64])
def test_complex_gammatone_get_impulse_response_and_frequency_response(width):
    bank = ComplexGammatoneFilterBank(LinearScaling(), num_filts=3, order=4)
    for filt_idx in range(bank.num_filts):
        ir = bank.get_impulse_response(filt_idx, width)
        assert isinstance(ir, np.ndarray)
        assert ir.shape == (width,)
        assert ir.dtype == np.complex128

        freq_resp = bank.get_frequency_response(filt_idx, width, half=True)
        assert isinstance(freq_resp, np.ndarray)
        expected_len = width // 2 + 1 if width % 2 == 0 else (width + 1) // 2
        assert freq_resp.shape == (expected_len,)
        assert freq_resp.dtype == np.complex128


@pytest.mark.parametrize("width", [64, 128])
def test_complex_gammatone_get_truncated_response_behavior(width):
    bank = ComplexGammatoneFilterBank(LinearScaling(), num_filts=1)
    filt_idx = 0

    # Normal case with support less than 2pi
    bank._wrap_supports_ang = (bank._wrap_supports_ang[0],)
    bank._supports_ang = (
        (0.1, 0.5),  # low and high angular freq in radians, forcibly <2pi range
    )

    start_idx, truncated = bank.get_truncated_response(filt_idx, width)
    assert isinstance(start_idx, int)
    assert isinstance(truncated, np.ndarray)
    assert truncated.size > 0

    # Wrap support covers entire period, disables truncation
    bank._wrap_supports_ang = (4 * np.pi,)
    bank._supports_ang = ((0.0, 2 * np.pi),)
    start_idx2, truncated2 = bank.get_truncated_response(filt_idx, width)
    assert start_idx2 == 0
    assert truncated2.shape == (width,)
    assert truncated2.dtype == np.complex128


def test_triangular_support_consistency():
    bank = TriangularOverlappingFilterBank(LinearScaling(), num_filts=5)
    supports = bank.supports
    supports_hz = bank.supports_hz
    assert len(supports) == bank.num_filts
    assert len(supports_hz) == bank.num_filts
    # Each support is a pair of ints
    for start, end in supports:
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert start <= end
    # Each support in Hz is a pair of floats with low < high
    for low, high in supports_hz:
        assert isinstance(low, float)
        assert isinstance(high, float)
        assert low < high


def test_fbank_support_consistency():
    bank = Fbank(num_filts=5)
    supports = bank.supports
    supports_hz = bank.supports_hz
    assert len(supports) == bank.num_filts
    assert len(supports_hz) == bank.num_filts
    for start, end in supports:
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert start <= end
    for low, high in supports_hz:
        assert isinstance(low, float)
        assert isinstance(high, float)
        assert low < high


def test_gabor_support_consistency():
    bank = GaborFilterBank(LinearScaling(), num_filts=3)
    supports = bank.supports
    supports_hz = bank.supports_hz
    assert len(supports) == bank.num_filts
    assert len(supports_hz) == bank.num_filts
    for start, end in supports:
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert start <= end or start == -end  # supports can be negative and symmetric
    for low, high in supports_hz:
        assert isinstance(low, float)
        assert isinstance(high, float)
        assert low < high


def test_complex_gammatone_support_consistency():
    bank = ComplexGammatoneFilterBank(LinearScaling(), num_filts=3, order=4)
    supports = bank.supports
    supports_hz = bank.supports_hz
    assert len(supports) == bank.num_filts
    assert len(supports_hz) == bank.num_filts
    for start, end in supports:
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert start <= end or start == -end
    for low, high in supports_hz:
        assert isinstance(low, float)
        assert isinstance(high, float)
        assert low < high


def test_gabor_scaled_l2_norm_and_erb_configs():
    bank1 = GaborFilterBank(LinearScaling(), scale_l2_norm=True, erb=True)
    bank2 = GaborFilterBank(LinearScaling(), scale_l2_norm=False, erb=False)
    assert bank1.scaled_l2_norm is True
    assert bank1.erb is True
    assert bank2.scaled_l2_norm is False
    assert bank2.erb is False


def test_complex_gammatone_scaled_l2_norm_and_erb_configs():
    bank1 = ComplexGammatoneFilterBank(LinearScaling(), scale_l2_norm=True, erb=True)
    bank2 = ComplexGammatoneFilterBank(LinearScaling(), scale_l2_norm=False, erb=False)
    assert bank1.scaled_l2_norm is True
    assert bank1.erb is True
    assert bank2.scaled_l2_norm is False
    assert bank2.erb is False


def test_triangular_and_fbank_supports_ms(monkeypatch):
    bank = TriangularOverlappingFilterBank(LinearScaling(), num_filts=3)
    supports = bank.supports
    supports_ms = bank.supports_ms
    for (start_sample, end_sample), (start_ms, end_ms) in zip(supports, supports_ms):
        duration = (end_sample - start_sample) * 1000 / bank.sampling_rate
        assert math.isclose(duration, end_ms - start_ms, rel_tol=1e-9)

    bank2 = Fbank(num_filts=3)
    supports2 = bank2.supports
    supports_ms2 = bank2.supports_ms
    for (start_sample, end_sample), (start_ms, end_ms) in zip(supports2, supports_ms2):
        duration = (end_sample - start_sample) * 1000 / bank2.sampling_rate
        assert math.isclose(duration, end_ms - start_ms, rel_tol=1e-9)
