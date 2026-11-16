import numpy as np
import pytest

from pydrobert.speech.filters import (
    LinearFilterBank,
    TriangularOverlappingFilterBank,
    Fbank,
    GaborFilterBank,
    ComplexGammatoneFilterBank,
)
from pydrobert.speech.scales import MelScaling

RATE = 16000
WIDTH = 256
LOW_HZ = 100.
HIGH_HZ = 7000.
NUM_FILTS = 8


def _tri(analytic=False):
    return TriangularOverlappingFilterBank(
        MelScaling(), num_filts=NUM_FILTS, high_hz=HIGH_HZ, low_hz=LOW_HZ,
        sampling_rate=RATE, analytic=analytic)


def _fbank(analytic=False):
    return Fbank(
        num_filts=NUM_FILTS, high_hz=HIGH_HZ, low_hz=LOW_HZ,
        sampling_rate=RATE, analytic=analytic)


def _gabor():
    return GaborFilterBank(
        MelScaling(), num_filts=NUM_FILTS, high_hz=HIGH_HZ, low_hz=LOW_HZ,
        sampling_rate=RATE)


def _gammatone():
    return ComplexGammatoneFilterBank(
        MelScaling(), num_filts=NUM_FILTS, high_hz=HIGH_HZ, low_hz=LOW_HZ,
        sampling_rate=RATE)


ALL_MAKERS = [
    pytest.param(lambda: _tri(False), id='tri'),
    pytest.param(lambda: _tri(True), id='tri-analytic'),
    pytest.param(lambda: _fbank(False), id='fbank'),
    pytest.param(lambda: _fbank(True), id='fbank-analytic'),
    pytest.param(_gabor, id='gabor'),
    pytest.param(_gammatone, id='gammatone'),
]


def _expected_half_len(width):
    if width % 2:
        return (width + 1) // 2
    return width // 2 + 1


def _rebuild_full(bin_idx, trnc, width):
    # reconstruct a full-length DFT from a (bin_idx, buf) truncated response
    full = np.zeros(width, dtype=np.complex128)
    for j, val in enumerate(trnc):
        full[(bin_idx + j) % width] += val
    return full


def test_all_banks_are_linear_filter_banks():
    for cls in (
            TriangularOverlappingFilterBank, Fbank, GaborFilterBank,
            ComplexGammatoneFilterBank):
        assert issubclass(cls, LinearFilterBank)


@pytest.mark.parametrize('make', ALL_MAKERS)
def test_num_filts_and_centers(make):
    bank = make()
    assert bank.num_filts == NUM_FILTS
    centers = np.asarray(bank.centers_hz)
    assert len(centers) == NUM_FILTS
    assert np.all(np.diff(centers) > 0)
    assert np.all(centers > LOW_HZ) and np.all(centers < HIGH_HZ)


@pytest.mark.parametrize('make', ALL_MAKERS)
def test_attributes_are_consistent(make):
    bank = make()
    assert bank.sampling_rate == RATE
    assert len(bank.supports) == bank.num_filts
    assert len(bank.supports_hz) == bank.num_filts
    assert len(bank.supports_ms) == bank.num_filts
    assert bank.is_real != bank.is_analytic or bank.is_real is False


@pytest.mark.parametrize('make', ALL_MAKERS)
def test_supports_are_integer_pairs(make):
    bank = make()
    for lo, hi in bank.supports:
        assert isinstance(lo, int)
        assert isinstance(hi, int)
        assert lo < hi


@pytest.mark.parametrize('make', ALL_MAKERS)
def test_supports_ms_scale_supports(make):
    bank = make()
    for (lo, hi), (lo_ms, hi_ms) in zip(bank.supports, bank.supports_ms):
        assert lo_ms == pytest.approx(lo * 1000 / RATE)
        assert hi_ms == pytest.approx(hi * 1000 / RATE)


@pytest.mark.parametrize('make', ALL_MAKERS)
def test_supports_hz_bracket_centers(make):
    bank = make()
    for center, (lo, hi) in zip(bank.centers_hz, bank.supports_hz):
        assert lo < center < hi


@pytest.mark.parametrize('low_hz,high_hz', [
    (-1., 4000.),
    (4000., 4000.),
    (4000., 3000.),
    (100., 9000.),
])
def test_triangular_invalid_range_raises(low_hz, high_hz):
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(
            MelScaling(), num_filts=4, high_hz=high_hz, low_hz=low_hz,
            sampling_rate=RATE)


def test_triangular_default_high_is_nyquist():
    bank = TriangularOverlappingFilterBank(
        MelScaling(), num_filts=4, sampling_rate=RATE)
    assert bank.supports_hz[-1][1] == pytest.approx(RATE // 2, rel=1e-6)


def test_triangular_real_impulse_response_symmetric():
    bank = _tri(False)
    ir = bank.get_impulse_response(2, WIDTH)
    assert ir.shape == (WIDTH,)
    assert ir.dtype == np.float64
    assert np.allclose(ir[1:], ir[1:][::-1])


@pytest.mark.parametrize('make', [
    pytest.param(lambda: _tri(True), id='tri-analytic'),
    pytest.param(_gabor, id='gabor'),
    pytest.param(_gammatone, id='gammatone'),
])
def test_complex_impulse_response_shape_and_dtype(make):
    bank = make()
    ir = bank.get_impulse_response(1, WIDTH)
    assert ir.shape == (WIDTH,)
    assert ir.dtype == np.complex128


def test_fbank_real_impulse_response_is_real_float():
    ir = _fbank(False).get_impulse_response(3, WIDTH)
    assert ir.shape == (WIDTH,)
    assert ir.dtype == np.float64


@pytest.mark.parametrize('make', ALL_MAKERS)
@pytest.mark.parametrize('width', [255, 256])
def test_frequency_response_lengths(make, width):
    bank = make()
    for i in range(bank.num_filts):
        full = bank.get_frequency_response(i, width, half=False)
        half = bank.get_frequency_response(i, width, half=True)
        assert full.shape == (width,)
        assert half.shape == (_expected_half_len(width),)


@pytest.mark.parametrize('make', ALL_MAKERS)
def test_frequency_response_dtype_follows_is_zero_phase(make):
    bank = make()
    res = bank.get_frequency_response(0, WIDTH)
    if bank.is_zero_phase:
        assert res.dtype == np.float64
    else:
        assert res.dtype == np.complex128


@pytest.mark.parametrize('make', [
    pytest.param(lambda: _tri(False), id='tri'),
    pytest.param(lambda: _fbank(False), id='fbank'),
])
def test_real_frequency_response_bounded_and_symmetric(make):
    bank = make()
    for i in range(bank.num_filts):
        res = bank.get_frequency_response(i, WIDTH)
        assert np.all(res >= 0)
        assert np.all(res <= 1 + 1e-12)
        assert res.max() > 0
        # real filters: res[k] == res[width - k] for k in 1..width-1
        assert np.allclose(res[1:], res[1:][::-1])


@pytest.mark.parametrize('make', [
    pytest.param(lambda: _tri(True), id='tri-analytic'),
    pytest.param(lambda: _fbank(True), id='fbank-analytic'),
])
def test_analytic_frequency_response_has_no_negative_bins(make):
    bank = make()
    assert bank.is_analytic
    assert not bank.is_real
    for i in range(bank.num_filts):
        res = bank.get_frequency_response(i, WIDTH, half=False)
        # all support lies below Nyquist, so the upper half must be empty
        assert np.all(res[WIDTH // 2 + 1:] == 0)


@pytest.mark.parametrize('make', [
    pytest.param(lambda: _tri(False), id='tri'),
    pytest.param(lambda: _tri(True), id='tri-analytic'),
    pytest.param(lambda: _fbank(False), id='fbank'),
    pytest.param(lambda: _fbank(True), id='fbank-analytic'),
])
def test_truncated_matches_half_response(make):
    bank = make()
    for i in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(i, WIDTH)
        half = bank.get_frequency_response(i, WIDTH, half=True)
        assert trnc.dtype == np.float64
        assert np.allclose(half[bin_idx:bin_idx + len(trnc)], trnc)
        assert np.all(half[:bin_idx] == 0)
        assert np.all(half[bin_idx + len(trnc):] == 0)


@pytest.mark.parametrize('make', ALL_MAKERS)
def test_truncated_bin_index_in_range(make):
    bank = make()
    for i in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(i, WIDTH)
        assert isinstance(bin_idx, (int, np.integer))
        assert 0 <= bin_idx < WIDTH
        assert len(trnc) >= 1


@pytest.mark.parametrize('make', [
    pytest.param(_gabor, id='gabor'),
    pytest.param(_gammatone, id='gammatone'),
])
def test_truncated_rebuild_matches_full_response(make):
    bank = make()
    for i in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(i, WIDTH)
        full = bank.get_frequency_response(i, WIDTH)
        rebuilt = _rebuild_full(bin_idx, trnc, WIDTH)
        tol = 1e-3 * np.abs(full).max()
        assert np.allclose(rebuilt, full, rtol=0, atol=tol)


@pytest.mark.parametrize('order', [0, -1, 2.5])
def test_gammatone_invalid_order_raises(order):
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(
            MelScaling(), num_filts=4, high_hz=HIGH_HZ, low_hz=LOW_HZ,
            sampling_rate=RATE, order=order)


def test_gammatone_order_and_flags():
    bank = ComplexGammatoneFilterBank(
        MelScaling(), num_filts=4, high_hz=HIGH_HZ, low_hz=LOW_HZ,
        sampling_rate=RATE, order=3)
    assert bank.order == 3
    assert not bank.is_zero_phase
    assert not bank.is_real
    assert bank.scaled_l2_norm is False
    assert bank.erb is False


def test_gabor_flags():
    bank = GaborFilterBank(
        MelScaling(), num_filts=4, high_hz=HIGH_HZ, low_hz=LOW_HZ,
        sampling_rate=RATE, scale_l2_norm=True, erb=True)
    assert bank.scaled_l2_norm is True
    assert bank.erb is True
    assert bank.is_zero_phase
    assert not bank.is_real


def test_gammatone_impulse_response_nonzero():
    bank = _gammatone()
    ir = bank.get_impulse_response(0, WIDTH)
    assert ir.shape == (WIDTH,)
    assert np.abs(ir).max() > 0
