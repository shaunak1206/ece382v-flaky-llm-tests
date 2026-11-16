import numpy as np
import pytest

from pydrobert.speech import config
from pydrobert.speech.filters import (
    ComplexGammatoneFilterBank,
    Fbank,
    GaborFilterBank,
    TriangularOverlappingFilterBank,
)
from pydrobert.speech.scales import MelScaling

RATE = 16000
NUM_FILTS = 12


def make_tri(**kwargs):
    kwargs.setdefault('num_filts', NUM_FILTS)
    kwargs.setdefault('sampling_rate', RATE)
    return TriangularOverlappingFilterBank(MelScaling(), **kwargs)


def make_fbank(**kwargs):
    kwargs.setdefault('num_filts', NUM_FILTS)
    kwargs.setdefault('sampling_rate', RATE)
    return Fbank(**kwargs)


def make_gabor(**kwargs):
    kwargs.setdefault('num_filts', NUM_FILTS)
    kwargs.setdefault('sampling_rate', RATE)
    return GaborFilterBank(MelScaling(), **kwargs)


def make_gammatone(**kwargs):
    kwargs.setdefault('num_filts', NUM_FILTS)
    kwargs.setdefault('sampling_rate', RATE)
    return ComplexGammatoneFilterBank(MelScaling(), **kwargs)


REAL_FACTORIES = [make_tri, make_fbank]
ALL_FACTORIES = [make_tri, make_fbank, make_gabor, make_gammatone]


def test_triangular_basic_properties():
    bank = make_tri()
    assert bank.num_filts == NUM_FILTS
    assert bank.sampling_rate == RATE
    assert bank.is_real
    assert not bank.is_analytic
    assert bank.is_zero_phase
    centers = np.asarray(bank.centers_hz)
    assert len(centers) == NUM_FILTS
    assert np.all(np.diff(centers) > 0)
    assert len(bank.supports) == NUM_FILTS
    assert len(bank.supports_hz) == NUM_FILTS


def test_triangular_analytic_flags():
    bank = make_tri(analytic=True)
    assert not bank.is_real
    assert bank.is_analytic
    assert bank.is_zero_phase


def test_triangular_default_high_is_nyquist():
    bank = make_tri()
    _, high = bank.supports_hz[-1]
    assert high == pytest.approx(RATE // 2)


@pytest.mark.parametrize('factory', [make_tri, make_fbank, make_gabor])
@pytest.mark.parametrize(
    'low_hz, high_hz', [(-1., 4000.), (1000., 500.), (100., 9000.)])
def test_invalid_frequency_range_raises(factory, low_hz, high_hz):
    with pytest.raises(ValueError):
        factory(low_hz=low_hz, high_hz=high_hz)


@pytest.mark.parametrize('order', [0, -2, 2.5])
def test_gammatone_invalid_order_raises(order):
    with pytest.raises(ValueError):
        make_gammatone(order=order)


@pytest.mark.parametrize('factory', REAL_FACTORIES)
def test_real_full_response_is_symmetric(factory):
    width = 1024
    bank = factory()
    for idx in range(bank.num_filts):
        res = bank.get_frequency_response(idx, width)
        assert res.shape == (width,)
        assert res.dtype == np.float64
        np.testing.assert_allclose(
            res[1:width // 2], res[width - 1:width // 2:-1])


@pytest.mark.parametrize('width, expected', [(511, 256), (512, 257)])
@pytest.mark.parametrize('factory', ALL_FACTORIES)
def test_half_response_length(factory, width, expected):
    bank = factory()
    for idx in range(bank.num_filts):
        res = bank.get_frequency_response(idx, width, half=True)
        assert res.shape == (expected,)


@pytest.mark.parametrize('factory', [make_tri, make_fbank, make_gabor])
def test_half_response_is_prefix_of_full(factory):
    width = 512
    bank = factory()
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        half = bank.get_frequency_response(idx, width, half=True)
        np.testing.assert_allclose(half, full[:len(half)], atol=1e-12)


def test_triangular_analytic_response_has_no_negative_frequencies():
    width = 1024
    bank = make_tri(analytic=True)
    for idx in range(bank.num_filts):
        res = bank.get_frequency_response(idx, width)
        assert np.all(res[width // 2 + 1:] == 0)
        assert res.max() > 0


@pytest.mark.parametrize('factory', REAL_FACTORIES)
def test_real_response_bounded_in_unit_interval(factory):
    bank = factory()
    for idx in range(bank.num_filts):
        res = bank.get_frequency_response(idx, 512)
        assert np.all(res >= -1e-12)
        assert np.all(res <= 1 + 1e-9)


@pytest.mark.parametrize('factory', [make_tri, make_gabor])
def test_response_peaks_near_centers(factory):
    width = 8192
    bank = factory()
    bin_hz = RATE / width
    for idx, center in enumerate(bank.centers_hz):
        res = bank.get_frequency_response(idx, width, half=True)
        peak_hz = np.argmax(res) * bin_hz
        assert abs(peak_hz - center) <= 2 * bin_hz
        assert res.max() <= 1 + 1e-6


def test_gammatone_response_peaks_at_centers():
    width = 4096
    bank = make_gammatone()
    for idx, center in enumerate(bank.centers_hz):
        res = np.abs(bank.get_frequency_response(idx, width))
        peak_bin = int(np.argmax(res))
        center_bin = center * width / RATE
        dist = abs(peak_bin - center_bin)
        dist = min(dist, width - dist)
        assert dist <= 2
        assert abs(res.max() - 1) < 0.05


@pytest.mark.parametrize('factory', REAL_FACTORIES)
def test_real_truncated_response_matches_half_response(factory):
    width = 1024
    bank = factory()
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        assert isinstance(bin_idx, int)
        assert trnc.dtype == np.float64
        half = bank.get_frequency_response(idx, width, half=True)
        np.testing.assert_allclose(
            trnc, half[bin_idx:bin_idx + len(trnc)], atol=1e-12)
        outside = np.ones(len(half), dtype=bool)
        outside[bin_idx:bin_idx + len(trnc)] = False
        assert np.all(half[outside] == 0)


@pytest.mark.parametrize(
    'factory, check_outside',
    [(make_gabor, True), (make_gammatone, False)])
def test_truncated_response_consistent_with_full_response(
        factory, check_outside):
    width = 2048
    tol = 10 * config.EFFECTIVE_SUPPORT_THRESHOLD
    bank = factory()
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        left, trnc = bank.get_truncated_response(idx, width)
        assert 0 <= left < width
        assert len(trnc) <= width
        bins = (left + np.arange(len(trnc))) % width
        np.testing.assert_allclose(full[bins], trnc, atol=tol)
        if check_outside:
            outside = np.ones(width, dtype=bool)
            outside[bins] = False
            assert np.all(np.abs(full[outside]) <= tol)


@pytest.mark.parametrize('factory', REAL_FACTORIES)
def test_real_impulse_response_shape_and_dtype(factory):
    width = 256
    bank = factory()
    res = bank.get_impulse_response(0, width)
    assert res.shape == (width,)
    assert res.dtype == np.float64


@pytest.mark.parametrize(
    'factory',
    [lambda: make_tri(analytic=True), make_gabor, make_gammatone])
def test_complex_impulse_response_shape_and_dtype(factory):
    width = 256
    bank = factory()
    res = bank.get_impulse_response(0, width)
    assert res.shape == (width,)
    assert res.dtype == np.complex128


def test_triangular_supports_are_integer_pairs_and_ms_conversion():
    bank = make_tri()
    assert len(bank.supports) == NUM_FILTS
    for lo, hi in bank.supports:
        assert isinstance(lo, int) and isinstance(hi, int)
        assert lo <= 0 <= hi
    for (lo, hi), (lo_ms, hi_ms) in zip(bank.supports, bank.supports_ms):
        assert lo_ms == pytest.approx(lo * 1000 / RATE)
        assert hi_ms == pytest.approx(hi * 1000 / RATE)


def test_supports_hz_bracket_centers():
    for factory in ALL_FACTORIES:
        bank = factory()
        assert len(bank.supports_hz) == NUM_FILTS
        for (lo, hi), center in zip(bank.supports_hz, bank.centers_hz):
            assert lo < center < hi


def test_gabor_properties_and_flags():
    bank = make_gabor()
    assert not bank.is_real
    assert bank.is_zero_phase
    assert not bank.scaled_l2_norm
    assert not bank.erb
    assert bank.num_filts == NUM_FILTS
    bank2 = make_gabor(scale_l2_norm=True, erb=True)
    assert bank2.scaled_l2_norm
    assert bank2.erb


def test_gammatone_properties():
    bank = make_gammatone(order=3)
    assert bank.order == 3
    assert not bank.is_real
    assert not bank.is_zero_phase
    assert bank.num_filts == NUM_FILTS
    assert len(bank.centers_hz) == NUM_FILTS


def test_gammatone_max_centered_shifts_support_negative():
    centered = make_gammatone(max_centered=True)
    causal = make_gammatone()
    for (lo_c, _), (lo, _) in zip(centered.supports, causal.supports):
        assert lo_c < 0
        assert lo == 0
