import numpy as np
import pytest

from pydrobert.speech import config
from pydrobert.speech.filters import ComplexGammatoneFilterBank
from pydrobert.speech.filters import Fbank
from pydrobert.speech.filters import GaborFilterBank
from pydrobert.speech.filters import LinearFilterBank
from pydrobert.speech.filters import TriangularOverlappingFilterBank
from pydrobert.speech.scales import MelScaling

RATE = 16000
WIDTH = 512

needs_np_math = pytest.mark.skipif(
    not hasattr(np, 'math'),
    reason='ComplexGammatoneFilterBank uses np.math (removed in numpy 2.0)')


def _bin_hz(k, width=WIDTH, rate=RATE):
    return k * rate / width


def _circular_hz_distance(hz_a, hz_b, rate=RATE):
    d = abs(hz_a - hz_b) % rate
    return min(d, rate - d)


@pytest.fixture
def tri_bank():
    return TriangularOverlappingFilterBank(
        MelScaling(), num_filts=10, low_hz=100., high_hz=7000.,
        sampling_rate=RATE)


@pytest.fixture
def tri_analytic_bank():
    return TriangularOverlappingFilterBank(
        MelScaling(), num_filts=10, low_hz=100., high_hz=7000.,
        sampling_rate=RATE, analytic=True)


@pytest.fixture
def fbank_bank():
    return Fbank(num_filts=10, low_hz=100., high_hz=7000., sampling_rate=RATE)


@pytest.fixture
def fbank_analytic_bank():
    return Fbank(
        num_filts=10, low_hz=100., high_hz=7000., sampling_rate=RATE,
        analytic=True)


@pytest.fixture
def gabor_bank():
    return GaborFilterBank(
        MelScaling(), num_filts=10, low_hz=100., high_hz=7000.,
        sampling_rate=RATE)


@pytest.fixture
def gammatone_bank():
    return ComplexGammatoneFilterBank(
        MelScaling(), num_filts=8, low_hz=100., high_hz=7000.,
        sampling_rate=RATE, order=4)


# ---------------------------------------------------------------------------
# TriangularOverlappingFilterBank
# ---------------------------------------------------------------------------


def test_tri_is_linear_bank_with_expected_layout(tri_bank):
    assert isinstance(tri_bank, LinearFilterBank)
    assert tri_bank.num_filts == 10
    centers = np.asarray(tri_bank.centers_hz)
    assert centers.shape == (10,)
    assert np.all(np.diff(centers) > 0)
    assert tri_bank.is_real
    assert not tri_bank.is_analytic
    assert tri_bank.is_zero_phase


def test_tri_invalid_frequency_range_raises():
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), low_hz=-1.)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(
            MelScaling(), low_hz=1000., high_hz=500., sampling_rate=RATE)


@pytest.mark.parametrize(
    'width,half,expected_len',
    [(512, False, 512), (512, True, 257), (511, False, 511),
     (511, True, 256)])
def test_tri_frequency_response_shape_and_dtype(
        tri_bank, width, half, expected_len):
    res = tri_bank.get_frequency_response(0, width, half=half)
    assert res.shape == (expected_len,)
    assert res.dtype == np.float64


def test_tri_real_frequency_response_is_hermitian_symmetric(tri_bank):
    idx = np.arange(WIDTH)
    for filt_idx in range(tri_bank.num_filts):
        full = tri_bank.get_frequency_response(filt_idx, WIDTH)
        assert np.allclose(full, full[(-idx) % WIDTH])


def test_tri_frequency_response_values_in_unit_interval(tri_bank):
    for filt_idx in range(tri_bank.num_filts):
        res = tri_bank.get_frequency_response(filt_idx, WIDTH)
        assert np.all(res >= -1e-12)
        assert np.all(res <= 1 + 1e-12)
        assert np.max(res) > 0


def test_tri_half_response_is_prefix_of_full_response(tri_bank):
    for filt_idx in range(tri_bank.num_filts):
        full = tri_bank.get_frequency_response(filt_idx, WIDTH)
        half = tri_bank.get_frequency_response(filt_idx, WIDTH, half=True)
        assert np.allclose(half, full[:len(half)])


def test_tri_frequency_response_zero_outside_support(tri_bank):
    half_bins = np.arange(257)
    hz = _bin_hz(half_bins)
    for filt_idx in range(tri_bank.num_filts):
        half = tri_bank.get_frequency_response(filt_idx, WIDTH, half=True)
        low, high = tri_bank.supports_hz[filt_idx]
        outside = (hz < low) | (hz > high)
        assert np.allclose(half[outside], 0, atol=1e-9)


@pytest.mark.parametrize('width', [512, 511])
def test_tri_truncated_response_matches_half_frequency_response(
        tri_bank, width):
    for filt_idx in range(tri_bank.num_filts):
        bin_idx, trnc = tri_bank.get_truncated_response(filt_idx, width)
        half = tri_bank.get_frequency_response(filt_idx, width, half=True)
        assert trnc.dtype == np.float64
        assert np.allclose(half[bin_idx:bin_idx + len(trnc)], trnc)
        assert np.allclose(half[:bin_idx], 0)
        assert np.allclose(half[bin_idx + len(trnc):], 0)


def test_tri_truncated_bin_index_is_ceil_of_low_edge(tri_bank):
    for filt_idx in range(tri_bank.num_filts):
        low, _ = tri_bank.supports_hz[filt_idx]
        bin_idx, _ = tri_bank.get_truncated_response(filt_idx, WIDTH)
        assert bin_idx == int(np.ceil(WIDTH * low / RATE))


def test_tri_analytic_response_has_no_negative_frequencies(tri_analytic_bank):
    assert tri_analytic_bank.is_analytic
    assert not tri_analytic_bank.is_real
    for filt_idx in range(tri_analytic_bank.num_filts):
        full = tri_analytic_bank.get_frequency_response(filt_idx, WIDTH)
        assert np.all(full[WIDTH // 2 + 1:] == 0)
        half = tri_analytic_bank.get_frequency_response(
            filt_idx, WIDTH, half=True)
        assert np.allclose(half, full[:len(half)])


# ---------------------------------------------------------------------------
# Fbank
# ---------------------------------------------------------------------------


def test_fbank_is_real_and_not_analytic_by_default(fbank_bank):
    assert isinstance(fbank_bank, LinearFilterBank)
    assert fbank_bank.is_real
    assert not fbank_bank.is_analytic
    assert fbank_bank.num_filts == 10


def test_fbank_frequency_response_symmetric_and_bounded(fbank_bank):
    idx = np.arange(WIDTH)
    for filt_idx in range(fbank_bank.num_filts):
        full = fbank_bank.get_frequency_response(filt_idx, WIDTH)
        assert full.dtype == np.float64
        assert np.allclose(full, full[(-idx) % WIDTH])
        assert np.all(full >= -1e-12)
        assert np.all(full <= 1 + 1e-12)


def test_fbank_truncated_response_matches_half_frequency_response(fbank_bank):
    for filt_idx in range(fbank_bank.num_filts):
        bin_idx, trnc = fbank_bank.get_truncated_response(filt_idx, WIDTH)
        half = fbank_bank.get_frequency_response(filt_idx, WIDTH, half=True)
        assert np.allclose(half[bin_idx:bin_idx + len(trnc)], trnc)
        assert np.allclose(half[:bin_idx], 0)
        assert np.allclose(half[bin_idx + len(trnc):], 0)


def test_fbank_analytic_response_has_no_negative_frequencies(
        fbank_analytic_bank):
    assert fbank_analytic_bank.is_analytic
    for filt_idx in range(fbank_analytic_bank.num_filts):
        full = fbank_analytic_bank.get_frequency_response(filt_idx, WIDTH)
        assert np.all(full[WIDTH // 2 + 1:] == 0)


def test_fbank_impulse_response_is_real_for_real_bank(fbank_bank):
    out = fbank_bank.get_impulse_response(0, WIDTH)
    assert out.shape == (WIDTH,)
    assert np.isrealobj(out)


# ---------------------------------------------------------------------------
# GaborFilterBank
# ---------------------------------------------------------------------------


def test_gabor_properties(gabor_bank):
    assert isinstance(gabor_bank, LinearFilterBank)
    assert not gabor_bank.is_real
    assert gabor_bank.is_zero_phase
    assert gabor_bank.num_filts == 10
    assert np.all(np.diff(np.asarray(gabor_bank.centers_hz)) > 0)


@pytest.mark.parametrize('half,expected_len', [(False, 512), (True, 257)])
def test_gabor_frequency_response_shape_and_dtype(
        gabor_bank, half, expected_len):
    res = gabor_bank.get_frequency_response(1, WIDTH, half=half)
    assert res.shape == (expected_len,)
    assert res.dtype == np.float64
    assert np.all(res >= 0)


def test_gabor_frequency_response_peaks_near_center(gabor_bank):
    for filt_idx in range(gabor_bank.num_filts):
        res = gabor_bank.get_frequency_response(filt_idx, WIDTH)
        peak_hz = _bin_hz(int(np.argmax(res)))
        assert abs(peak_hz - gabor_bank.centers_hz[filt_idx]) <= RATE / WIDTH


def test_gabor_truncated_response_matches_frequency_response(gabor_bank):
    checked = 0
    for filt_idx in range(gabor_bank.num_filts):
        # filters whose support wraps below zero frequency are excluded
        # from this comparison, as their periodization differs by design
        if gabor_bank.supports_hz[filt_idx][0] < 0:
            continue
        bin_idx, trnc = gabor_bank.get_truncated_response(filt_idx, WIDTH)
        full = gabor_bank.get_frequency_response(filt_idx, WIDTH)
        wrapped = full[(bin_idx + np.arange(len(trnc))) % WIDTH]
        assert np.allclose(
            wrapped, trnc, atol=10 * config.EFFECTIVE_SUPPORT_THRESHOLD)
        checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# ComplexGammatoneFilterBank
# ---------------------------------------------------------------------------


def test_gammatone_invalid_order_raises():
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=0)


@needs_np_math
def test_gammatone_properties(gammatone_bank):
    assert isinstance(gammatone_bank, LinearFilterBank)
    assert not gammatone_bank.is_real
    assert not gammatone_bank.is_zero_phase
    assert gammatone_bank.order == 4
    assert gammatone_bank.num_filts == 8


@needs_np_math
@pytest.mark.parametrize('half,expected_len', [(False, 512), (True, 257)])
def test_gammatone_frequency_response_shape_and_dtype(
        gammatone_bank, half, expected_len):
    res = gammatone_bank.get_frequency_response(2, WIDTH, half=half)
    assert res.shape == (expected_len,)
    assert np.iscomplexobj(res)


@needs_np_math
def test_gammatone_truncated_response_peaks_near_center(gammatone_bank):
    for filt_idx in range(gammatone_bank.num_filts):
        bin_idx, trnc = gammatone_bank.get_truncated_response(filt_idx, WIDTH)
        assert 0 <= bin_idx < WIDTH
        assert len(trnc) > 0
        assert np.iscomplexobj(trnc)
        k = (bin_idx + int(np.argmax(np.abs(trnc)))) % WIDTH
        dist = _circular_hz_distance(
            _bin_hz(k), gammatone_bank.centers_hz[filt_idx])
        assert dist <= RATE / WIDTH * 1.01
