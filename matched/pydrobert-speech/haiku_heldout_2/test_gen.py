import numpy as np
import pytest

from pydrobert.speech.filters import ComplexGammatoneFilterBank
from pydrobert.speech.filters import Fbank
from pydrobert.speech.filters import GaborFilterBank
from pydrobert.speech.filters import LinearFilterBank
from pydrobert.speech.filters import TriangularOverlappingFilterBank
from pydrobert.speech.scales import MelScaling

WIDTH = 512
RATE = 16000
COMMON = dict(num_filts=20, high_hz=7000., low_hz=100., sampling_rate=RATE)


def make_bank(name):
    if name == 'tri':
        return TriangularOverlappingFilterBank(MelScaling(), **COMMON)
    if name == 'fbank':
        return Fbank(**COMMON)
    if name == 'gabor':
        return GaborFilterBank(MelScaling(), **COMMON)
    if name == 'gammatone':
        return ComplexGammatoneFilterBank(MelScaling(), **COMMON)
    raise ValueError(name)


ALL_BANKS = ['tri', 'fbank', 'gabor', 'gammatone']


@pytest.fixture(params=ALL_BANKS)
def bank(request):
    return make_bank(request.param)


def test_num_filts_and_centers_increasing(bank):
    assert isinstance(bank, LinearFilterBank)
    assert bank.num_filts == 20
    centers = np.asarray(bank.centers_hz)
    assert len(centers) == 20
    assert np.all(np.diff(centers) > 0)


def test_frequency_response_lengths(bank):
    for idx in (0, bank.num_filts - 1):
        full = bank.get_frequency_response(idx, WIDTH)
        half = bank.get_frequency_response(idx, WIDTH, half=True)
        assert full.shape == (WIDTH,)
        assert half.shape == (WIDTH // 2 + 1,)


def test_half_length_for_odd_width(bank):
    half = bank.get_frequency_response(0, 511, half=True)
    full = bank.get_frequency_response(0, 511)
    assert half.shape == (256,)
    assert full.shape == (511,)


def test_half_response_is_prefix_of_full(bank):
    for idx in (0, bank.num_filts // 2, bank.num_filts - 1):
        full = bank.get_frequency_response(idx, WIDTH)
        half = bank.get_frequency_response(idx, WIDTH, half=True)
        np.testing.assert_allclose(half, full[:len(half)], atol=1e-10)


def test_frequency_response_dtype(bank):
    res = bank.get_frequency_response(0, WIDTH)
    expected = np.float64 if bank.is_zero_phase else np.complex128
    assert res.dtype == expected


@pytest.mark.parametrize('name', ['tri', 'fbank'])
def test_real_filter_full_response_is_symmetric(name):
    bank = make_bank(name)
    full = bank.get_frequency_response(bank.num_filts // 2, WIDTH)
    k = np.arange(1, WIDTH // 2 + 1)
    np.testing.assert_allclose(full[k], full[WIDTH - k], atol=1e-10)


@pytest.mark.parametrize('name', ['tri', 'fbank'])
def test_truncated_response_matches_half_spectrum(name):
    bank = make_bank(name)
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, WIDTH)
        half = bank.get_frequency_response(idx, WIDTH, half=True)
        assert isinstance(bin_idx, int)
        assert trnc.dtype == np.float64
        assert bin_idx + len(trnc) <= len(half)
        embedded = np.zeros_like(half)
        embedded[bin_idx:bin_idx + len(trnc)] = trnc
        np.testing.assert_allclose(embedded, half, atol=1e-10)


def test_gabor_truncated_response_matches_full():
    bank = make_bank('gabor')
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, WIDTH)
        bin_idx, trnc = bank.get_truncated_response(idx, WIDTH)
        assert isinstance(bin_idx, int)
        assert 0 <= bin_idx < WIDTH
        for i, val in enumerate(trnc):
            np.testing.assert_allclose(
                full[(bin_idx + i) % WIDTH], val, atol=1e-6)


def test_triangular_truncated_length_matches_support():
    bank = make_bank('tri')
    for idx, (low, high) in enumerate(bank.supports_hz):
        bin_idx, trnc = bank.get_truncated_response(idx, WIDTH)
        left = int(np.ceil(WIDTH * low / RATE))
        right = int(WIDTH * high / RATE)
        assert bin_idx == left
        assert len(trnc) == 1 + right - left


def test_triangular_analytic_has_no_negative_frequencies():
    bank = TriangularOverlappingFilterBank(
        MelScaling(), analytic=True, **COMMON)
    assert bank.is_analytic
    assert not bank.is_real
    for idx in range(bank.num_filts):
        res = bank.get_frequency_response(idx, WIDTH)
        assert np.all(res[WIDTH // 2 + 1:] == 0)
        assert np.any(res != 0)


@pytest.mark.parametrize(
    'low_hz,high_hz', [(-1., 4000.), (1000., 500.), (100., 9000.)])
def test_triangular_invalid_frequency_range(low_hz, high_hz):
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(
            MelScaling(), num_filts=5, high_hz=high_hz, low_hz=low_hz,
            sampling_rate=RATE)


def test_triangular_default_high_hz_is_nyquist():
    bank = TriangularOverlappingFilterBank(
        MelScaling(), num_filts=5, low_hz=100., sampling_rate=RATE)
    _, high = bank.supports_hz[-1]
    assert high == pytest.approx(RATE / 2, rel=1e-6)


def test_supports_ms_scales_supports(bank):
    supports = bank.supports
    supports_ms = bank.supports_ms
    assert len(supports) == len(supports_ms) == bank.num_filts
    for (s0, s1), (m0, m1) in zip(supports, supports_ms):
        assert m0 == pytest.approx(s0 * 1000 / RATE)
        assert m1 == pytest.approx(s1 * 1000 / RATE)


def test_fbank_impulse_response_is_real_and_sized():
    bank = make_bank('fbank')
    ir = bank.get_impulse_response(3, WIDTH)
    assert ir.shape == (WIDTH,)
    assert np.isrealobj(ir)
    assert np.all(np.isfinite(ir))


def test_gabor_impulse_response_is_complex_and_sized():
    bank = make_bank('gabor')
    ir = bank.get_impulse_response(0, WIDTH)
    assert ir.shape == (WIDTH,)
    assert ir.dtype == np.complex128


def test_gabor_flags_and_nonnegative_response():
    bank = GaborFilterBank(
        MelScaling(), scale_l2_norm=True, erb=True, **COMMON)
    assert bank.scaled_l2_norm
    assert bank.erb
    assert not bank.is_real
    assert bank.is_zero_phase
    for idx in range(bank.num_filts):
        res = bank.get_frequency_response(idx, WIDTH)
        assert np.all(res >= 0)


@pytest.mark.parametrize('order', [0, -1, 2.5])
def test_gammatone_invalid_order_raises(order):
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(
            MelScaling(), order=order, **COMMON)


def test_gammatone_properties():
    bank = make_bank('gammatone')
    assert bank.order == 4
    assert not bank.is_zero_phase
    assert not bank.is_real


def test_gammatone_truncated_response_type():
    bank = make_bank('gammatone')
    bin_idx, trnc = bank.get_truncated_response(5, WIDTH)
    assert isinstance(bin_idx, int)
    assert 0 <= bin_idx < WIDTH
    assert trnc.dtype == np.complex128
    assert len(trnc) >= 1
