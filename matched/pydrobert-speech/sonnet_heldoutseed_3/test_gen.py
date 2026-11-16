import numpy as np
import pytest

from pydrobert.speech import config
from pydrobert.speech.filters import ComplexGammatoneFilterBank
from pydrobert.speech.filters import Fbank
from pydrobert.speech.filters import GaborFilterBank
from pydrobert.speech.filters import TriangularOverlappingFilterBank
from pydrobert.speech.scales import MelScaling

THRESH = config.EFFECTIVE_SUPPORT_THRESHOLD


def _tri(**kwargs):
    kwargs.setdefault("num_filts", 10)
    return TriangularOverlappingFilterBank(MelScaling(), **kwargs)


def _fbank(**kwargs):
    kwargs.setdefault("num_filts", 10)
    return Fbank(**kwargs)


def _gabor(**kwargs):
    kwargs.setdefault("num_filts", 20)
    return GaborFilterBank(MelScaling(), **kwargs)


def _gammatone(**kwargs):
    kwargs.setdefault("num_filts", 20)
    return ComplexGammatoneFilterBank(MelScaling(), **kwargs)


def _half_len(width):
    return width // 2 + 1 if width % 2 == 0 else (width + 1) // 2


def _full_from_truncated(bin_idx, trnc, width):
    full = np.zeros(width, dtype=trnc.dtype)
    wrap = min(bin_idx + len(trnc), width) - bin_idx
    full[bin_idx:bin_idx + wrap] = trnc[:wrap]
    full[:len(trnc) - wrap] = trnc[wrap:]
    return full


def _check_truncated(bank, filt_idx, width, atol):
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    assert 0 <= bin_idx < width
    if bank.is_real:
        half_width = (width + width % 2) // 2 + 1 - width % 2
        assert half_width == _half_len(width)
        half = np.zeros(half_width, dtype=trnc.dtype)
        half[bin_idx:bin_idx + len(trnc)] = trnc
        expected = bank.get_frequency_response(filt_idx, width, half=True)
        np.testing.assert_allclose(half, expected, atol=atol)
    else:
        full = _full_from_truncated(bin_idx, trnc, width)
        expected = bank.get_frequency_response(filt_idx, width)
        np.testing.assert_allclose(full, expected, atol=atol)


ALL_BANKS = [
    pytest.param(_tri, id="triangular"),
    pytest.param(lambda: _tri(analytic=True), id="triangular-analytic"),
    pytest.param(_fbank, id="fbank"),
    pytest.param(_gabor, id="gabor"),
    pytest.param(_gammatone, id="gammatone"),
]


def test_triangular_properties():
    bank = _tri(num_filts=12)
    assert bank.num_filts == 12
    assert bank.sampling_rate == 16000
    assert bank.is_real
    assert not bank.is_analytic
    assert bank.is_zero_phase
    centers = bank.centers_hz
    assert len(centers) == 12
    assert all(b > a for a, b in zip(centers, centers[1:]))
    assert len(bank.supports_hz) == 12
    for (low, high), center in zip(bank.supports_hz, centers):
        assert low < center < high
    assert bank.supports_hz[0][0] == pytest.approx(20.0, rel=1e-4)
    assert bank.supports_hz[-1][1] == pytest.approx(8000.0, rel=1e-4)
    assert len(bank.supports) == 12
    for left, right in bank.supports:
        assert left < 0 < right
        assert left == -right

    analytic = _tri(analytic=True, sampling_rate=8000)
    assert analytic.is_analytic
    assert not analytic.is_real
    assert analytic.supports_hz[-1][1] == pytest.approx(4000.0, rel=1e-4)


@pytest.mark.parametrize("low_hz,high_hz", [
    (-1.0, 4000.0),
    (100.0, 50.0),
    (100.0, 100.0),
    (20.0, 9000.0),
])
@pytest.mark.parametrize("factory", [
    lambda low, high: TriangularOverlappingFilterBank(
        MelScaling(), low_hz=low, high_hz=high),
    lambda low, high: GaborFilterBank(
        MelScaling(), low_hz=low, high_hz=high),
    lambda low, high: ComplexGammatoneFilterBank(
        MelScaling(), low_hz=low, high_hz=high),
    lambda low, high: Fbank(low_hz=low, high_hz=high),
])
def test_invalid_frequency_ranges_raise(factory, low_hz, high_hz):
    with pytest.raises(ValueError):
        factory(low_hz, high_hz)


@pytest.mark.parametrize("width", [128, 255, 256])
def test_triangular_frequency_response_shape_and_symmetry(width):
    bank = _tri()
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        assert full.shape == (width,)
        assert full.dtype == np.float64
        assert np.all(full >= 0)
        assert np.all(full <= 1.0 + 1e-12)
        # real filters are conjugate (here: even) symmetric
        np.testing.assert_allclose(full[1:], full[1:][::-1], atol=1e-12)
        half = bank.get_frequency_response(idx, width, half=True)
        assert half.shape == (_half_len(width),)


def test_triangular_frequency_response_peaks_near_center():
    width = 1024
    bank = _tri()
    for idx in (2, 5, 7):
        resp = bank.get_frequency_response(idx, width, half=True)
        expected_bin = bank.centers_hz[idx] * width / bank.sampling_rate
        assert abs(int(np.argmax(resp)) - expected_bin) <= 1.0
        assert resp.max() > 0.8
        assert resp.max() <= 1.0 + 1e-12


def test_triangular_response_zero_outside_supports_hz():
    width = 512
    bank = _tri()
    freqs = np.arange(width) * bank.sampling_rate / width
    for idx in range(bank.num_filts):
        resp = bank.get_frequency_response(idx, width)
        low, high = bank.supports_hz[idx]
        # the response is mirrored for real filters
        folded = np.minimum(freqs, bank.sampling_rate - freqs)
        outside = (folded < low - 1e-6) | (folded > high + 1e-6)
        assert np.all(resp[outside] == 0)


def test_triangular_analytic_has_no_negative_frequencies():
    width = 256
    real = _tri()
    analytic = _tri(analytic=True)
    for idx in range(analytic.num_filts):
        resp = analytic.get_frequency_response(idx, width)
        assert np.all(resp[width // 2 + 1:] == 0)
        np.testing.assert_allclose(
            resp[:width // 2 + 1],
            real.get_frequency_response(idx, width)[:width // 2 + 1],
            atol=1e-12)


@pytest.mark.parametrize("width", [200, 201, 512])
@pytest.mark.parametrize("analytic", [False, True])
def test_triangular_truncated_matches_frequency_response(width, analytic):
    bank = _tri(analytic=analytic)
    for idx in range(bank.num_filts):
        _check_truncated(bank, idx, width, atol=1e-12)
        _, trnc = bank.get_truncated_response(idx, width)
        assert trnc.max() > 0


def test_triangular_impulse_response_types_and_symmetry():
    width = 64
    real = _tri()
    ir = real.get_impulse_response(3, width)
    assert ir.shape == (width,)
    assert ir.dtype == np.float64
    np.testing.assert_allclose(ir[1:], ir[1:][::-1], atol=1e-12)

    analytic = _tri(analytic=True)
    ir = analytic.get_impulse_response(3, width)
    assert ir.shape == (width,)
    assert ir.dtype == np.complex128


@pytest.mark.parametrize("factory", ALL_BANKS)
def test_supports_ms_matches_supports(factory):
    bank = factory()
    supports = bank.supports
    supports_ms = bank.supports_ms
    assert len(supports_ms) == len(supports) == bank.num_filts
    for (s0, s1), (m0, m1) in zip(supports, supports_ms):
        assert m0 == pytest.approx(s0 * 1000 / bank.sampling_rate)
        assert m1 == pytest.approx(s1 * 1000 / bank.sampling_rate)


@pytest.mark.parametrize("width", [100, 101])
@pytest.mark.parametrize("factory", ALL_BANKS)
def test_half_response_is_prefix_of_full_response(factory, width):
    bank = factory()
    for idx in (0, bank.num_filts // 2, bank.num_filts - 1):
        full = bank.get_frequency_response(idx, width)
        half = bank.get_frequency_response(idx, width, half=True)
        assert full.shape == (width,)
        assert half.shape == (_half_len(width),)
        assert half.dtype == full.dtype
        np.testing.assert_allclose(half, full[:len(half)], atol=1e-12)


def test_fbank_properties_and_response():
    bank = _fbank()
    assert bank.num_filts == 10
    assert bank.is_real
    assert not bank.is_analytic
    assert bank.is_zero_phase
    assert len(bank.centers_hz) == 10
    assert all(b > a for a, b in zip(bank.centers_hz, bank.centers_hz[1:]))
    width = 1024
    for idx in (1, 4, 8):
        full = bank.get_frequency_response(idx, width)
        assert full.dtype == np.float64
        assert np.all(full >= 0)
        assert np.all(full <= 1.0 + 1e-12)
        np.testing.assert_allclose(full[1:], full[1:][::-1], atol=1e-12)
        expected_bin = bank.centers_hz[idx] * width / bank.sampling_rate
        assert abs(int(np.argmax(full[:width // 2 + 1])) - expected_bin) <= 1
    analytic = _fbank(analytic=True)
    assert analytic.is_analytic
    resp = analytic.get_frequency_response(3, 256)
    assert np.all(resp[129:] == 0)


@pytest.mark.parametrize("width", [256, 255])
def test_fbank_truncated_and_impulse_response(width):
    bank = _fbank()
    for idx in range(bank.num_filts):
        _check_truncated(bank, idx, width, atol=1e-12)
    ir = bank.get_impulse_response(4, width)
    assert ir.shape == (width,)
    assert ir.dtype == np.float64
    # impulse response is the inverse DFT of the frequency response
    np.testing.assert_allclose(
        np.fft.fft(ir), bank.get_frequency_response(4, width), atol=1e-10)


def test_gabor_properties():
    bank = _gabor(scale_l2_norm=True, erb=True)
    assert bank.num_filts == 20
    assert not bank.is_real
    assert bank.is_zero_phase
    assert bank.scaled_l2_norm
    assert bank.erb
    assert isinstance(bank.is_analytic, bool)
    centers = bank.centers_hz
    assert len(centers) == 20
    assert all(b > a for a, b in zip(centers, centers[1:]))
    for (low, high), center in zip(bank.supports_hz, centers):
        assert low < center < high
    for left, right in bank.supports:
        assert left < 0 < right
        assert left == -right
    plain = _gabor()
    assert not plain.scaled_l2_norm
    assert not plain.erb


def test_gabor_frequency_response_peaks_at_center():
    width = 1024
    bank = _gabor()
    for idx in (5, 10, 15):
        resp = bank.get_frequency_response(idx, width)
        assert resp.dtype == np.float64
        expected_bin = bank.centers_hz[idx] * width / bank.sampling_rate
        assert abs(int(np.argmax(resp)) - expected_bin) <= 1
        assert 0.9 < resp.max() <= 1.0 + 1e-6


def test_gabor_response_small_outside_supports_hz():
    width = 512
    bank = _gabor()
    freqs = np.arange(width) * bank.sampling_rate / width
    idx = 10
    resp = bank.get_frequency_response(idx, width)
    low, high = bank.supports_hz[idx]
    outside = (freqs < low) | (freqs > high)
    assert outside.any()
    assert np.all(resp[outside] < 5 * THRESH)


@pytest.mark.parametrize("scale_l2_norm", [False, True])
def test_gabor_impulse_response_matches_inverse_of_frequency_response(
        scale_l2_norm):
    width = 2048
    bank = _gabor(scale_l2_norm=scale_l2_norm)
    ir = bank.get_impulse_response(10, width)
    assert ir.shape == (width,)
    assert ir.dtype == np.complex128
    freq = bank.get_frequency_response(10, width)
    np.testing.assert_allclose(np.fft.fft(ir), freq, atol=1e-6)


def test_gabor_scaled_l2_norm_gives_unit_energy_filters():
    bank = _gabor(scale_l2_norm=True)
    ir = bank.get_impulse_response(10, 4096)
    assert np.sum(np.abs(ir) ** 2) == pytest.approx(1.0, abs=1e-2)


@pytest.mark.parametrize("width", [512, 500])
def test_gabor_truncated_matches_frequency_response(width):
    bank = _gabor()
    for idx in (3, 10, 17):
        _check_truncated(bank, idx, width, atol=10 * THRESH)
        _, trnc = bank.get_truncated_response(idx, width)
        assert trnc.dtype == np.float64


@pytest.mark.parametrize("order", [0, -1, 2.5])
def test_gammatone_invalid_order_raises(order):
    with pytest.raises(ValueError):
        _gammatone(order=order)


def test_gammatone_properties_and_max_centered():
    bank = _gammatone(order=3)
    assert bank.order == 3
    assert bank.num_filts == 20
    assert not bank.is_real
    assert not bank.is_zero_phase
    assert not bank.scaled_l2_norm
    assert not bank.erb
    assert isinstance(bank.is_analytic, bool)
    centers = bank.centers_hz
    assert all(b > a for a, b in zip(centers, centers[1:]))
    for (low, high), center in zip(bank.supports_hz, centers):
        assert low < center < high
    # causal by default
    for left, right in bank.supports:
        assert left == 0
        assert right > 0
    centered = _gammatone(max_centered=True)
    for left, right in centered.supports:
        assert left < 0 < right


def test_gammatone_frequency_response_peaks_at_center():
    width = 1024
    bank = _gammatone()
    for idx in (5, 10, 15):
        resp = bank.get_frequency_response(idx, width)
        assert resp.dtype == np.complex128
        mags = np.abs(resp)
        expected_bin = bank.centers_hz[idx] * width / bank.sampling_rate
        assert abs(int(np.argmax(mags)) - expected_bin) <= 1
        assert 0.9 < mags.max() <= 1.0 + 1e-3


def test_gammatone_impulse_response_matches_inverse_of_frequency_response():
    width = 2048
    bank = _gammatone()
    ir = bank.get_impulse_response(10, width)
    assert ir.shape == (width,)
    assert ir.dtype == np.complex128
    freq = bank.get_frequency_response(10, width)
    np.testing.assert_allclose(np.fft.fft(ir), freq, atol=1e-4)


@pytest.mark.parametrize("width", [512, 500])
@pytest.mark.parametrize("max_centered", [False, True])
def test_gammatone_truncated_matches_frequency_response(width, max_centered):
    bank = _gammatone(max_centered=max_centered)
    for idx in (3, 10, 17):
        _check_truncated(bank, idx, width, atol=10 * THRESH)
        _, trnc = bank.get_truncated_response(idx, width)
        assert trnc.dtype == np.complex128
