import numpy as np
import pytest

from pydrobert.speech import config
from pydrobert.speech.filters import ComplexGammatoneFilterBank
from pydrobert.speech.filters import Fbank
from pydrobert.speech.filters import GaborFilterBank
from pydrobert.speech.filters import TriangularOverlappingFilterBank
from pydrobert.speech.scales import MelScaling


RATE = 16000


def _tri(**kwargs):
    args = dict(num_filts=10, high_hz=7000, low_hz=100, sampling_rate=RATE)
    args.update(kwargs)
    return TriangularOverlappingFilterBank(MelScaling(), **args)


def _fbank(**kwargs):
    args = dict(num_filts=10, high_hz=7000, low_hz=100, sampling_rate=RATE)
    args.update(kwargs)
    return Fbank(**args)


def _gabor(**kwargs):
    args = dict(num_filts=8, high_hz=7000, low_hz=100, sampling_rate=RATE)
    args.update(kwargs)
    return GaborFilterBank(MelScaling(), **args)


def _gammatone(**kwargs):
    args = dict(
        num_filts=6, high_hz=4000, low_hz=100, sampling_rate=RATE, order=4)
    args.update(kwargs)
    return ComplexGammatoneFilterBank(MelScaling(), **args)


def _mask_from_truncated(bin_idx, trnc, width):
    # bins covered by a truncated response, including conjugate bins
    idx = (bin_idx + np.arange(len(trnc))) % width
    mask = np.zeros(width, dtype=bool)
    mask[idx] = True
    mask[(-idx) % width] = True
    return mask


def test_triangular_basic_properties():
    bank = _tri()
    assert bank.num_filts == 10
    assert bank.sampling_rate == RATE
    assert bank.is_real
    assert not bank.is_analytic
    assert bank.is_zero_phase
    centers = np.array(bank.centers_hz)
    assert len(centers) == 10
    assert np.all(np.diff(centers) > 0)
    assert np.all(centers > 100) and np.all(centers < 7000)
    for i, (lo, hi) in enumerate(bank.supports_hz):
        assert lo < centers[i] < hi


def test_triangular_supports_are_valid_sample_ranges():
    bank = _tri()
    supports = bank.supports
    assert len(supports) == bank.num_filts
    for left, right in supports:
        assert isinstance(left, int) and isinstance(right, int)
        assert left < right
    assert len(bank.supports_ms) == bank.num_filts


def test_triangular_full_response_shape_and_symmetry():
    bank = _tri()
    idx = np.arange(WIDTH) if False else np.arange(512)
    for filt in range(bank.num_filts):
        res = bank.get_frequency_response(filt, 512)
        assert res.shape == (512,)
        assert res.dtype == np.float64
        np.testing.assert_allclose(res, res[(-idx) % 512])


def test_triangular_half_response_lengths():
    bank = _tri()
    assert bank.get_frequency_response(0, 512, half=True).shape == (257,)
    assert bank.get_frequency_response(0, 511, half=True).shape == (256,)


def test_triangular_half_response_matches_full_prefix():
    bank = _tri()
    for width, dft in ((512, 257), (511, 256)):
        for filt in range(bank.num_filts):
            full = bank.get_frequency_response(filt, width)
            half = bank.get_frequency_response(filt, width, half=True)
            np.testing.assert_allclose(half, full[:dft])


def test_triangular_peak_reaches_one_at_center():
    bank = _tri()
    width = 8192
    for filt, center in enumerate(bank.centers_hz):
        res = bank.get_frequency_response(filt, width)
        bin_idx = int(round(center * width / RATE))
        assert res.max() <= 1 + 1e-9
        assert abs(res[bin_idx] - 1) < 0.05


def test_triangular_analytic_has_no_negative_frequencies():
    bank = _tri(analytic=True)
    assert not bank.is_real
    assert bank.is_analytic
    assert bank.is_zero_phase
    nonzero_found = False
    for filt in range(bank.num_filts):
        res = bank.get_frequency_response(filt, 512)
        assert np.all(res[257:] == 0)
        if np.any(res[1:257] != 0):
            nonzero_found = True
    assert nonzero_found


def test_triangular_truncated_matches_frequency_response():
    bank = _tri()
    width = 512
    for filt in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(filt, width)
        full = bank.get_frequency_response(filt, width)
        assert isinstance(bin_idx, (int, np.integer))
        assert trnc.ndim == 1 and len(trnc) >= 1
        np.testing.assert_allclose(
            trnc, full[(bin_idx + np.arange(len(trnc))) % width])


def test_triangular_truncated_support_is_exact():
    bank = _tri()
    width = 512
    for filt in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(filt, width)
        full = bank.get_frequency_response(filt, width)
        mask = _mask_from_truncated(bin_idx, trnc, width)
        assert np.all(full[~mask] == 0)


def test_triangular_invalid_frequency_range_raises():
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(
            MelScaling(), high_hz=100, low_hz=200, sampling_rate=RATE)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(
            MelScaling(), high_hz=4000, low_hz=-10, sampling_rate=RATE)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(
            MelScaling(), high_hz=9000, low_hz=100, sampling_rate=RATE)


def test_fbank_basic_properties():
    bank = _fbank()
    assert bank.num_filts == 10
    assert bank.sampling_rate == RATE
    assert bank.is_real
    assert not bank.is_analytic
    assert bank.is_zero_phase
    centers = np.array(bank.centers_hz)
    assert len(centers) == 10
    assert np.all(np.diff(centers) > 0)


def test_fbank_frequency_response_bounded_and_symmetric():
    bank = _fbank()
    idx = np.arange(512)
    for filt in range(bank.num_filts):
        res = bank.get_frequency_response(filt, 512)
        assert res.shape == (512,)
        assert res.dtype == np.float64
        assert np.all(res >= 0)
        assert np.all(res <= 1 + 1e-9)
        np.testing.assert_allclose(res, res[(-idx) % 512], atol=1e-12)


def test_fbank_half_response_odd_width_matches_prefix():
    bank = _fbank()
    for filt in range(bank.num_filts):
        full = bank.get_frequency_response(filt, 511)
        half = bank.get_frequency_response(filt, 511, half=True)
        assert half.shape == (256,)
        np.testing.assert_allclose(half, full[:256])


def test_fbank_truncated_matches_full_and_is_zero_outside():
    bank = _fbank()
    width = 512
    for filt in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(filt, width)
        full = bank.get_frequency_response(filt, width)
        assert np.all(trnc >= 0)
        np.testing.assert_allclose(
            trnc, full[(bin_idx + np.arange(len(trnc))) % width], atol=1e-12)
        mask = _mask_from_truncated(bin_idx, trnc, width)
        assert np.all(full[~mask] == 0)


def test_gabor_basic_properties():
    bank = _gabor()
    assert bank.num_filts == 8
    assert bank.sampling_rate == RATE
    assert not bank.is_real
    assert bank.is_zero_phase
    centers = np.array(bank.centers_hz)
    assert len(centers) == 8
    assert np.all(np.diff(centers) > 0)
    assert np.all(centers > 100) and np.all(centers < 7000)
    assert len(bank.supports) == 8


def test_gabor_frequency_response_peak_near_center():
    bank = _gabor()
    width = 4096
    for filt, center in enumerate(bank.centers_hz):
        res = bank.get_frequency_response(filt, width)
        assert res.dtype == np.float64
        assert np.all(res >= 0)
        bin_idx = int(round(center * width / RATE))
        assert res[bin_idx] > 0.9


def test_gabor_half_response_matches_full_prefix():
    bank = _gabor()
    for filt in range(bank.num_filts):
        full = bank.get_frequency_response(filt, 256)
        half = bank.get_frequency_response(filt, 256, half=True)
        assert half.shape == (129,)
        np.testing.assert_allclose(half, full[:129])


def test_gabor_truncated_matches_frequency_response():
    bank = _gabor()
    width = 1024
    eps = config.EFFECTIVE_SUPPORT_THRESHOLD
    for filt in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(filt, width)
        full = bank.get_frequency_response(filt, width)
        assert len(trnc) >= 1
        np.testing.assert_allclose(
            trnc, full[(bin_idx + np.arange(len(trnc))) % width],
            rtol=1e-4, atol=max(1e-6, 10 * eps))


def test_gammatone_basic_properties():
    bank = _gammatone()
    assert bank.num_filts == 6
    assert bank.order == 4
    assert bank.sampling_rate == RATE
    assert not bank.is_real
    assert not bank.is_zero_phase
    centers = np.array(bank.centers_hz)
    assert len(centers) == 6
    assert np.all(np.diff(centers) > 0)


def test_gammatone_half_response_shape_and_prefix():
    bank = _gammatone()
    for filt in range(bank.num_filts):
        full = bank.get_frequency_response(filt, 256)
        assert full.dtype == np.complex128
        assert full.shape == (256,)
        half = bank.get_frequency_response(filt, 255, half=True)
        assert half.shape == (128,)
        full_odd = bank.get_frequency_response(filt, 255)
        np.testing.assert_allclose(half, full_odd[:128])
        half_even = bank.get_frequency_response(filt, 256, half=True)
        np.testing.assert_allclose(half_even, full[:129])


def test_gammatone_truncated_matches_frequency_response():
    bank = _gammatone()
    width = 1024
    for filt in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(filt, width)
        full = bank.get_frequency_response(filt, width)
        assert trnc.dtype == np.complex128
        scale = np.abs(full).max()
        np.testing.assert_allclose(
            trnc, full[(bin_idx + np.arange(len(trnc))) % width],
            rtol=1e-3, atol=1e-4 * scale)


def test_gammatone_invalid_order_raises():
    with pytest.raises(ValueError):
        _gammatone(order=0)
    with pytest.raises(ValueError):
        _gammatone(order=-1)
    with pytest.raises(ValueError):
        _gammatone(order=2.5)
