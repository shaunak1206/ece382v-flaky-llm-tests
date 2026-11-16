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

SR = 16000


def make_tri(**kwargs):
    kwargs.setdefault('num_filts', 10)
    return TriangularOverlappingFilterBank(MelScaling(), sampling_rate=SR, **kwargs)


def make_fbank(**kwargs):
    kwargs.setdefault('num_filts', 10)
    return Fbank(sampling_rate=SR, **kwargs)


def make_gabor(**kwargs):
    kwargs.setdefault('num_filts', 20)
    return GaborFilterBank(MelScaling(), sampling_rate=SR, **kwargs)


def make_gammatone(**kwargs):
    kwargs.setdefault('num_filts', 8)
    return ComplexGammatoneFilterBank(MelScaling(), sampling_rate=SR, **kwargs)


def expected_half_len(width):
    return width // 2 + 1 if width % 2 == 0 else (width + 1) // 2


def test_base_class_declares_abstract_interface():
    for name in ('get_impulse_response', 'get_frequency_response',
                 'get_truncated_response'):
        method = getattr(LinearFilterBank, name)
        assert getattr(method, '__isabstractmethod__', False)
    for name in ('is_real', 'is_analytic', 'is_zero_phase', 'num_filts',
                 'sampling_rate', 'supports_hz', 'supports'):
        prop = getattr(LinearFilterBank, name)
        assert getattr(prop, '__isabstractmethod__', False)


@pytest.mark.parametrize(
    'cls', [TriangularOverlappingFilterBank, Fbank, GaborFilterBank,
            ComplexGammatoneFilterBank])
def test_subclasses_implement_abstract_interface(cls):
    assert not getattr(cls, '__abstractmethods__', frozenset())


def test_triangular_counts_and_shapes():
    bank = make_tri(num_filts=10)
    assert bank.num_filts == 10
    assert len(bank.centers_hz) == 10
    assert len(bank.supports_hz) == 10
    assert len(bank.supports) == 10


def test_triangular_centers_evenly_spaced_on_scale():
    mel = MelScaling()
    bank = make_tri(num_filts=12)
    scaled = [mel.hertz_to_scale(c) for c in bank.centers_hz]
    diffs = np.diff(scaled)
    assert np.allclose(diffs, diffs[0], rtol=1e-6)


def test_triangular_real_and_analytic_flags():
    real = make_tri(analytic=False)
    assert real.is_real is True
    assert real.is_analytic is False
    assert real.is_zero_phase is True
    analytic = make_tri(analytic=True)
    assert analytic.is_real is False
    assert analytic.is_analytic is True


@pytest.mark.parametrize('factory', [
    lambda: make_tri(low_hz=-1.),
    lambda: make_tri(low_hz=1000., high_hz=500.),
    lambda: make_tri(high_hz=9000.),
    lambda: make_fbank(low_hz=-1.),
    lambda: make_gabor(high_hz=9000.),
])
def test_invalid_frequency_range_raises(factory):
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize('bank', [make_tri(), make_fbank()])
def test_real_response_bounded_in_unit_interval(bank):
    width = 4096
    for idx in range(bank.num_filts):
        resp = bank.get_frequency_response(idx, width)
        assert resp.shape == (width,)
        assert np.all(resp >= -1e-12)
        assert np.all(resp <= 1 + 1e-9)
        assert resp.max() > 0


@pytest.mark.parametrize('bank', [make_tri(), make_fbank()])
def test_real_full_response_is_conjugate_symmetric(bank):
    width = 512
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        assert np.allclose(full[1:], full[1:][::-1])


@pytest.mark.parametrize('width', [512, 511])
@pytest.mark.parametrize('bank', [make_tri(), make_fbank()])
def test_half_spectrum_lengths_and_values(bank, width):
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        half = bank.get_frequency_response(idx, width, half=True)
        assert len(half) == expected_half_len(width)
        assert np.allclose(half, full[:len(half)])


@pytest.mark.parametrize('bank', [make_tri(), make_fbank()])
def test_real_truncated_matches_half_spectrum(bank):
    width = 512
    half_len = expected_half_len(width)
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        assert len(trnc) >= 1
        assert bin_idx + len(trnc) <= half_len
        half = bank.get_frequency_response(idx, width, half=True)
        assert np.allclose(half[bin_idx:bin_idx + len(trnc)], trnc)
        outside = np.ones(half_len, dtype=bool)
        outside[bin_idx:bin_idx + len(trnc)] = False
        assert np.allclose(half[outside], 0)


def test_triangular_truncated_starts_at_left_edge_bin():
    width = 512
    bank = make_tri()
    for idx in range(bank.num_filts):
        left_hz = bank.supports_hz[idx][0]
        bin_idx, _ = bank.get_truncated_response(idx, width)
        assert bin_idx == int(np.ceil(width * left_hz / bank.sampling_rate))


def test_triangular_analytic_response_is_one_sided():
    width = 512
    bank = make_tri(analytic=True)
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        assert np.all(full[width // 2 + 1:] == 0)
        assert np.any(full != 0)


def test_fbank_basic_properties():
    mel = MelScaling()
    bank = make_fbank(num_filts=20)
    assert bank.num_filts == 20
    assert bank.is_real is True
    assert bank.is_analytic is False
    assert bank.is_zero_phase is True
    scaled = [mel.hertz_to_scale(c) for c in bank.centers_hz]
    diffs = np.diff(scaled)
    assert np.allclose(diffs, diffs[0], rtol=1e-6)


def test_gabor_basic_properties():
    bank = make_gabor(num_filts=20)
    assert bank.num_filts == 20
    assert len(bank.centers_hz) == 20
    assert len(bank.supports_hz) == 20
    assert len(bank.supports) == 20
    assert bank.is_real is False
    assert bank.is_zero_phase is True
    assert isinstance(bank.is_analytic, bool)
    centers = np.array(bank.centers_hz)
    assert np.all(np.diff(centers) > 0)
    for lo, hi in bank.supports:
        assert lo == -hi


def test_gabor_flags_follow_constructor_arguments():
    default = make_gabor()
    assert default.scaled_l2_norm is False
    assert default.erb is False
    custom = make_gabor(scale_l2_norm=True, erb=True)
    assert custom.scaled_l2_norm is True
    assert custom.erb is True


def test_gabor_frequency_response_peaks_near_one():
    width = 4096
    bank = make_gabor(scale_l2_norm=False)
    for idx in range(bank.num_filts):
        resp = bank.get_frequency_response(idx, width)
        assert resp.shape == (width,)
        assert 0.9 < resp.max() <= 1.0 + 1e-6


def test_gabor_truncated_matches_full_response():
    width = 512
    bank = make_gabor()
    checked = 0
    for idx in range(bank.num_filts):
        bin_idx, buf = bank.get_truncated_response(idx, width)
        if bin_idx + len(buf) <= width:
            full = bank.get_frequency_response(idx, width)
            assert np.allclose(full[bin_idx:bin_idx + len(buf)], buf,
                               atol=1e-8)
            checked += 1
    assert checked > 0


@pytest.mark.parametrize('bank', [make_gabor(), make_tri(), make_fbank()])
def test_supports_contain_zero(bank):
    for lo, hi in bank.supports:
        assert lo <= 0 <= hi


def test_supports_ms_scales_supports_by_sampling_rate():
    bank = make_tri()
    for s, ms in zip(bank.supports, bank.supports_ms):
        assert np.allclose(
            ms, (s[0] * 1000 / SR, s[1] * 1000 / SR))


def test_gammatone_flags_and_order():
    bank = make_gammatone(num_filts=8, order=3)
    assert bank.num_filts == 8
    assert bank.order == 3
    assert bank.is_real is False
    assert bank.is_zero_phase is False
    assert bank.sampling_rate == SR


@pytest.mark.parametrize('order', [0, -2, 2.5])
def test_gammatone_invalid_order_raises(order):
    with pytest.raises(ValueError):
        make_gammatone(order=order)


def test_gammatone_frequency_response_shapes_and_dtype():
    bank = make_gammatone()
    for width in (256, 255):
        full = bank.get_frequency_response(0, width)
        assert full.shape == (width,)
        assert full.dtype == np.complex128
        half = bank.get_frequency_response(0, width, half=True)
        assert half.shape == (expected_half_len(width),)
        assert half.dtype == np.complex128


def test_gammatone_truncated_response_is_complex_array():
    bank = make_gammatone()
    width = 256
    for idx in range(bank.num_filts):
        bin_idx, buf = bank.get_truncated_response(idx, width)
        assert isinstance(bin_idx, int)
        assert 0 <= bin_idx < width
        assert buf.ndim == 1
        assert len(buf) >= 1
        assert buf.dtype == np.complex128
