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

EPS = config.EFFECTIVE_SUPPORT_THRESHOLD


def _half_width(width):
    return (width + width % 2) // 2 + 1 - width % 2


def _full_from_trunc_complex(bin_idx, trnc, width):
    full = np.zeros(width, dtype=trnc.dtype)
    wrap = min(bin_idx + len(trnc), width) - bin_idx
    full[bin_idx:bin_idx + wrap] = trnc[:wrap]
    full[:len(trnc) - wrap] = trnc[wrap:]
    return full


def _linear_bank(**kwargs):
    params = dict(
        num_filts=4, low_hz=100., high_hz=600., sampling_rate=8000)
    params.update(kwargs)
    return TriangularOverlappingFilterBank('linear', **params)


def test_triangular_basic_properties():
    np.random.seed(0)
    bank = TriangularOverlappingFilterBank('mel', num_filts=10)
    assert bank.num_filts == 10
    assert bank.sampling_rate == 16000
    assert bank.is_real
    assert not bank.is_analytic
    assert bank.is_zero_phase
    assert len(bank.centers_hz) == 10
    assert len(bank.supports_hz) == 10
    assert len(bank.supports) == 10
    centers = np.array(bank.centers_hz)
    assert np.all(np.diff(centers) > 0)
    for (low, high), center in zip(bank.supports_hz, bank.centers_hz):
        assert low < center < high
    assert bank.centers_hz[0] > 20.
    assert bank.supports_hz[-1][1] <= 8000. + 1e-6
    analytic = TriangularOverlappingFilterBank(
        'mel', num_filts=10, analytic=True)
    assert analytic.is_analytic
    assert not analytic.is_real


def test_triangular_linear_scale_vertices():
    np.random.seed(0)
    bank = _linear_bank()
    assert bank.num_filts == 4
    np.testing.assert_allclose(
        bank.centers_hz, [200., 300., 400., 500.], atol=1e-6)
    np.testing.assert_allclose(
        bank.supports_hz,
        [(100., 300.), (200., 400.), (300., 500.), (400., 600.)],
        atol=1e-6)


def test_triangular_scaling_function_instance_matches_string():
    np.random.seed(0)
    from_str = TriangularOverlappingFilterBank('mel', num_filts=7)
    from_obj = TriangularOverlappingFilterBank(MelScaling(), num_filts=7)
    np.testing.assert_allclose(from_str.centers_hz, from_obj.centers_hz)
    np.testing.assert_allclose(from_str.supports_hz, from_obj.supports_hz)


def test_triangular_invalid_frequency_range():
    np.random.seed(0)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank('mel', low_hz=-1.)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank('mel', low_hz=500., high_hz=400.)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(
            'mel', high_hz=9000., sampling_rate=16000)


def test_triangular_frequency_response_values():
    np.random.seed(0)
    bank = _linear_bank()
    width = 800  # 10 Hz per bin
    res = bank.get_frequency_response(0, width)
    assert res.shape == (width,)
    assert res.dtype == np.float64
    for idx, expected in [
            (0, 0.), (9, 0.), (12, .2), (15, .5), (20, 1.), (25, .5),
            (28, .2), (31, 0.), (400, 0.)]:
        assert res[idx] == pytest.approx(expected, abs=1e-6)
    # real filter: mirrored negative frequencies
    assert res[width - 20] == pytest.approx(1., abs=1e-6)
    assert res[width - 15] == pytest.approx(.5, abs=1e-6)
    res3 = bank.get_frequency_response(3, width)
    assert res3[50] == pytest.approx(1., abs=1e-6)
    assert res3[45] == pytest.approx(.5, abs=1e-6)
    assert res3[20] == pytest.approx(0., abs=1e-6)


@pytest.mark.parametrize('width', [512, 513])
def test_triangular_half_response_and_symmetry(width):
    np.random.seed(0)
    bank = TriangularOverlappingFilterBank('mel', num_filts=10)
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        half = bank.get_frequency_response(idx, width, half=True)
        assert full.shape == (width,)
        assert half.shape == (_half_width(width),)
        np.testing.assert_allclose(full[:len(half)], half, atol=1e-12)
        np.testing.assert_allclose(
            full[1:], full[1:][::-1], atol=1e-12)
        assert full.min() >= 0.
        assert full.max() <= 1. + 1e-9


def test_triangular_analytic_response_suppresses_negative_freqs():
    np.random.seed(0)
    real = TriangularOverlappingFilterBank('mel', num_filts=8)
    analytic = TriangularOverlappingFilterBank(
        'mel', num_filts=8, analytic=True)
    width = 256
    half_len = _half_width(width)
    for idx in range(real.num_filts):
        r = real.get_frequency_response(idx, width)
        a = analytic.get_frequency_response(idx, width)
        np.testing.assert_allclose(a[:half_len], r[:half_len], atol=1e-12)
        assert np.all(a[half_len:] == 0)


@pytest.mark.parametrize('width', [400, 401])
def test_triangular_truncated_response_matches_half(width):
    np.random.seed(0)
    bank = TriangularOverlappingFilterBank('mel', num_filts=10)
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        half = np.zeros(_half_width(width), dtype=trnc.dtype)
        half[bin_idx:bin_idx + len(trnc)] = trnc
        expected = bank.get_frequency_response(idx, width, half=True)
        np.testing.assert_allclose(half, expected, atol=1e-12)
        assert trnc.dtype == np.float64
        assert np.all(trnc >= 0)


def test_triangular_truncated_response_analytic():
    np.random.seed(0)
    bank = TriangularOverlappingFilterBank(
        'mel', num_filts=6, analytic=True)
    width = 300
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        full = _full_from_trunc_complex(bin_idx, trnc, width)
        expected = bank.get_frequency_response(idx, width)
        np.testing.assert_allclose(full, expected, atol=1e-12)


def test_triangular_supports_and_supports_ms():
    np.random.seed(0)
    bank = TriangularOverlappingFilterBank('mel', num_filts=6)
    supports = bank.supports
    assert len(supports) == 6
    for start, end in supports:
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert start < 0 < end
    ms = bank.supports_ms
    assert len(ms) == 6
    for (s0, s1), (m0, m1) in zip(supports, ms):
        assert m0 == pytest.approx(s0 * 1000 / 16000)
        assert m1 == pytest.approx(s1 * 1000 / 16000)
    for low, high in bank.supports_hz:
        assert low < high


def test_triangular_impulse_response_shape_and_symmetry():
    np.random.seed(0)
    width = 128
    real = _linear_bank()
    ir = real.get_impulse_response(1, width)
    assert ir.shape == (width,)
    assert ir.dtype == np.float64
    np.testing.assert_allclose(ir[1:], ir[1:][::-1], atol=1e-12)
    analytic = _linear_bank(analytic=True)
    air = analytic.get_impulse_response(1, width)
    assert air.shape == (width,)
    assert air.dtype == np.complex128


def test_fbank_properties_and_responses():
    np.random.seed(0)
    bank = Fbank(num_filts=8)
    assert bank.num_filts == 8
    assert bank.is_real
    assert not bank.is_analytic
    assert bank.is_zero_phase
    assert len(bank.centers_hz) == 8
    assert np.all(np.diff(bank.centers_hz) > 0)
    width = 512
    for idx in range(bank.num_filts - 1):
        full = bank.get_frequency_response(idx, width)
        half = bank.get_frequency_response(idx, width, half=True)
        assert full.shape == (width,)
        assert half.shape == (_half_width(width),)
        assert not np.any(np.isnan(full))
        assert full.min() >= 0.
        assert full.max() <= 1. + 1e-9
        np.testing.assert_allclose(full[:len(half)], half, atol=1e-12)
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        rebuilt = np.zeros(len(half))
        rebuilt[bin_idx:bin_idx + len(trnc)] = trnc
        np.testing.assert_allclose(rebuilt, half, atol=1e-12)
    ir = bank.get_impulse_response(2, width)
    assert ir.shape == (width,)
    assert ir.dtype == np.float64
    air = Fbank(num_filts=8, analytic=True).get_impulse_response(2, width)
    assert air.shape == (width,)
    assert air.dtype == np.complex128


def test_gabor_properties():
    np.random.seed(0)
    bank = GaborFilterBank('mel', num_filts=5, sampling_rate=8000)
    assert bank.num_filts == 5
    assert not bank.is_real
    assert bank.is_zero_phase
    assert bank.sampling_rate == 8000
    assert not bank.scaled_l2_norm
    assert not bank.erb
    centers = np.array(bank.centers_hz)
    assert len(centers) == 5
    assert np.all(np.diff(centers) > 0)
    assert centers[0] > 20. and centers[-1] < 4000.
    for (low, high), center in zip(bank.supports_hz, bank.centers_hz):
        assert low < center < high
    for start, end in bank.supports:
        assert start == -end
        assert end > 0
    assert bank.is_analytic == all(s[0] >= 0 for s in bank.supports_hz)
    bank2 = GaborFilterBank(
        'mel', num_filts=5, sampling_rate=8000, scale_l2_norm=True, erb=True)
    assert bank2.scaled_l2_norm
    assert bank2.erb


def test_gabor_invalid_frequency_range():
    np.random.seed(0)
    with pytest.raises(ValueError):
        GaborFilterBank('mel', low_hz=-5.)
    with pytest.raises(ValueError):
        GaborFilterBank('mel', low_hz=300., high_hz=200.)
    with pytest.raises(ValueError):
        GaborFilterBank('mel', high_hz=9000., sampling_rate=16000)


def test_gabor_frequency_response_peak_and_half():
    np.random.seed(0)
    rate = 8000
    width = 1024
    bank = GaborFilterBank('mel', num_filts=5, sampling_rate=rate)
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        assert full.shape == (width,)
        assert full.dtype == np.float64
        center_bin = int(round(bank.centers_hz[idx] * width / rate))
        assert abs(int(np.argmax(full)) - center_bin) <= 1
        assert 0.95 <= full.max() <= 1. + 1e-6
    half = bank.get_frequency_response(2, 256, half=True)
    full = bank.get_frequency_response(2, 256)
    assert half.shape == (_half_width(256),)
    np.testing.assert_allclose(full[:len(half)], half, atol=1e-12)
    half_odd = bank.get_frequency_response(2, 255, half=True)
    assert half_odd.shape == (_half_width(255),)


def test_gabor_truncated_response_matches_full():
    np.random.seed(0)
    width = 256
    bank = GaborFilterBank('mel', num_filts=6, sampling_rate=8000)
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        assert 0 <= bin_idx < width
        full = _full_from_trunc_complex(bin_idx, trnc, width)
        expected = bank.get_frequency_response(idx, width)
        np.testing.assert_allclose(full, expected, atol=5 * EPS)


def test_gabor_impulse_response_matches_frequency_response():
    np.random.seed(0)
    width = 512
    bank = GaborFilterBank('mel', num_filts=4, sampling_rate=8000)
    for idx in (0, 3):
        ir = bank.get_impulse_response(idx, width)
        assert ir.shape == (width,)
        assert ir.dtype == np.complex128
        spectrum = np.fft.fft(ir)
        expected = bank.get_frequency_response(idx, width)
        np.testing.assert_allclose(spectrum.real, expected, atol=1e-6)
        np.testing.assert_allclose(spectrum.imag, 0., atol=1e-6)


def test_gammatone_properties_and_invalid_args():
    np.random.seed(0)
    bank = ComplexGammatoneFilterBank(
        'mel', num_filts=4, sampling_rate=8000)
    assert bank.num_filts == 4
    assert bank.order == 4
    assert not bank.is_real
    assert not bank.is_zero_phase
    assert not bank.scaled_l2_norm
    assert not bank.erb
    assert np.all(np.diff(bank.centers_hz) > 0)
    assert len(bank.supports) == 4
    assert len(bank.supports_hz) == 4
    for low, high in bank.supports_hz:
        assert low < high
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank('mel', order=0)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank('mel', order=2.5)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank('mel', low_hz=-1.)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank('mel', low_hz=400., high_hz=300.)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(
            'mel', high_hz=9000., sampling_rate=16000)


def test_gammatone_frequency_response_peak_and_half():
    np.random.seed(0)
    rate = 8000
    width = 512
    bank = ComplexGammatoneFilterBank('mel', num_filts=4, sampling_rate=rate)
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        assert full.shape == (width,)
        assert full.dtype == np.complex128
        center_bin = int(round(bank.centers_hz[idx] * width / rate))
        assert 0.85 <= abs(full[center_bin]) <= 1.15
    for w in (128, 129):
        full = bank.get_frequency_response(1, w)
        half = bank.get_frequency_response(1, w, half=True)
        assert half.shape == (_half_width(w),)
        np.testing.assert_allclose(full[:len(half)], half, atol=1e-9)


def test_gammatone_truncated_response_matches_full():
    np.random.seed(0)
    width = 256
    bank = ComplexGammatoneFilterBank('mel', num_filts=5, sampling_rate=8000)
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        assert 0 <= bin_idx < width
        assert trnc.dtype == np.complex128
        full = _full_from_trunc_complex(bin_idx, trnc, width)
        expected = bank.get_frequency_response(idx, width)
        np.testing.assert_allclose(full, expected, atol=10 * EPS)


def test_gammatone_impulse_response_causal_and_matches_frequency_response():
    np.random.seed(0)
    width = 1024
    bank = ComplexGammatoneFilterBank('mel', num_filts=4, sampling_rate=8000)
    for idx in (0, 3):
        ir = bank.get_impulse_response(idx, width)
        assert ir.shape == (width,)
        assert ir.dtype == np.complex128
        assert abs(ir[0]) <= EPS
        expected = bank.get_frequency_response(idx, width)
        np.testing.assert_allclose(np.fft.fft(ir), expected, atol=0.05)


def test_gammatone_max_centered_supports_start_before_zero():
    np.random.seed(0)
    causal = ComplexGammatoneFilterBank(
        'mel', num_filts=4, sampling_rate=8000)
    centered = ComplexGammatoneFilterBank(
        'mel', num_filts=4, sampling_rate=8000, max_centered=True)
    for start, end in causal.supports:
        assert start == 0
        assert end > 0
    for start, end in centered.supports:
        assert start < 0
        assert end > start
    # frequency response magnitudes are unaffected by the shift
    a = causal.get_frequency_response(1, 128)
    b = centered.get_frequency_response(1, 128)
    np.testing.assert_allclose(np.abs(a), np.abs(b), atol=1e-9)
