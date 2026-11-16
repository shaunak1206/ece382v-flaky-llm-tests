import numpy as np
import pytest

from pydrobert.speech import config
from pydrobert.speech.filters import ComplexGammatoneFilterBank
from pydrobert.speech.filters import Fbank
from pydrobert.speech.filters import GaborFilterBank
from pydrobert.speech.filters import LinearFilterBank
from pydrobert.speech.filters import TriangularOverlappingFilterBank
from pydrobert.speech.scales import LinearScaling
from pydrobert.speech.scales import MelScaling

BANK_FACTORIES = {
    'tri': lambda: TriangularOverlappingFilterBank(MelScaling(), num_filts=10),
    'tri_analytic': lambda: TriangularOverlappingFilterBank(
        MelScaling(), num_filts=10, analytic=True),
    'fbank': lambda: Fbank(num_filts=10),
    'gabor': lambda: GaborFilterBank(MelScaling(), num_filts=10),
    'gammatone': lambda: ComplexGammatoneFilterBank(
        MelScaling(), num_filts=10),
}


def _reconstruct_wrapped(bin_idx, trnc, width):
    # reconstruction of a complex filter's full spectrum from its truncation
    full = np.zeros(width, dtype=trnc.dtype)
    wrap = min(bin_idx + len(trnc), width) - bin_idx
    full[bin_idx:bin_idx + wrap] = trnc[:wrap]
    full[:len(trnc) - wrap] = trnc[wrap:]
    return full


def test_triangular_properties():
    bank = TriangularOverlappingFilterBank(
        MelScaling(), num_filts=12, sampling_rate=8000, high_hz=3000.,
        low_hz=50.)
    assert bank.num_filts == 12
    assert bank.sampling_rate == 8000
    assert bank.is_real
    assert not bank.is_analytic
    assert bank.is_zero_phase
    assert len(bank.centers_hz) == 12
    assert all(a < b for a, b in zip(bank.centers_hz[:-1], bank.centers_hz[1:]))
    supports_hz = bank.supports_hz
    assert len(supports_hz) == 12
    assert supports_hz[0][0] == pytest.approx(50., rel=1e-6)
    assert supports_hz[-1][1] == pytest.approx(3000., rel=1e-6)
    for (low, high), center in zip(supports_hz, bank.centers_hz):
        assert low < center < high
    analytic = TriangularOverlappingFilterBank(
        MelScaling(), num_filts=12, analytic=True)
    assert analytic.is_analytic
    assert not analytic.is_real


@pytest.mark.parametrize('kwargs', [
    dict(high_hz=8001.),
    dict(high_hz=10., low_hz=20.),
    dict(high_hz=4000., low_hz=-5.),
])
def test_triangular_invalid_frequency_range(kwargs):
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), **kwargs)


def test_triangular_supports_and_supports_ms():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=8)
    supports = bank.supports
    assert len(supports) == 8
    for start, end in supports:
        assert start < 0 < end
        assert isinstance(start, int) and isinstance(end, int)
    supports_ms = bank.supports_ms
    assert len(supports_ms) == 8
    for (s, e), (sm, em) in zip(supports, supports_ms):
        assert sm == pytest.approx(s * 1000 / 16000)
        assert em == pytest.approx(e * 1000 / 16000)


def test_triangular_frequency_response_shape_and_peak():
    bank = TriangularOverlappingFilterBank(LinearScaling(), num_filts=10)
    width = 4096
    idx = 4
    res = bank.get_frequency_response(idx, width)
    assert res.shape == (width,)
    assert res.dtype == np.float64
    assert res.min() >= 0
    assert res.max() <= 1 + 1e-9
    assert res.max() > 0.95
    # nonzero region is inside the support
    low, high = bank.supports_hz[idx]
    nonzero = np.nonzero(res[:width // 2])[0]
    assert nonzero.min() * 16000 / width >= low - 16000 / width
    assert nonzero.max() * 16000 / width <= high + 16000 / width
    # peak close to center
    peak_hz = np.argmax(res[:width // 2]) * 16000 / width
    assert peak_hz == pytest.approx(bank.centers_hz[idx], abs=2 * 16000 / width)


def test_triangular_real_symmetric_and_analytic_one_sided():
    width = 128
    real_bank = TriangularOverlappingFilterBank(
        MelScaling(), num_filts=10)
    res = real_bank.get_frequency_response(5, width)
    assert np.allclose(res[1:], res[1:][::-1])
    assert res.max() > 0
    analytic = TriangularOverlappingFilterBank(
        MelScaling(), num_filts=10, analytic=True)
    ares = analytic.get_frequency_response(5, width)
    assert np.allclose(ares[:width // 2 + 1], res[:width // 2 + 1])
    assert np.all(ares[width // 2 + 1:] == 0)


@pytest.mark.parametrize('name', sorted(BANK_FACTORIES))
@pytest.mark.parametrize('width', [100, 101])
def test_half_frequency_response_length(name, width):
    bank = BANK_FACTORIES[name]()
    half = bank.get_frequency_response(3, width, half=True)
    full = bank.get_frequency_response(3, width, half=False)
    assert full.shape == (width,)
    if width % 2:
        assert half.shape == ((width + 1) // 2,)
    else:
        assert half.shape == (width // 2 + 1,)
    assert half.dtype == full.dtype
    assert np.allclose(half, full[:len(half)])


@pytest.mark.parametrize('filt_idx', [0, 5, 9])
@pytest.mark.parametrize('width', [100, 101, 256])
def test_triangular_truncated_matches_half_response(filt_idx, width):
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=10)
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    half_expected = bank.get_frequency_response(filt_idx, width, half=True)
    half_width = (width + width % 2) // 2 + 1 - width % 2
    assert len(half_expected) == half_width
    half = np.zeros(half_width, dtype=trnc.dtype)
    half[bin_idx:bin_idx + len(trnc)] = trnc
    assert np.allclose(half, half_expected)


def test_triangular_truncated_real_full_reconstruction():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=10)
    width = 100
    filt_idx = 5
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    assert bin_idx > 0
    full = np.zeros(width, dtype=trnc.dtype)
    full[bin_idx:bin_idx + len(trnc)] = trnc
    full[width - bin_idx - len(trnc) + 1:width - bin_idx + 1] = \
        trnc[::-1].conj()
    expected = bank.get_frequency_response(filt_idx, width)
    assert np.allclose(full, expected)


def test_triangular_analytic_truncated_response():
    bank = TriangularOverlappingFilterBank(
        MelScaling(), num_filts=10, analytic=True)
    width = 200
    for filt_idx in (0, 4, 8):
        bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
        full = np.zeros(width, dtype=trnc.dtype)
        full[bin_idx:bin_idx + len(trnc)] = trnc
        expected = bank.get_frequency_response(filt_idx, width)
        assert np.allclose(full, expected)
        assert trnc.max() > 0


def test_triangular_impulse_response_types():
    width = 64
    real_bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=5)
    ir = real_bank.get_impulse_response(2, width)
    assert ir.shape == (width,)
    assert ir.dtype == np.float64
    # symmetric (even) real filter
    assert np.allclose(ir[1:], ir[1:][::-1])
    analytic = TriangularOverlappingFilterBank(
        MelScaling(), num_filts=5, analytic=True)
    air = analytic.get_impulse_response(2, width)
    assert air.shape == (width,)
    assert air.dtype == np.complex128


def test_fbank_frequency_response_and_truncated():
    bank = Fbank(num_filts=10)
    assert bank.num_filts == 10
    assert bank.is_real
    assert bank.is_zero_phase
    width = 4096
    res = bank.get_frequency_response(5, width)
    assert res.dtype == np.float64
    assert res.min() >= 0
    assert res.max() <= 1 + 1e-9
    assert res.max() > 0.95
    for filt_idx in (0, 5, 9):
        for w in (200, 201):
            bin_idx, trnc = bank.get_truncated_response(filt_idx, w)
            half = bank.get_frequency_response(filt_idx, w, half=True)
            rebuilt = np.zeros(len(half), dtype=trnc.dtype)
            rebuilt[bin_idx:bin_idx + len(trnc)] = trnc
            assert np.allclose(rebuilt, half)


def test_fbank_impulse_response_inverts_frequency_response():
    bank = Fbank(num_filts=10)
    for width in (128, 129):
        ir = bank.get_impulse_response(4, width)
        assert ir.shape == (width,)
        assert not np.iscomplexobj(ir)
        half = bank.get_frequency_response(4, width, half=True)
        assert np.allclose(np.fft.rfft(ir), half, atol=1e-8)


def test_gabor_properties():
    bank = GaborFilterBank(MelScaling(), num_filts=20)
    assert bank.num_filts == 20
    assert not bank.is_real
    assert bank.is_zero_phase
    assert isinstance(bank.is_analytic, bool)
    assert bank.sampling_rate == 16000
    assert not bank.scaled_l2_norm
    assert not bank.erb
    centers = bank.centers_hz
    assert len(centers) == 20
    assert all(a < b for a, b in zip(centers[:-1], centers[1:]))
    for (start, end), (low, high), center in zip(
            bank.supports, bank.supports_hz, centers):
        assert start == -end
        assert end > 0
        assert low < center < high
    assert GaborFilterBank(
        MelScaling(), num_filts=5, scale_l2_norm=True, erb=True).erb


def test_gabor_invalid_frequency_range():
    with pytest.raises(ValueError):
        GaborFilterBank(MelScaling(), high_hz=9000.)
    with pytest.raises(ValueError):
        GaborFilterBank(MelScaling(), high_hz=100., low_hz=200.)


def test_gabor_frequency_response_peaks_at_center():
    bank = GaborFilterBank(MelScaling(), num_filts=40)
    width = 1024
    idx = 20
    res = bank.get_frequency_response(idx, width)
    assert res.dtype == np.float64
    assert res.shape == (width,)
    expected_bin = bank.centers_hz[idx] * width / bank.sampling_rate
    assert abs(np.argmax(res) - expected_bin) <= 1.5
    assert 0.95 < res.max() <= 1.01


@pytest.mark.parametrize('scale_l2_norm', [False, True])
def test_gabor_impulse_response_is_inverse_of_frequency_response(
        scale_l2_norm):
    bank = GaborFilterBank(
        MelScaling(), num_filts=40, scale_l2_norm=scale_l2_norm)
    width = 512
    ir = bank.get_impulse_response(20, width)
    assert ir.dtype == np.complex128
    assert ir.shape == (width,)
    freq = bank.get_frequency_response(20, width)
    assert np.allclose(np.fft.fft(ir), freq, atol=1e-4)


@pytest.mark.parametrize('filt_idx', [0, 20, 39])
@pytest.mark.parametrize('width', [255, 256])
def test_gabor_truncated_response_reconstructs_full(filt_idx, width):
    bank = GaborFilterBank(MelScaling(), num_filts=40)
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    assert 0 <= bin_idx < width
    assert 0 < len(trnc) <= width
    full = _reconstruct_wrapped(bin_idx, trnc, width)
    expected = bank.get_frequency_response(filt_idx, width)
    atol = 4 * config.EFFECTIVE_SUPPORT_THRESHOLD
    assert np.allclose(full, expected, atol=atol)


def test_gammatone_properties_and_invalid_order():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=15, order=3)
    assert bank.num_filts == 15
    assert bank.order == 3
    assert not bank.is_real
    assert not bank.is_zero_phase
    assert len(bank.centers_hz) == 15
    assert all(
        a < b for a, b in zip(bank.centers_hz[:-1], bank.centers_hz[1:]))
    for (start, end), (low, high) in zip(bank.supports, bank.supports_hz):
        assert end > start
        assert low < high
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=0)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=2.5)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), high_hz=20000.)


def test_gammatone_frequency_response_peaks_at_center():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=20)
    width = 4096
    idx = 10
    res = bank.get_frequency_response(idx, width)
    assert res.dtype == np.complex128
    assert res.shape == (width,)
    mags = np.abs(res)
    assert 0.95 < mags.max() < 1.05
    expected_bin = bank.centers_hz[idx] * width / bank.sampling_rate
    assert abs(np.argmax(mags) - expected_bin) <= 2


@pytest.mark.parametrize('filt_idx', [0, 10, 19])
@pytest.mark.parametrize('width', [256, 257])
def test_gammatone_truncated_response_reconstructs_full(filt_idx, width):
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=20)
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    assert 0 <= bin_idx < width
    assert 0 < len(trnc) <= width
    full = _reconstruct_wrapped(bin_idx, trnc, width)
    expected = bank.get_frequency_response(filt_idx, width)
    assert np.allclose(full, expected, atol=1e-2)


def test_gammatone_impulse_response_is_causal_complex():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=10)
    width = 512
    ir = bank.get_impulse_response(5, width)
    assert ir.dtype == np.complex128
    assert ir.shape == (width,)
    assert ir[0] == 0
    assert np.abs(ir).max() > 0


def test_from_alias_creates_banks():
    tri = LinearFilterBank.from_alias('tri', 'mel', num_filts=7)
    assert isinstance(tri, TriangularOverlappingFilterBank)
    assert tri.num_filts == 7
    gabor = LinearFilterBank.from_alias('gabor', 'mel', num_filts=6)
    assert isinstance(gabor, GaborFilterBank)
    assert gabor.num_filts == 6
    tone = LinearFilterBank.from_alias('gammatone', 'mel', num_filts=5)
    assert isinstance(tone, ComplexGammatoneFilterBank)
    assert tone.num_filts == 5
