import numpy as np
import pytest

from pydrobert.speech import config
from pydrobert.speech.filters import ComplexGammatoneFilterBank
from pydrobert.speech.filters import Fbank
from pydrobert.speech.filters import GaborFilterBank
from pydrobert.speech.filters import TriangularOverlappingFilterBank
from pydrobert.speech.scales import MelScaling

EPS = config.EFFECTIVE_SUPPORT_THRESHOLD

BANK_FACTORIES = {
    "tri": lambda: TriangularOverlappingFilterBank("mel", num_filts=10),
    "tri_analytic": lambda: TriangularOverlappingFilterBank(
        "mel", num_filts=10, analytic=True
    ),
    "gabor": lambda: GaborFilterBank("mel", num_filts=10),
    "gammatone": lambda: ComplexGammatoneFilterBank("mel", num_filts=10),
}


@pytest.fixture(params=list(BANK_FACTORIES))
def bank(request):
    return BANK_FACTORIES[request.param]()


def _half_len(width):
    return (width + 1) // 2 if width % 2 else width // 2 + 1


def _full_from_truncated(bank, filt_idx, width):
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    full = np.zeros(width, dtype=trnc.dtype)
    if bank.is_real:
        for k, v in enumerate(trnc):
            idx = bin_idx + k
            full[idx] = v
            if idx:
                full[width - idx] = np.conj(v)
    else:
        wrap = min(bin_idx + len(trnc), width) - bin_idx
        full[bin_idx:bin_idx + wrap] = trnc[:wrap]
        full[:len(trnc) - wrap] = trnc[wrap:]
    return bin_idx, full


# ---------------------------------------------------------------------------
# Triangular filter bank
# ---------------------------------------------------------------------------


def test_tri_basic_properties():
    bank = TriangularOverlappingFilterBank("mel", num_filts=12, sampling_rate=8000)
    assert bank.num_filts == 12
    assert bank.sampling_rate == 8000
    assert bank.is_real
    assert not bank.is_analytic
    assert bank.is_zero_phase
    assert len(bank.centers_hz) == 12
    assert len(bank.supports_hz) == 12
    assert len(bank.supports) == 12
    assert all(a < b for a, b in zip(bank.centers_hz[:-1], bank.centers_hz[1:]))
    for (low, high), center in zip(bank.supports_hz, bank.centers_hz):
        assert low < center < high


def test_tri_centers_uniform_in_mel_scale():
    bank = TriangularOverlappingFilterBank(
        "mel", num_filts=10, low_hz=20.0, high_hz=8000.0
    )
    mel = MelScaling()
    scales = [mel.hertz_to_scale(c) for c in bank.centers_hz]
    diffs = np.diff(scales)
    assert np.allclose(diffs, diffs[0], rtol=1e-6)
    assert bank.supports_hz[0][0] == pytest.approx(20.0, rel=1e-6)
    assert bank.supports_hz[-1][1] == pytest.approx(8000.0, rel=1e-6)


def test_tri_supports_and_supports_ms():
    bank = TriangularOverlappingFilterBank("mel", num_filts=10)
    for start, end in bank.supports:
        assert start <= -1
        assert end >= 1
    for (s0, s1), (m0, m1) in zip(bank.supports, bank.supports_ms):
        assert m0 == pytest.approx(s0 * 1000 / bank.sampling_rate)
        assert m1 == pytest.approx(s1 * 1000 / bank.sampling_rate)


def test_tri_analytic_flags_and_negative_frequencies_zero():
    bank = TriangularOverlappingFilterBank("mel", num_filts=10, analytic=True)
    assert bank.is_analytic
    assert not bank.is_real
    width = 512
    for filt_idx in range(bank.num_filts):
        fr = bank.get_frequency_response(filt_idx, width)
        assert np.all(fr[width // 2 + 1:] == 0)
        assert fr.max() > 0


def test_tri_frequency_response_peak_and_symmetry():
    bank = TriangularOverlappingFilterBank("mel", num_filts=10)
    width = 1024
    filt_idx = 5
    fr = bank.get_frequency_response(filt_idx, width)
    assert fr.dtype == np.float64
    assert fr.shape == (width,)
    assert np.all(fr >= -1e-9)
    assert fr.max() <= 1 + 1e-9
    assert fr.max() > 0.9
    # real filters have conjugate (here: even) symmetry
    assert np.allclose(fr[1:], fr[:0:-1])
    bin_hz = bank.sampling_rate / width
    peak_hz = np.argmax(fr[: width // 2 + 1]) * bin_hz
    assert abs(peak_hz - bank.centers_hz[filt_idx]) <= 2 * bin_hz


def test_tri_truncated_response_real_half_spectrum():
    bank = TriangularOverlappingFilterBank("mel", num_filts=10)
    for width in (512, 511):
        half_width = _half_len(width)
        for filt_idx in range(bank.num_filts):
            bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
            assert bin_idx >= 0
            assert trnc.max() <= 1 + 1e-9
            half = np.zeros(half_width, dtype=trnc.dtype)
            half[bin_idx:bin_idx + len(trnc)] = trnc
            expected = bank.get_frequency_response(filt_idx, width, half=True)
            assert np.allclose(half, expected)


# ---------------------------------------------------------------------------
# Fbank
# ---------------------------------------------------------------------------


def test_fbank_basic_responses():
    bank = Fbank(num_filts=10, high_hz=7000.0)
    assert bank.num_filts == 10
    assert bank.is_real
    assert bank.is_zero_phase
    width = 500
    for filt_idx in range(bank.num_filts):
        full = bank.get_frequency_response(filt_idx, width)
        half = bank.get_frequency_response(filt_idx, width, half=True)
        assert full.shape == (width,)
        assert half.shape == (width // 2 + 1,)
        assert np.allclose(full[: len(half)], half)
        assert full.min() >= -1e-9
        assert full.max() <= 1 + 1e-6
        _, rebuilt = _full_from_truncated(bank, filt_idx, width)
        assert np.allclose(rebuilt, full)
        ir = bank.get_impulse_response(filt_idx, width)
        assert ir.shape == (width,)
        assert np.isrealobj(ir)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(high_hz=9000.0),
        dict(low_hz=-1.0, high_hz=4000.0),
        dict(low_hz=3000.0, high_hz=2000.0),
    ],
)
@pytest.mark.parametrize(
    "factory",
    [
        lambda **kw: TriangularOverlappingFilterBank("mel", **kw),
        lambda **kw: GaborFilterBank("mel", **kw),
        lambda **kw: ComplexGammatoneFilterBank("mel", **kw),
        lambda **kw: Fbank(**kw),
    ],
)
def test_invalid_frequency_range_raises(factory, kwargs):
    with pytest.raises(ValueError):
        factory(sampling_rate=16000, **kwargs)


@pytest.mark.parametrize("order", [0, -1, 2.5])
def test_gammatone_invalid_order_raises(order):
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", num_filts=5, order=order)


# ---------------------------------------------------------------------------
# Generic tests over all banks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [256, 257])
def test_frequency_response_half_matches_full(bank, width):
    for filt_idx in range(bank.num_filts):
        full = bank.get_frequency_response(filt_idx, width)
        half = bank.get_frequency_response(filt_idx, width, half=True)
        assert full.shape == (width,)
        assert half.shape == (_half_len(width),)
        assert np.allclose(full[: len(half)], half)


def test_frequency_response_dtype(bank):
    fr = bank.get_frequency_response(3, 128)
    if bank.is_zero_phase:
        assert fr.dtype == np.float64
    else:
        assert fr.dtype == np.complex128


def test_impulse_response_shape_and_dtype(bank):
    width = 128
    for filt_idx in (0, bank.num_filts // 2, bank.num_filts - 1):
        ir = bank.get_impulse_response(filt_idx, width)
        assert ir.shape == (width,)
        if bank.is_real:
            assert ir.dtype == np.float64
        else:
            assert ir.dtype == np.complex128


@pytest.mark.parametrize("width", [512, 511])
def test_truncated_response_reconstructs_full_response(bank, width):
    for filt_idx in range(bank.num_filts):
        bin_idx, rebuilt = _full_from_truncated(bank, filt_idx, width)
        assert 0 <= bin_idx < width
        expected = bank.get_frequency_response(filt_idx, width)
        assert np.allclose(rebuilt, expected, atol=5 * EPS)


# ---------------------------------------------------------------------------
# Gabor
# ---------------------------------------------------------------------------


def test_gabor_properties():
    bank = GaborFilterBank("mel", num_filts=10, scale_l2_norm=True, erb=True)
    assert bank.num_filts == 10
    assert not bank.is_real
    assert bank.is_zero_phase
    assert bank.scaled_l2_norm is True
    assert bank.erb is True
    assert isinstance(bank.is_analytic, bool)
    assert all(a < b for a, b in zip(bank.centers_hz[:-1], bank.centers_hz[1:]))
    for (low, high), center in zip(bank.supports_hz, bank.centers_hz):
        assert low < center < high
    for start, end in bank.supports:
        assert start < 0 < end
        assert start == -end


def test_gabor_peak_gain_near_center():
    bank = GaborFilterBank("mel", num_filts=10)
    width = 1024
    filt_idx = 5
    fr = bank.get_frequency_response(filt_idx, width)
    assert 0.9 < fr.max() <= 1.01
    bin_hz = bank.sampling_rate / width
    peak_hz = np.argmax(fr[: width // 2 + 1]) * bin_hz
    assert abs(peak_hz - bank.centers_hz[filt_idx]) <= 2 * bin_hz


def test_gabor_frequency_response_matches_fft_of_impulse_response():
    bank = GaborFilterBank("mel", num_filts=10)
    width = 1024
    for filt_idx in range(bank.num_filts):
        ir = bank.get_impulse_response(filt_idx, width)
        fr = bank.get_frequency_response(filt_idx, width)
        assert np.allclose(np.fft.fft(ir), fr, atol=1e-3)


def test_gabor_l2_normalized_impulse_response_has_unit_energy():
    bank = GaborFilterBank("mel", num_filts=10, scale_l2_norm=True)
    width = 1024
    for filt_idx in range(bank.num_filts):
        ir = bank.get_impulse_response(filt_idx, width)
        assert np.sum(np.abs(ir) ** 2) == pytest.approx(1.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Complex gammatone
# ---------------------------------------------------------------------------


def test_gammatone_properties():
    bank = ComplexGammatoneFilterBank("mel", num_filts=10, order=3)
    assert bank.num_filts == 10
    assert bank.order == 3
    assert not bank.is_real
    assert not bank.is_zero_phase
    assert bank.scaled_l2_norm is False
    assert bank.erb is False
    assert all(a < b for a, b in zip(bank.centers_hz[:-1], bank.centers_hz[1:]))
    assert len(bank.supports) == 10
    for start, end in bank.supports:
        assert start <= 0 < end
    for (low, high), center in zip(bank.supports_hz, bank.centers_hz):
        assert low < center < high


def test_gammatone_peak_gain_near_center():
    bank = ComplexGammatoneFilterBank("mel", num_filts=10)
    width = 1024
    filt_idx = 5
    fr = bank.get_frequency_response(filt_idx, width)
    assert fr.dtype == np.complex128
    mag = np.abs(fr)
    assert 0.8 < mag.max() < 1.2
    bin_hz = bank.sampling_rate / width
    peak_hz = np.argmax(mag[: width // 2 + 1]) * bin_hz
    assert abs(peak_hz - bank.centers_hz[filt_idx]) <= 2 * bin_hz


def test_gammatone_impulse_response_is_causal_by_default():
    bank = ComplexGammatoneFilterBank("mel", num_filts=10)
    width = 1024
    ir = bank.get_impulse_response(5, width)
    assert ir.shape == (width,)
    assert ir[0] == 0
    assert np.abs(ir[1:50]).max() > 0


def test_gammatone_max_centered_peaks_at_zero():
    bank = ComplexGammatoneFilterBank("mel", num_filts=10, max_centered=True)
    width = 1024
    ir = bank.get_impulse_response(5, width)
    assert int(np.argmax(np.abs(ir))) in (0, 1, width - 1)
