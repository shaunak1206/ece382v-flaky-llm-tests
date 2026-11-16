import numpy as np
import pytest

from pydrobert.speech import config
from pydrobert.speech.scales import MelScaling
from pydrobert.speech.filters import (
    ComplexGammatoneFilterBank,
    Fbank,
    GaborFilterBank,
    LinearFilterBank,
    TriangularOverlappingFilterBank,
)

EPS = config.EFFECTIVE_SUPPORT_THRESHOLD
NUM_FILTS = 10
RATE = 16000

BANK_KINDS = [
    "tri",
    "tri_analytic",
    "fbank",
    "fbank_analytic",
    "gabor",
    "gammatone",
]


def _make_bank(kind):
    if kind == "tri":
        return TriangularOverlappingFilterBank(MelScaling(), num_filts=NUM_FILTS)
    if kind == "tri_analytic":
        return TriangularOverlappingFilterBank(
            MelScaling(), num_filts=NUM_FILTS, analytic=True
        )
    if kind == "fbank":
        return Fbank(num_filts=NUM_FILTS)
    if kind == "fbank_analytic":
        return Fbank(num_filts=NUM_FILTS, analytic=True)
    if kind == "gabor":
        return GaborFilterBank(MelScaling(), num_filts=NUM_FILTS)
    if kind == "gammatone":
        return ComplexGammatoneFilterBank(MelScaling(), num_filts=NUM_FILTS)
    raise ValueError(kind)


@pytest.fixture(params=BANK_KINDS)
def bank(request):
    return _make_bank(request.param)


def _half_len(width):
    return width // 2 + 1 if width % 2 == 0 else (width + 1) // 2


def test_banks_are_linear_filter_banks_with_basic_properties(bank):
    assert isinstance(bank, LinearFilterBank)
    assert bank.num_filts == NUM_FILTS
    assert bank.sampling_rate == RATE
    assert len(bank.centers_hz) == NUM_FILTS
    assert not (bank.is_real and bank.is_analytic)
    centers = np.array(bank.centers_hz)
    assert np.all(np.diff(centers) > 0)
    assert np.all(centers > 0)
    assert np.all(centers <= RATE / 2)


def test_supports_structure(bank):
    supports = bank.supports
    supports_hz = bank.supports_hz
    supports_ms = bank.supports_ms
    assert len(supports) == NUM_FILTS
    assert len(supports_hz) == NUM_FILTS
    assert len(supports_ms) == NUM_FILTS
    for idx in range(NUM_FILTS):
        start, end = supports[idx]
        assert start <= 0 <= end
        assert start < end
        low, high = supports_hz[idx]
        assert low < high
        assert low <= bank.centers_hz[idx] <= high
        assert supports_ms[idx][0] == pytest.approx(start * 1000 / RATE)
        assert supports_ms[idx][1] == pytest.approx(end * 1000 / RATE)


@pytest.mark.parametrize("width", [200, 201])
def test_frequency_response_length_and_dtype(bank, width):
    for idx in (0, NUM_FILTS // 2, NUM_FILTS - 1):
        full = bank.get_frequency_response(idx, width)
        assert full.shape == (width,)
        half = bank.get_frequency_response(idx, width, half=True)
        assert half.shape == (_half_len(width),)
        if bank.is_zero_phase:
            assert full.dtype == np.float64
            assert half.dtype == np.float64
        else:
            assert full.dtype == np.complex128
            assert half.dtype == np.complex128


@pytest.mark.parametrize("width", [200, 201])
def test_half_frequency_response_matches_full(bank, width):
    hl = _half_len(width)
    for idx in range(NUM_FILTS):
        full = bank.get_frequency_response(idx, width)
        half = bank.get_frequency_response(idx, width, half=True)
        assert np.allclose(full[:hl], half)


@pytest.mark.parametrize("width", [200, 201])
def test_truncated_response_reconstructs_frequency_response(bank, width):
    atol = 20 * EPS
    for idx in range(NUM_FILTS):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        assert 0 <= bin_idx < width
        if bank.is_real:
            expected = bank.get_frequency_response(idx, width, half=True)
            recon = np.zeros(len(expected), dtype=trnc.dtype)
            assert bin_idx + len(trnc) <= len(recon)
            recon[bin_idx : bin_idx + len(trnc)] = trnc
        else:
            expected = bank.get_frequency_response(idx, width)
            recon = np.zeros(width, dtype=trnc.dtype)
            wrap = min(bin_idx + len(trnc), width) - bin_idx
            recon[bin_idx : bin_idx + wrap] = trnc[:wrap]
            recon[: len(trnc) - wrap] = trnc[wrap:]
        assert np.allclose(recon, expected, atol=atol)


def test_impulse_response_shape_and_dtype(bank):
    width = 128
    for idx in (0, NUM_FILTS - 1):
        ir = bank.get_impulse_response(idx, width)
        assert ir.shape == (width,)
        if bank.is_real:
            assert ir.dtype == np.float64
        else:
            assert ir.dtype == np.complex128
        assert np.all(np.isfinite(ir))


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (lambda **kw: TriangularOverlappingFilterBank(MelScaling(), **kw), {"high_hz": 9000}),
        (lambda **kw: TriangularOverlappingFilterBank(MelScaling(), **kw), {"low_hz": -1.0, "high_hz": 4000}),
        (lambda **kw: TriangularOverlappingFilterBank(MelScaling(), **kw), {"low_hz": 100.0, "high_hz": 50.0}),
        (lambda **kw: Fbank(**kw), {"high_hz": 9000}),
        (lambda **kw: Fbank(**kw), {"low_hz": 100.0, "high_hz": 100.0}),
        (lambda **kw: GaborFilterBank(MelScaling(), **kw), {"high_hz": 9000}),
        (lambda **kw: GaborFilterBank(MelScaling(), **kw), {"low_hz": -5.0, "high_hz": 4000}),
        (lambda **kw: ComplexGammatoneFilterBank(MelScaling(), **kw), {"high_hz": 9000}),
        (lambda **kw: ComplexGammatoneFilterBank(MelScaling(), **kw), {"low_hz": 100.0, "high_hz": 50.0}),
    ],
)
def test_invalid_frequency_ranges_raise(factory, kwargs):
    with pytest.raises(ValueError):
        factory(num_filts=5, **kwargs)


def test_triangular_frequency_response_shape():
    width = 512
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=NUM_FILTS)
    assert bank.is_real and not bank.is_analytic and bank.is_zero_phase
    for idx in range(NUM_FILTS):
        res = bank.get_frequency_response(idx, width)
        assert np.all(res >= 0)
        assert np.all(res <= 1 + 1e-9)
        assert res[0] == 0
        # Hermitian symmetry of a real zero-phase filter
        assert np.allclose(res[1:], res[1:][::-1])
        # nonzero region lies inside the support (in Hz)
        low, high = bank.supports_hz[idx]
        nonzero = np.nonzero(res[: width // 2 + 1])[0]
        assert len(nonzero) > 0
        assert nonzero[0] * RATE / width >= low - 1e-6
        assert nonzero[-1] * RATE / width <= high + 1e-6
        # close to unity at the center
        center_bin = bank.centers_hz[idx] * width / RATE
        assert abs(np.argmax(res[: width // 2 + 1]) - center_bin) <= 1
    wide = bank.get_frequency_response(NUM_FILTS - 1, 1024)
    assert wide.max() > 0.9


def test_triangular_analytic_has_no_negative_frequencies():
    width = 512
    bank = TriangularOverlappingFilterBank(
        MelScaling(), num_filts=NUM_FILTS, analytic=True
    )
    assert bank.is_analytic and not bank.is_real
    for idx in range(NUM_FILTS):
        full = bank.get_frequency_response(idx, width)
        assert np.all(full[width // 2 + 1 :] == 0)
        assert full.max() > 0
        half = bank.get_frequency_response(idx, width, half=True)
        assert np.allclose(full[: len(half)], half)


@pytest.mark.parametrize("analytic", [False, True])
def test_triangular_impulse_response_dft_matches_frequency_response(analytic):
    width = 1024
    bank = TriangularOverlappingFilterBank(
        MelScaling(), num_filts=NUM_FILTS, analytic=analytic
    )
    idx = 7
    ir = bank.get_impulse_response(idx, width)
    freq = bank.get_frequency_response(idx, width)
    dft = np.fft.fft(ir)
    assert np.allclose(dft, freq, atol=2e-2)
    if not analytic:
        # real, even impulse response
        assert np.allclose(ir[1:], ir[1:][::-1])


def test_triangular_sampling_rate_and_scaling_arg():
    bank = TriangularOverlappingFilterBank(
        MelScaling(), num_filts=8, sampling_rate=8000, low_hz=50.0, high_hz=3000.0
    )
    assert bank.sampling_rate == 8000
    assert bank.num_filts == 8
    centers = np.array(bank.centers_hz)
    assert np.all(centers > 50.0) and np.all(centers < 3000.0)
    assert bank.supports_hz[0][0] == pytest.approx(50.0)
    assert bank.supports_hz[-1][1] == pytest.approx(3000.0)
    # default high_hz is Nyquist
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=8, sampling_rate=8000)
    assert bank.supports_hz[-1][1] == pytest.approx(4000.0)


@pytest.mark.parametrize("analytic", [False, True])
def test_fbank_impulse_response_inverts_frequency_response(analytic):
    width = 512
    bank = Fbank(num_filts=NUM_FILTS, analytic=analytic)
    for idx in range(NUM_FILTS):
        ir = bank.get_impulse_response(idx, width)
        if analytic:
            assert ir.dtype == np.complex128
            freq = bank.get_frequency_response(idx, width)
            assert np.allclose(np.fft.fft(ir), freq, atol=1e-10)
        else:
            assert ir.dtype == np.float64
            freq = bank.get_frequency_response(idx, width, half=True)
            assert np.allclose(np.fft.rfft(ir), freq, atol=1e-10)


def test_fbank_frequency_response_peak_and_range():
    width = 1024
    bank = Fbank(num_filts=NUM_FILTS)
    for idx in range(NUM_FILTS):
        res = bank.get_frequency_response(idx, width)
        assert np.all(res >= 0)
        assert np.all(res <= 1 + 1e-9)
        assert res.max() > 0.9
        center_bin = bank.centers_hz[idx] * width / RATE
        assert abs(np.argmax(res[: width // 2 + 1]) - center_bin) <= 1
        assert np.allclose(res[1:], res[1:][::-1])


def test_gabor_peak_and_attributes():
    width = 1024
    bank = GaborFilterBank(MelScaling(), num_filts=NUM_FILTS, erb=True)
    assert bank.erb is True
    assert bank.scaled_l2_norm is False
    assert not bank.is_real
    assert bank.is_zero_phase
    for idx in range(NUM_FILTS):
        res = bank.get_frequency_response(idx, width)
        assert res.max() <= 1.001
        assert res.max() > 0.9
        center_bin = bank.centers_hz[idx] * width / RATE
        assert abs(np.argmax(res) - center_bin) <= 1
    bank2 = GaborFilterBank(MelScaling(), num_filts=NUM_FILTS, scale_l2_norm=True)
    assert bank2.scaled_l2_norm is True
    assert bank2.erb is False


@pytest.mark.parametrize("scale_l2_norm", [False, True])
def test_gabor_impulse_response_dft_matches_frequency_response(scale_l2_norm):
    width = 1024
    bank = GaborFilterBank(
        MelScaling(), num_filts=NUM_FILTS, scale_l2_norm=scale_l2_norm
    )
    for idx in (0, 4, NUM_FILTS - 1):
        ir = bank.get_impulse_response(idx, width)
        freq = bank.get_frequency_response(idx, width)
        assert np.allclose(np.fft.fft(ir), freq, atol=1e-6)


@pytest.mark.parametrize("erb", [False, True])
def test_gabor_l2_norm_scaling_gives_unit_energy(erb):
    width = 2048
    bank = GaborFilterBank(
        MelScaling(), num_filts=NUM_FILTS, scale_l2_norm=True, erb=erb
    )
    for idx in range(NUM_FILTS):
        ir = bank.get_impulse_response(idx, width)
        assert np.sum(np.abs(ir) ** 2) == pytest.approx(1.0, rel=1e-3)


@pytest.mark.parametrize("order", [0, -1, 2.5])
def test_gammatone_invalid_order(order):
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), num_filts=5, order=order)


def test_gammatone_properties_and_peak():
    width = 1024
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=NUM_FILTS, order=3)
    assert bank.order == 3
    assert not bank.is_real
    assert not bank.is_zero_phase
    assert bank.erb is False
    assert bank.scaled_l2_norm is False
    for idx in range(NUM_FILTS):
        res = np.abs(bank.get_frequency_response(idx, width))
        assert 0.8 < res.max() < 1.05
        center_bin = bank.centers_hz[idx] * width / RATE
        assert abs(np.argmax(res) - center_bin) <= 1


def test_gammatone_impulse_response_dft_matches_frequency_response():
    width = 2048
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=NUM_FILTS)
    idx = NUM_FILTS - 2
    ir = bank.get_impulse_response(idx, width)
    freq = bank.get_frequency_response(idx, width)
    assert ir.dtype == np.complex128
    assert np.allclose(np.fft.fft(ir), freq, atol=5e-2)


def test_gammatone_max_centered_shifts_supports_before_zero():
    plain = ComplexGammatoneFilterBank(MelScaling(), num_filts=NUM_FILTS)
    centered = ComplexGammatoneFilterBank(
        MelScaling(), num_filts=NUM_FILTS, max_centered=True
    )
    for idx in range(NUM_FILTS):
        assert plain.supports[idx][0] == 0
        assert centered.supports[idx][0] < 0
        assert centered.supports[idx][1] > centered.supports[idx][0]
    # centered magnitude response is still peaked at the same place
    width = 512
    for idx in (2, 6):
        a = np.abs(plain.get_frequency_response(idx, width))
        b = np.abs(centered.get_frequency_response(idx, width))
        assert abs(np.argmax(a) - np.argmax(b)) <= 1
