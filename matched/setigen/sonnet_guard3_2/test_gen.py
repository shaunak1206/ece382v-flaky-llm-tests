import numpy as np
import pytest
from astropy import units as u
from astropy.stats import sigma_clip

from setigen.frame import Frame


FCHANS = 16
TCHANS = 8


def make_frame(fchans=FCHANS, tchans=TCHANS, data=None):
    # df * dt = 6 -> chi2_df = 24
    return Frame(fchans=fchans,
                 tchans=tchans,
                 df=2.0 * u.Hz,
                 dt=3.0 * u.s,
                 fch1=8 * u.GHz,
                 data=data)


def test_add_noise_returns_array_of_frame_shape():
    np.random.seed(1234)
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    assert isinstance(noise, np.ndarray)
    assert noise.shape == frame.shape == (TCHANS, FCHANS)


def test_add_noise_on_zero_frame_data_equals_noise():
    np.random.seed(1234)
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    assert np.allclose(frame.data, noise)


def test_chi2_noise_is_positive():
    np.random.seed(1234)
    frame = make_frame()
    noise = frame.add_noise(x_mean=5, noise_type='chi2')
    assert np.all(noise > 0)


def test_chi2_sets_noise_stats_from_parameters_on_empty_frame():
    np.random.seed(1234)
    frame = make_frame()
    assert frame.get_noise_stats() == (0, 0)
    frame.add_noise(x_mean=10)
    expected_std = np.sqrt(2 * frame.chi2_df) * 10 / frame.chi2_df
    assert frame.noise_mean == 10
    assert frame.noise_std == pytest.approx(expected_std)


def test_chi2_ignores_x_std_argument():
    np.random.seed(1234)
    frame = make_frame()
    frame.add_noise(x_mean=10, x_std=1000, noise_type='chi2')
    expected_std = np.sqrt(2 * frame.chi2_df) * 10 / frame.chi2_df
    assert frame.noise_std == pytest.approx(expected_std)


def test_chi2_sample_mean_close_to_requested_mean():
    np.random.seed(1234)
    frame = make_frame(fchans=100, tchans=100)
    noise = frame.add_noise(x_mean=10)
    assert np.mean(noise) == pytest.approx(10, rel=0.05)


def test_gaussian_sets_noise_stats_and_statistics():
    np.random.seed(1234)
    frame = make_frame(fchans=50, tchans=50)
    noise = frame.add_noise(x_mean=10, x_std=2, noise_type='gaussian')
    assert frame.noise_mean == 10
    assert frame.noise_std == 2
    assert np.mean(noise) == pytest.approx(10, abs=0.5)
    assert np.std(noise) == pytest.approx(2, abs=0.5)


def test_normal_and_gaussian_are_equivalent_with_same_seed():
    np.random.seed(42)
    frame1 = make_frame()
    noise1 = frame1.add_noise(x_mean=3, x_std=1, noise_type='normal')

    np.random.seed(42)
    frame2 = make_frame()
    noise2 = frame2.add_noise(x_mean=3, x_std=1, noise_type='gaussian')

    assert np.allclose(noise1, noise2)
    assert np.allclose(frame1.data, frame2.data)


def test_gaussian_without_std_exits():
    np.random.seed(1234)
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=10, noise_type='gaussian')
    # data untouched
    assert np.all(frame.data == 0)


def test_invalid_noise_type_exits():
    np.random.seed(1234)
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=10, x_std=1, noise_type='bogus')
    assert np.all(frame.data == 0)


def test_truncated_gaussian_respects_minimum():
    np.random.seed(1234)
    frame = make_frame(fchans=40, tchans=40)
    noise = frame.add_noise(x_mean=1, x_std=2, x_min=0, noise_type='gaussian')
    assert noise.shape == frame.shape
    assert np.all(noise >= 0 - 1e-9)


def test_same_seed_gives_identical_chi2_noise():
    np.random.seed(7)
    noise1 = make_frame().add_noise(x_mean=4)
    np.random.seed(7)
    noise2 = make_frame().add_noise(x_mean=4)
    assert np.array_equal(noise1, noise2)


def test_repeated_add_noise_accumulates_and_updates_stats():
    np.random.seed(1234)
    frame = make_frame(fchans=32, tchans=32)
    noise1 = frame.add_noise(x_mean=10)
    noise2 = frame.add_noise(x_mean=10)
    assert np.allclose(frame.data, noise1 + noise2)

    clipped = sigma_clip(frame.data, sigma=3, maxiters=5, masked=False)
    assert frame.noise_mean == pytest.approx(np.mean(clipped))
    assert frame.noise_std == pytest.approx(np.std(clipped))
    assert frame.noise_std > 0


def test_add_noise_to_preloaded_data_adds_on_top():
    np.random.seed(1234)
    base = np.ones((TCHANS, FCHANS))
    frame = make_frame(data=base.copy())
    noise = frame.add_noise(x_mean=5, x_std=1, noise_type='gaussian')
    assert np.allclose(frame.data, base + noise)
    assert np.isfinite(frame.noise_mean)
    assert np.isfinite(frame.noise_std)


def test_get_data_returns_underlying_data_by_default():
    np.random.seed(1234)
    frame = make_frame()
    frame.add_noise(x_mean=10)
    assert frame.get_data() is frame.data
    assert frame.get_data(use_db=False) is frame.data


def test_get_data_use_db_matches_log10_formula():
    np.random.seed(1234)
    frame = make_frame()
    frame.add_noise(x_mean=10)
    result = frame.get_data(use_db=True)
    assert result.shape == frame.shape
    assert np.allclose(result, 10 * np.log10(frame.data))


def test_get_data_use_db_does_not_modify_data():
    np.random.seed(1234)
    data = np.full((TCHANS, FCHANS), 100.0)
    frame = make_frame(data=data)
    before = frame.data.copy()
    result = frame.get_data(use_db=True)
    assert np.allclose(result, 20.0)
    assert np.array_equal(frame.data, before)
    assert result is not frame.data


def test_get_data_use_db_zero_gives_negative_infinity():
    np.random.seed(1234)
    frame = make_frame()
    with np.errstate(divide='ignore'):
        result = frame.get_data(use_db=True)
    assert np.all(np.isneginf(result))


def test_get_data_after_zero_data_is_all_zeros():
    np.random.seed(1234)
    frame = make_frame()
    frame.add_noise(x_mean=10)
    frame.zero_data()
    assert np.all(frame.get_data() == 0)
    assert frame.get_data().shape == frame.shape
