import numpy as np
import pytest
from astropy import units as u

import setigen as stg


def make_frame(fchans=256, tchans=64, df=1.0, dt=10.0, data=None):
    # df * dt = 10 -> chi2_df = 40
    return stg.Frame(fchans=fchans,
                     tchans=tchans,
                     df=df * u.Hz,
                     dt=dt * u.s,
                     fch1=6000 * u.MHz,
                     data=data)


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(1234)


# ---------------- add_noise ----------------

def test_add_noise_returns_array_of_frame_shape():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    assert isinstance(noise, np.ndarray)
    assert noise.shape == frame.shape


def test_add_noise_to_zero_frame_data_equals_noise():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.data, noise)


def test_add_noise_chi2_mean_close_to_x_mean():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    assert np.mean(noise) == pytest.approx(10, rel=0.05)


def test_add_noise_chi2_values_are_positive():
    frame = make_frame()
    noise = frame.add_noise(x_mean=5)
    assert np.all(noise > 0)


def test_add_noise_chi2_sets_noise_stats_from_ideal_distribution():
    frame = make_frame()
    x_mean = 10
    frame.add_noise(x_mean=x_mean)
    expected_std = np.sqrt(2 * frame.chi2_df) * x_mean / frame.chi2_df
    assert frame.noise_mean == pytest.approx(x_mean)
    assert frame.noise_std == pytest.approx(expected_std)


def test_add_noise_chi2_ignores_x_std_argument():
    frame = make_frame()
    x_mean = 10
    frame.add_noise(x_mean=x_mean, x_std=123.0, noise_type='chi2')
    expected_std = np.sqrt(2 * frame.chi2_df) * x_mean / frame.chi2_df
    assert frame.noise_std == pytest.approx(expected_std)


def test_add_noise_gaussian_sets_noise_stats_to_parameters():
    frame = make_frame()
    frame.add_noise(x_mean=10, x_std=2, noise_type='gaussian')
    assert frame.noise_mean == 10
    assert frame.noise_std == 2


def test_add_noise_gaussian_statistics():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10, x_std=2, noise_type='gaussian')
    assert noise.shape == frame.shape
    assert np.mean(noise) == pytest.approx(10, rel=0.05)
    assert np.std(noise) == pytest.approx(2, rel=0.1)


def test_add_noise_normal_alias_matches_gaussian_behavior():
    frame = make_frame()
    noise = frame.add_noise(x_mean=7, x_std=1, noise_type='normal')
    assert noise.shape == frame.shape
    assert np.mean(noise) == pytest.approx(7, rel=0.05)
    assert frame.noise_mean == 7
    assert frame.noise_std == 1


def test_add_noise_gaussian_with_x_min_respects_lower_bound():
    frame = make_frame()
    x_min = 0.5
    noise = frame.add_noise(x_mean=1, x_std=1, x_min=x_min,
                            noise_type='gaussian')
    assert noise.shape == frame.shape
    assert np.all(noise >= x_min - 1e-9)


def test_add_noise_gaussian_without_x_std_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=10, noise_type='gaussian')


def test_add_noise_invalid_noise_type_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=10, noise_type='poisson')


def test_add_noise_accumulates_into_data():
    frame = make_frame()
    n1 = frame.add_noise(x_mean=10, x_std=1, noise_type='gaussian')
    n2 = frame.add_noise(x_mean=5, x_std=1, noise_type='gaussian')
    np.testing.assert_allclose(frame.data, n1 + n2)


def test_add_noise_second_call_updates_stats_from_data():
    frame = make_frame()
    frame.add_noise(x_mean=10, x_std=1, noise_type='gaussian')
    frame.add_noise(x_mean=5, x_std=1, noise_type='gaussian')
    # Stats are now estimated from the data (sum of the two noises)
    assert frame.noise_mean == pytest.approx(15, rel=0.05)
    assert frame.noise_std == pytest.approx(np.sqrt(2), rel=0.15)


def test_add_noise_on_preloaded_data_estimates_stats_from_data():
    data = np.ones((16, 32))
    frame = stg.Frame.from_data(df=1 * u.Hz, dt=10 * u.s,
                                fch1=6000 * u.MHz, data=data.copy())
    noise = frame.add_noise(x_mean=0, x_std=1, noise_type='gaussian')
    np.testing.assert_allclose(frame.data, 1 + noise)
    # Not simply set to parameters (0, 1) since the frame already had data
    assert frame.noise_mean == pytest.approx(1, abs=0.5)


# ---------------- get_data ----------------

def test_get_data_returns_underlying_array():
    frame = make_frame(fchans=16, tchans=8)
    assert frame.get_data() is frame.data
    assert frame.get_data().shape == (8, 16)


def test_get_data_reflects_added_noise():
    frame = make_frame(fchans=16, tchans=8)
    noise = frame.add_noise(x_mean=3)
    np.testing.assert_allclose(frame.get_data(), noise)


def test_get_data_use_db_converts_to_decibels():
    data = np.full((4, 8), 100.0)
    data[0, 0] = 10.0
    frame = stg.Frame.from_data(df=1 * u.Hz, dt=10 * u.s,
                                fch1=6000 * u.MHz, data=data.copy())
    db = frame.get_data(use_db=True)
    assert db.shape == (4, 8)
    assert db[0, 0] == pytest.approx(10.0)
    assert db[1, 1] == pytest.approx(20.0)


def test_get_data_use_db_does_not_modify_frame_data():
    data = np.full((4, 8), 100.0)
    frame = stg.Frame.from_data(df=1 * u.Hz, dt=10 * u.s,
                                fch1=6000 * u.MHz, data=data.copy())
    _ = frame.get_data(use_db=True)
    np.testing.assert_allclose(frame.data, data)
    assert frame.get_data(use_db=False) is frame.data


def test_get_data_use_db_of_zero_frame_is_negative_infinity():
    frame = make_frame(fchans=8, tchans=4)
    with np.errstate(divide='ignore'):
        db = frame.get_data(use_db=True)
    assert np.all(np.isneginf(db))
