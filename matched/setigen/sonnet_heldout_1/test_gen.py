import numpy as np
import pytest
from astropy import units as u

import setigen as stg


def make_frame(fchans=256, tchans=128, df=2.7939677238464355, dt=18.25361108, data=None):
    return stg.Frame(fchans=fchans,
                     tchans=tchans,
                     df=df * u.Hz,
                     dt=dt * u.s,
                     fch1=6095.214842353016 * u.MHz,
                     data=data)


@pytest.fixture(autouse=True)
def seed_rng():
    np.random.seed(12345)


# ---------------- add_noise ----------------

def test_add_noise_returns_array_of_frame_shape():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    assert isinstance(noise, np.ndarray)
    assert noise.shape == frame.shape


def test_add_noise_on_zero_frame_data_equals_noise():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.data, noise)


def test_add_noise_accumulates_across_calls():
    frame = make_frame()
    n1 = frame.add_noise(x_mean=5)
    n2 = frame.add_noise(x_mean=7)
    np.testing.assert_allclose(frame.data, n1 + n2)


def test_add_noise_chi2_mean_close_to_x_mean():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10, noise_type='chi2')
    assert abs(np.mean(noise) - 10) < 0.2


def test_add_noise_chi2_is_nonnegative():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    assert np.all(noise >= 0)


def test_add_noise_chi2_sets_noise_stats_from_ideal_distribution():
    frame = make_frame()
    assert frame.get_noise_stats() == (0, 0)
    x_mean = 10
    frame.add_noise(x_mean=x_mean)
    expected_std = np.sqrt(2 * frame.chi2_df) * x_mean / frame.chi2_df
    mean, std = frame.get_noise_stats()
    assert mean == x_mean
    assert std == pytest.approx(expected_std)


def test_add_noise_chi2_ignores_x_std_argument():
    frame = make_frame()
    x_mean = 4
    frame.add_noise(x_mean=x_mean, x_std=1000, noise_type='chi2')
    expected_std = np.sqrt(2 * frame.chi2_df) * x_mean / frame.chi2_df
    assert frame.noise_std == pytest.approx(expected_std)


def test_add_noise_gaussian_statistics():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10, x_std=2, noise_type='gaussian')
    assert noise.shape == frame.shape
    assert abs(np.mean(noise) - 10) < 0.1
    assert abs(np.std(noise) - 2) < 0.1


def test_add_noise_gaussian_sets_noise_stats_to_params_on_empty_frame():
    frame = make_frame()
    frame.add_noise(x_mean=10, x_std=2, noise_type='gaussian')
    assert frame.noise_mean == 10
    assert frame.noise_std == 2


def test_add_noise_normal_alias_matches_gaussian_behavior():
    frame = make_frame()
    noise = frame.add_noise(x_mean=3, x_std=0.5, noise_type='normal')
    assert noise.shape == frame.shape
    assert abs(np.mean(noise) - 3) < 0.05
    assert frame.get_noise_stats() == (3, 0.5)


def test_add_noise_gaussian_with_x_min_enforces_lower_bound():
    frame = make_frame(fchans=128, tchans=64)
    x_min = 9.0
    noise = frame.add_noise(x_mean=10, x_std=2, x_min=x_min, noise_type='gaussian')
    assert noise.shape == frame.shape
    assert np.all(noise >= x_min)


def test_add_noise_gaussian_without_std_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=10, noise_type='gaussian')


def test_add_noise_invalid_noise_type_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=10, noise_type='poisson')


def test_add_noise_second_call_recomputes_noise_stats_with_sigma_clipping():
    frame = make_frame()
    frame.add_noise(x_mean=10, x_std=1, noise_type='gaussian')
    frame.add_noise(x_mean=5, x_std=1, noise_type='gaussian')
    # Stats are re-estimated from the data (not simply set to the last params)
    assert frame.noise_mean == pytest.approx(15, abs=0.1)
    assert frame.noise_std == pytest.approx(np.sqrt(2), abs=0.1)


def test_add_noise_on_preloaded_data_updates_stats_from_data():
    base = np.full((16, 32), 100.0)
    frame = make_frame(fchans=32, tchans=16, data=base)
    noise = frame.add_noise(x_mean=10, x_std=1, noise_type='gaussian')
    assert frame.noise_mean != 10  # not simply set to parameters
    assert frame.noise_mean == pytest.approx(np.mean(frame.data), abs=1.0)
    np.testing.assert_allclose(frame.data, base + noise)


# ---------------- get_data ----------------

def test_get_data_returns_underlying_array_by_default():
    frame = make_frame(fchans=16, tchans=8)
    assert frame.get_data() is frame.data
    assert frame.get_data().shape == (8, 16)


def test_get_data_after_add_noise_reflects_noise():
    frame = make_frame(fchans=16, tchans=8)
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.get_data(), noise)


def test_get_data_use_db_converts_to_decibels():
    data = np.array([[1.0, 10.0], [100.0, 1000.0]])
    frame = stg.Frame.from_data(df=1 * u.Hz, dt=1 * u.s, fch1=1 * u.GHz, data=data)
    result = frame.get_data(use_db=True)
    np.testing.assert_allclose(result, [[0.0, 10.0], [20.0, 30.0]])


def test_get_data_use_db_does_not_modify_stored_data():
    data = np.array([[1.0, 10.0], [100.0, 1000.0]])
    frame = stg.Frame.from_data(df=1 * u.Hz, dt=1 * u.s, fch1=1 * u.GHz, data=data.copy())
    frame.get_data(use_db=True)
    np.testing.assert_allclose(frame.data, data)


def test_get_data_use_db_on_zero_data_is_negative_infinity():
    frame = make_frame(fchans=4, tchans=2)
    with np.errstate(divide='ignore'):
        result = frame.get_data(use_db=True)
    assert result.shape == (2, 4)
    assert np.all(np.isneginf(result))
