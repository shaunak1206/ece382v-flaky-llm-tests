import numpy as np
import pytest
from astropy import units as u

import setigen as stg


def make_frame(fchans=64, tchans=32, df=10 * u.Hz, dt=10 * u.s):
    return stg.Frame(fchans=fchans, tchans=tchans, df=df, dt=dt,
                     fch1=6 * u.GHz)


@pytest.fixture(autouse=True)
def seed_rng():
    np.random.seed(1234)


def test_chi2_df_computed_from_resolution():
    frame = make_frame()
    assert frame.chi2_df == 4 * round(10 * 10)


def test_add_noise_returns_array_of_frame_shape():
    frame = make_frame(fchans=40, tchans=20)
    noise = frame.add_noise(x_mean=5)
    assert isinstance(noise, np.ndarray)
    assert noise.shape == (20, 40)


def test_add_noise_adds_to_data_on_empty_frame():
    frame = make_frame()
    noise = frame.add_noise(x_mean=5)
    np.testing.assert_allclose(frame.data, noise)


def test_add_noise_accumulates_on_existing_data():
    frame = make_frame()
    n1 = frame.add_noise(x_mean=5)
    n2 = frame.add_noise(x_mean=3)
    np.testing.assert_allclose(frame.data, n1 + n2)


def test_chi2_noise_mean_close_to_requested_mean():
    frame = make_frame(fchans=200, tchans=100)
    noise = frame.add_noise(x_mean=10)
    assert np.mean(noise) == pytest.approx(10, rel=0.05)


def test_chi2_noise_is_nonnegative():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    assert np.all(noise >= 0)


def test_chi2_sets_noise_stats_from_ideal_distribution():
    frame = make_frame()
    frame.add_noise(x_mean=10)
    expected_std = np.sqrt(2 * frame.chi2_df) * 10 / frame.chi2_df
    mean, std = frame.get_noise_stats()
    assert mean == 10
    assert std == pytest.approx(expected_std)


def test_chi2_ignores_x_std_argument():
    frame = make_frame()
    frame.add_noise(x_mean=10, x_std=999)
    expected_std = np.sqrt(2 * frame.chi2_df) * 10 / frame.chi2_df
    assert frame.noise_std == pytest.approx(expected_std)


def test_gaussian_noise_statistics():
    frame = make_frame(fchans=200, tchans=100)
    noise = frame.add_noise(x_mean=5, x_std=2, noise_type='gaussian')
    assert noise.shape == frame.shape
    assert np.mean(noise) == pytest.approx(5, abs=0.1)
    assert np.std(noise) == pytest.approx(2, rel=0.05)


def test_normal_noise_type_alias_sets_noise_stats():
    frame = make_frame()
    frame.add_noise(x_mean=5, x_std=2, noise_type='normal')
    assert frame.get_noise_stats() == (5, 2)


def test_truncated_gaussian_respects_minimum():
    frame = make_frame()
    noise = frame.add_noise(x_mean=1, x_std=2, x_min=0, noise_type='gaussian')
    assert noise.shape == frame.shape
    assert np.all(noise >= 0)


def test_gaussian_without_std_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=5, noise_type='gaussian')


def test_invalid_noise_type_exits_and_leaves_data_unchanged():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=5, noise_type='poisson')
    assert np.all(frame.data == 0)


def test_second_add_noise_updates_noise_stats_from_data():
    frame = make_frame(fchans=100, tchans=50)
    frame.add_noise(x_mean=10)
    frame.add_noise(x_mean=10)
    # Data is now the sum of two noise frames, so mean should be about 20
    assert frame.noise_mean == pytest.approx(20, rel=0.1)
    assert frame.noise_std > 0


def test_get_data_returns_underlying_array():
    frame = make_frame()
    assert frame.get_data() is frame.data
    frame.add_noise(x_mean=5)
    np.testing.assert_array_equal(frame.get_data(), frame.data)


def test_get_data_use_db_converts_to_decibels():
    data = np.full((4, 8), 100.0)
    data[0, 0] = 10.0
    frame = stg.Frame.from_data(df=1 * u.Hz, dt=1 * u.s, fch1=1 * u.GHz,
                                data=data.copy())
    db = frame.get_data(use_db=True)
    assert db.shape == (4, 8)
    assert db[0, 0] == pytest.approx(10.0)
    assert db[1, 1] == pytest.approx(20.0)
    # Underlying data is not modified
    assert frame.data[0, 0] == 10.0


def test_get_data_use_db_of_zeros_is_negative_infinity():
    frame = make_frame(fchans=4, tchans=3)
    with np.errstate(divide='ignore'):
        db = frame.get_data(use_db=True)
    assert np.all(np.isneginf(db))
