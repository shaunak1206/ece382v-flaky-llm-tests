import numpy as np
import pytest
from astropy import units as u

import setigen as stg


def make_frame(fchans=256, tchans=64, df=2.0, dt=3.0, data=None):
    return stg.Frame(fchans=fchans,
                     tchans=tchans,
                     df=df * u.Hz,
                     dt=dt * u.s,
                     fch1=6 * u.GHz,
                     data=data)


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(12345)


def test_add_noise_returns_array_of_frame_shape():
    frame = make_frame(fchans=32, tchans=8)
    noise = frame.add_noise(x_mean=10)
    assert isinstance(noise, np.ndarray)
    assert noise.shape == (8, 32)


def test_add_noise_on_zero_frame_data_equals_noise():
    frame = make_frame(fchans=32, tchans=8)
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.get_data(), noise)


def test_add_noise_adds_to_existing_data():
    base = np.ones((8, 32))
    frame = make_frame(fchans=32, tchans=8, data=base.copy())
    noise = frame.add_noise(x_mean=5)
    np.testing.assert_allclose(frame.data, base + noise)


def test_chi2_noise_mean_close_to_requested_mean():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    assert np.mean(noise) == pytest.approx(10, rel=0.05)


def test_chi2_noise_is_non_negative():
    frame = make_frame(fchans=64, tchans=16)
    noise = frame.add_noise(x_mean=7)
    assert np.all(noise >= 0)


def test_chi2_noise_std_matches_theory():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    expected_std = np.sqrt(2 * frame.chi2_df) * 10 / frame.chi2_df
    assert np.std(noise) == pytest.approx(expected_std, rel=0.1)


def test_chi2_sets_noise_stats_on_empty_frame():
    frame = make_frame()
    frame.add_noise(x_mean=10)
    mean, std = frame.get_noise_stats()
    expected_std = np.sqrt(2 * frame.chi2_df) * 10 / frame.chi2_df
    assert mean == 10
    assert std == pytest.approx(expected_std)


def test_chi2_ignores_x_std_argument():
    frame = make_frame(fchans=32, tchans=8)
    frame.add_noise(x_mean=10, x_std=1000)
    expected_std = np.sqrt(2 * frame.chi2_df) * 10 / frame.chi2_df
    assert frame.noise_std == pytest.approx(expected_std)


def test_gaussian_noise_statistics():
    frame = make_frame()
    noise = frame.add_noise(x_mean=20, x_std=2, noise_type='gaussian')
    assert noise.shape == frame.shape
    assert np.mean(noise) == pytest.approx(20, rel=0.02)
    assert np.std(noise) == pytest.approx(2, rel=0.05)
    assert frame.get_noise_stats() == (20, 2)


def test_normal_alias_matches_gaussian_behavior():
    frame = make_frame()
    noise = frame.add_noise(x_mean=5, x_std=1, noise_type='normal')
    assert np.mean(noise) == pytest.approx(5, abs=0.1)
    assert np.std(noise) == pytest.approx(1, rel=0.05)
    assert frame.noise_mean == 5
    assert frame.noise_std == 1


def test_truncated_gaussian_respects_minimum():
    frame = make_frame(fchans=64, tchans=16)
    noise = frame.add_noise(x_mean=1, x_std=3, x_min=0, noise_type='gaussian')
    assert noise.shape == frame.shape
    assert np.all(noise >= 0)


def test_gaussian_without_std_exits():
    frame = make_frame(fchans=16, tchans=4)
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=10, noise_type='gaussian')
    # data must be untouched
    assert np.all(frame.data == 0)


def test_invalid_noise_type_exits():
    frame = make_frame(fchans=16, tchans=4)
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=10, x_std=1, noise_type='poisson')
    assert np.all(frame.data == 0)


def test_second_add_noise_updates_noise_stats_from_data():
    frame = make_frame()
    frame.add_noise(x_mean=10)
    frame.add_noise(x_mean=10)
    mean, std = frame.get_noise_stats()
    assert mean == pytest.approx(20, rel=0.1)
    assert std > 0
    # Total data mean should be about the sum of both means
    assert np.mean(frame.data) == pytest.approx(20, rel=0.05)


def test_add_noise_after_zero_data_resets_to_parameters():
    frame = make_frame(fchans=32, tchans=8)
    frame.add_noise(x_mean=10)
    frame.zero_data()
    noise = frame.add_noise(x_mean=3, x_std=0.5, noise_type='gaussian')
    assert frame.get_noise_stats() == (3, 0.5)
    np.testing.assert_allclose(frame.data, noise)


def test_get_data_returns_underlying_array():
    frame = make_frame(fchans=16, tchans=4)
    assert frame.get_data() is frame.data
    assert frame.get_data(use_db=False) is frame.data
    assert frame.get_data().shape == (4, 16)


def test_get_data_reflects_added_noise():
    frame = make_frame(fchans=16, tchans=4)
    before = frame.get_data().copy()
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.get_data(), before + noise)


def test_get_data_use_db_converts_to_decibels():
    data = np.full((4, 16), 100.0)
    data[0, 0] = 10.0
    frame = make_frame(fchans=16, tchans=4, data=data)
    db = frame.get_data(use_db=True)
    assert db.shape == (4, 16)
    assert db[1, 1] == pytest.approx(20.0)
    assert db[0, 0] == pytest.approx(10.0)
    np.testing.assert_allclose(db, 10 * np.log10(data))


def test_get_data_use_db_does_not_modify_frame_data():
    data = np.full((4, 16), 100.0)
    frame = make_frame(fchans=16, tchans=4, data=data.copy())
    db = frame.get_data(use_db=True)
    assert db is not frame.data
    np.testing.assert_allclose(frame.data, data)


def test_get_data_use_db_with_zero_gives_negative_infinity():
    frame = make_frame(fchans=8, tchans=2)
    with np.errstate(divide='ignore'):
        db = frame.get_data(use_db=True)
    assert np.all(np.isneginf(db))
