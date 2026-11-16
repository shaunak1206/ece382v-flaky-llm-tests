import numpy as np
import pytest
from astropy import units as u

import setigen as stg


def make_frame(fchans=64, tchans=32, df=1.0, dt=4.0, data=None):
    return stg.Frame(fchans=fchans,
                     tchans=tchans,
                     df=df * u.Hz,
                     dt=dt * u.s,
                     fch1=6000 * u.MHz,
                     data=data)


# ---------------------------- add_noise: chi2 ----------------------------

def test_add_noise_chi2_returns_frame_shaped_array():
    np.random.seed(0)
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    assert isinstance(noise, np.ndarray)
    assert noise.shape == frame.shape


def test_add_noise_chi2_adds_noise_to_zero_data():
    np.random.seed(1)
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    assert np.allclose(frame.data, noise)


def test_add_noise_chi2_is_nonnegative():
    np.random.seed(2)
    frame = make_frame()
    noise = frame.add_noise(x_mean=5)
    assert np.all(noise >= 0)


def test_add_noise_chi2_mean_close_to_requested_mean():
    np.random.seed(3)
    frame = make_frame(fchans=256, tchans=64)
    noise = frame.add_noise(x_mean=10)
    assert np.isclose(np.mean(noise), 10, rtol=0.05)


def test_add_noise_chi2_sets_noise_stats_from_parameters():
    np.random.seed(4)
    frame = make_frame()
    frame.add_noise(x_mean=10)
    expected_std = np.sqrt(2 * frame.chi2_df) * 10 / frame.chi2_df
    mean, std = frame.get_noise_stats()
    assert mean == 10
    assert np.isclose(std, expected_std)


def test_add_noise_chi2_ignores_x_std_and_x_min():
    np.random.seed(5)
    frame = make_frame()
    frame.add_noise(x_mean=10, x_std=1000, x_min=500)
    expected_std = np.sqrt(2 * frame.chi2_df) * 10 / frame.chi2_df
    assert np.isclose(frame.noise_std, expected_std)


def test_add_noise_same_seed_is_reproducible():
    frame1 = make_frame()
    frame2 = make_frame()
    np.random.seed(123)
    n1 = frame1.add_noise(x_mean=7)
    np.random.seed(123)
    n2 = frame2.add_noise(x_mean=7)
    assert np.array_equal(n1, n2)
    assert np.array_equal(frame1.data, frame2.data)


# --------------------------- add_noise: gaussian --------------------------

def test_add_noise_gaussian_shape_and_data():
    np.random.seed(6)
    frame = make_frame()
    noise = frame.add_noise(x_mean=10, x_std=2, noise_type='gaussian')
    assert noise.shape == frame.shape
    assert np.allclose(frame.data, noise)


def test_add_noise_gaussian_statistics():
    np.random.seed(7)
    frame = make_frame(fchans=256, tchans=64)
    noise = frame.add_noise(x_mean=10, x_std=2, noise_type='gaussian')
    assert np.isclose(np.mean(noise), 10, atol=0.2)
    assert np.isclose(np.std(noise), 2, rtol=0.1)


def test_add_noise_gaussian_sets_noise_stats_to_parameters():
    np.random.seed(8)
    frame = make_frame()
    frame.add_noise(x_mean=10, x_std=2, noise_type='gaussian')
    assert frame.get_noise_stats() == (10, 2)


def test_add_noise_normal_alias_matches_gaussian():
    frame1 = make_frame()
    frame2 = make_frame()
    np.random.seed(42)
    n1 = frame1.add_noise(x_mean=10, x_std=2, noise_type='normal')
    np.random.seed(42)
    n2 = frame2.add_noise(x_mean=10, x_std=2, noise_type='gaussian')
    assert np.array_equal(n1, n2)


def test_add_noise_gaussian_with_min_respects_lower_bound():
    np.random.seed(9)
    frame = make_frame()
    noise = frame.add_noise(x_mean=1, x_std=2, x_min=0, noise_type='gaussian')
    assert noise.shape == frame.shape
    assert np.all(noise >= 0)
    assert np.all(frame.data >= 0)


def test_add_noise_gaussian_without_std_exits():
    np.random.seed(10)
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=10, noise_type='gaussian')
    assert np.all(frame.data == 0)


def test_add_noise_invalid_type_exits():
    np.random.seed(11)
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=10, x_std=1, noise_type='bogus')
    assert np.all(frame.data == 0)


# ------------------------ add_noise: accumulation ------------------------

def test_add_noise_accumulates_and_updates_stats():
    np.random.seed(12)
    frame = make_frame(fchans=256, tchans=64)
    n1 = frame.add_noise(x_mean=10, x_std=1, noise_type='gaussian')
    n2 = frame.add_noise(x_mean=10, x_std=1, noise_type='gaussian')
    assert np.allclose(frame.data, n1 + n2)
    # Second call re-estimates stats from the data by sigma clipping
    assert np.isclose(frame.noise_mean, 20, rtol=0.05)
    assert np.isclose(frame.noise_std, np.sqrt(2), rtol=0.1)


def test_add_noise_on_preloaded_data_adds_to_existing():
    np.random.seed(13)
    base = np.ones((32, 64))
    frame = make_frame(data=base.copy())
    noise = frame.add_noise(x_mean=3, x_std=0.5, noise_type='gaussian')
    assert np.allclose(frame.data, base + noise)


def test_add_noise_after_zero_data_resets_stats_to_parameters():
    np.random.seed(14)
    frame = make_frame()
    frame.add_noise(x_mean=10, x_std=2, noise_type='gaussian')
    frame.zero_data()
    assert frame.get_noise_stats() == (0, 0)
    frame.add_noise(x_mean=5, x_std=1, noise_type='gaussian')
    assert frame.get_noise_stats() == (5, 1)


# ------------------------------- get_data --------------------------------

def test_get_data_returns_underlying_array():
    np.random.seed(15)
    frame = make_frame()
    assert frame.get_data() is frame.data
    assert frame.get_data().shape == (32, 64)


def test_get_data_reflects_added_noise():
    np.random.seed(16)
    frame = make_frame()
    noise = frame.add_noise(x_mean=4)
    assert np.allclose(frame.get_data(), noise)


def test_get_data_use_db_converts_to_decibels_without_mutating():
    np.random.seed(17)
    data = np.full((4, 8), 100.0)
    frame = make_frame(fchans=8, tchans=4, data=data)
    db = frame.get_data(use_db=True)
    assert db.shape == (4, 8)
    assert np.allclose(db, 20.0)
    assert np.allclose(frame.data, 100.0)


def test_get_data_use_db_matches_log10_of_noise_data():
    np.random.seed(18)
    frame = make_frame()
    frame.add_noise(x_mean=10)
    with np.errstate(divide='ignore'):
        expected = 10 * np.log10(frame.data)
    assert np.allclose(frame.get_data(use_db=True), expected)


def test_get_data_use_db_zero_gives_negative_infinity():
    np.random.seed(19)
    frame = make_frame(fchans=4, tchans=2)
    with np.errstate(divide='ignore'):
        db = frame.get_data(use_db=True)
    assert np.all(np.isneginf(db))
