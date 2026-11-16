import numpy as np
import pytest
from astropy import units as u

import setigen as stg


FCHANS = 128
TCHANS = 64
DF = 2.7939677238464355 * u.Hz
DT = 18.25361108 * u.s


def make_frame(fchans=FCHANS, tchans=TCHANS):
    return stg.Frame(fchans=fchans,
                     tchans=tchans,
                     df=DF,
                     dt=DT,
                     fch1=6095.214842353016 * u.MHz)


# ---------------------------------------------------------------- add_noise

def test_add_noise_chi2_returns_array_of_frame_shape():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    assert isinstance(noise, np.ndarray)
    assert noise.shape == frame.shape == (TCHANS, FCHANS)


def test_add_noise_on_empty_frame_data_equals_noise():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.data, noise)


def test_add_noise_chi2_is_positive_with_expected_mean():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    assert np.all(noise >= 0)
    assert abs(np.mean(noise) - 10) < 0.2


def test_add_noise_chi2_sets_noise_stats_from_parameters_on_empty_frame():
    frame = make_frame()
    assert frame.get_noise_stats() == (0, 0)
    frame.add_noise(x_mean=10)
    expected_std = np.sqrt(2 * frame.chi2_df) * 10 / frame.chi2_df
    assert frame.noise_mean == 10
    assert frame.noise_std == pytest.approx(expected_std)


def test_add_noise_chi2_ignores_x_std_argument():
    frame = make_frame()
    frame.add_noise(x_mean=10, x_std=1000, noise_type='chi2')
    expected_std = np.sqrt(2 * frame.chi2_df) * 10 / frame.chi2_df
    assert frame.noise_std == pytest.approx(expected_std)


def test_add_noise_gaussian_statistics():
    frame = make_frame()
    noise = frame.add_noise(x_mean=5, x_std=2, noise_type='gaussian')
    assert noise.shape == frame.shape
    assert abs(np.mean(noise) - 5) < 0.2
    assert abs(np.std(noise) - 2) < 0.2


def test_add_noise_gaussian_sets_noise_stats_to_parameters():
    frame = make_frame()
    frame.add_noise(x_mean=5, x_std=2, noise_type='gaussian')
    assert frame.noise_mean == 5
    assert frame.noise_std == 2


def test_add_noise_normal_alias_matches_gaussian_behavior():
    frame = make_frame()
    noise = frame.add_noise(x_mean=3, x_std=0.5, noise_type='normal')
    assert noise.shape == frame.shape
    assert abs(np.mean(noise) - 3) < 0.1
    assert frame.get_noise_stats() == (3, 0.5)


def test_add_noise_truncated_gaussian_respects_minimum():
    frame = make_frame()
    noise = frame.add_noise(x_mean=1, x_std=2, x_min=0.5,
                            noise_type='gaussian')
    assert noise.shape == frame.shape
    assert np.all(noise >= 0.5)


def test_add_noise_gaussian_without_std_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=5, noise_type='gaussian')
    # data must be untouched
    assert np.all(frame.data == 0)


def test_add_noise_invalid_type_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=5, x_std=1, noise_type='poisson')
    assert np.all(frame.data == 0)


def test_add_noise_accumulates_across_calls():
    frame = make_frame()
    n1 = frame.add_noise(x_mean=10)
    n2 = frame.add_noise(x_mean=4, x_std=1, noise_type='gaussian')
    np.testing.assert_allclose(frame.data, n1 + n2)


def test_add_noise_second_call_recomputes_noise_stats_from_data():
    frame = make_frame()
    frame.add_noise(x_mean=10)
    frame.add_noise(x_mean=10)
    # Combined mean should be about 20; stats are re-estimated, not reset
    assert abs(frame.noise_mean - 20) < 0.5
    assert frame.noise_std > 0


def test_add_noise_on_preloaded_data_updates_stats_from_data():
    data = np.full((TCHANS, FCHANS), 100.0)
    frame = stg.Frame.from_data(DF, DT, 6095.2 * u.MHz, data)
    frame.add_noise(x_mean=10)
    assert frame.noise_mean > 100
    assert abs(frame.noise_mean - 110) < 0.5


# ----------------------------------------------------------------- get_data

def test_get_data_returns_underlying_array():
    frame = make_frame()
    assert frame.get_data() is frame.data
    assert frame.get_data().shape == (TCHANS, FCHANS)


def test_get_data_default_is_not_db_scaled():
    data = np.full((4, 8), 100.0)
    frame = stg.Frame.from_data(DF, DT, 6095.2 * u.MHz, data)
    np.testing.assert_array_equal(frame.get_data(use_db=False), data)


def test_get_data_use_db_converts_to_decibels():
    data = np.array([[1.0, 10.0, 100.0], [1000.0, 0.1, 1.0]])
    frame = stg.Frame.from_data(DF, DT, 6095.2 * u.MHz, data.copy())
    db = frame.get_data(use_db=True)
    np.testing.assert_allclose(db, [[0, 10, 20], [30, -10, 0]])


def test_get_data_use_db_does_not_modify_frame_data():
    data = np.full((4, 8), 10.0)
    frame = stg.Frame.from_data(DF, DT, 6095.2 * u.MHz, data)
    db = frame.get_data(use_db=True)
    assert db is not frame.data
    np.testing.assert_allclose(frame.data, 10.0)
    np.testing.assert_allclose(db, 10.0)


def test_get_data_use_db_of_zero_is_negative_infinity():
    frame = make_frame(fchans=8, tchans=4)
    with np.errstate(divide='ignore'):
        db = frame.get_data(use_db=True)
    assert db.shape == (4, 8)
    assert np.all(np.isneginf(db))


def test_get_data_reflects_added_noise():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.get_data(), noise)
    np.testing.assert_allclose(frame.get_data(use_db=True),
                               10 * np.log10(noise))
