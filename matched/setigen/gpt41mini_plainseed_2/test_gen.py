import pytest
import numpy as np
import setigen as stg
from astropy import units as u


@pytest.fixture
def example_frame():
    fchans = 100
    tchans = 50
    df = 1 * u.Hz
    dt = 1 * u.s
    fch1 = 1000 * u.Hz
    return stg.Frame(fchans=fchans, tchans=tchans, df=df, dt=dt, fch1=fch1)


def test_add_noise_chi2_basic(example_frame):
    frame = example_frame
    noise = frame.add_noise(x_mean=10)
    assert noise.shape == frame.shape
    assert np.all(noise >= 0)
    # Noise added to data
    assert np.any(frame.data != 0)
    # noise_mean and noise_std updated to initial values
    assert frame.noise_mean == 10
    std_expected = np.sqrt(2 * frame.chi2_df) * 10 / frame.chi2_df
    assert np.isclose(frame.noise_std, std_expected)


def test_add_noise_chi2_multiple_calls_updates_noise_stats(example_frame):
    frame = example_frame
    noise1 = frame.add_noise(x_mean=10)
    data_before = frame.data.copy()
    noise2 = frame.add_noise(x_mean=15)
    # Data should have increased values
    assert np.all(frame.data >= data_before)
    # noise_mean and noise_std are updated (not equal to params exactly)
    assert frame.noise_mean != 10
    assert frame.noise_std != 0


def test_add_noise_gaussian_with_std_and_min(example_frame):
    frame = example_frame
    noise = frame.add_noise(x_mean=5, x_std=1, x_min=0, noise_type='gaussian')
    assert noise.shape == frame.shape
    # noise_min passed, so noise values should be >= x_min
    assert np.all(noise >= 0)
    assert frame.noise_mean == 5
    assert frame.noise_std == 1


def test_add_noise_gaussian_with_std_no_min(example_frame):
    frame = example_frame
    noise = frame.add_noise(x_mean=5, x_std=2, noise_type='normal')
    assert noise.shape == frame.shape
    assert frame.noise_mean == 5
    assert frame.noise_std == 2


def test_add_noise_gaussian_missing_std_sys_exit(example_frame):
    frame = example_frame
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=5, noise_type='normal')


def test_add_noise_invalid_noise_type_sys_exit(example_frame):
    frame = example_frame
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=5, noise_type='invalid_noise')


def test_get_data_without_db_returns_data_array(example_frame):
    data = example_frame.get_data(use_db=False)
    assert isinstance(data, np.ndarray)
    np.testing.assert_array_equal(data, example_frame.data)


def test_get_data_with_db_returns_log10_of_data(example_frame):
    frame = example_frame
    frame.data = np.ones(frame.shape) * 10
    data_db = frame.get_data(use_db=True)
    expected = 10 * np.log10(frame.data)
    np.testing.assert_array_almost_equal(data_db, expected)


def test_get_data_with_zero_and_negative_values_db(example_frame):
    frame = example_frame
    # Set some zero and negative values in data
    frame.data[0, 0] = 0
    frame.data[0, 1] = -1
    with pytest.warns(RuntimeWarning):
        # Log of zero or negative will raise runtime warning
        _ = frame.get_data(use_db=True)
    # Returned array shape matches
    assert _.shape == frame.shape


def test_add_noise_sets_noise_params_if_zero_stats(example_frame):
    frame = example_frame
    # Initially noise_mean and noise_std are 0
    assert frame.noise_mean == 0
    assert frame.noise_std == 0
    frame.add_noise(x_mean=7, x_std=1.5, noise_type='gaussian')
    assert frame.noise_mean == 7
    assert frame.noise_std == 1.5


def test_add_noise_updates_noise_stats_if_already_set(example_frame):
    frame = example_frame
    # First call sets noise_mean and noise_std
    frame.add_noise(x_mean=7, x_std=1.5, noise_type='gaussian')
    # Save current stats
    mean1, std1 = frame.noise_mean, frame.noise_std
    # Second call should update stats, usually different than old
    frame.add_noise(x_mean=8, x_std=1.3, noise_type='gaussian')
    mean2, std2 = frame.noise_mean, frame.noise_std
    assert (mean1 != mean2) or (std1 != std2)


def test_add_noise_with_float_and_int_x_mean(example_frame):
    frame = example_frame
    # int x_mean
    noise_int = frame.add_noise(x_mean=5, noise_type='chi2')
    # float x_mean
    noise_float = frame.add_noise(x_mean=5.0, noise_type='chi2')
    assert noise_int.shape == noise_float.shape == frame.shape


def test_add_noise_with_zero_x_mean(example_frame):
    frame = example_frame
    noise = frame.add_noise(x_mean=0, noise_type='chi2')
    # Noise should be zero array
    assert np.allclose(noise, 0)


def test_add_noise_with_negative_x_min_gaussian(example_frame):
    frame = example_frame
    noise = frame.add_noise(x_mean=5, x_std=1, x_min=-10, noise_type='normal')
    # noise values should be >= x_min (negative allowed)
    assert np.all(noise >= -10)


def test_add_noise_with_none_x_min_gaussian(example_frame):
    frame = example_frame
    noise = frame.add_noise(x_mean=5, x_std=1, x_min=None, noise_type='normal')
    assert noise.shape == frame.shape


def test_add_noise_with_x_std_zero_gaussian(example_frame):
    frame = example_frame
    noise = frame.add_noise(x_mean=5, x_std=0, noise_type='normal')
    # Noise should be nearly all equal to x_mean (5)
    assert np.allclose(noise, 5)


def test_get_data_db_with_small_values(example_frame):
    frame = example_frame
    frame.data = np.full(frame.shape, 1e-9)
    data_db = frame.get_data(use_db=True)
    # log10 of very small number negative but finite
    assert np.all(data_db < 0)
    assert data_db.shape == frame.shape


def test_add_noise_returned_noise_matches_data_diff(example_frame):
    frame = example_frame
    data_before = frame.data.copy()
    noise = frame.add_noise(x_mean=10)
    diff = frame.data - data_before
    np.testing.assert_array_equal(noise, diff)


def test_add_noise_chi2_noise_std_calc_correct(example_frame):
    frame = example_frame
    x_mean = 10.0
    noise = frame.add_noise(x_mean=x_mean, noise_type='chi2')
    expected_std = np.sqrt(2 * frame.chi2_df) * x_mean / frame.chi2_df
    assert np.isclose(frame.noise_std, expected_std)


def test_get_data_returns_floats(example_frame):
    data = example_frame.get_data()
    assert np.issubdtype(data.dtype, np.floating)


def test_add_noise_with_np_float_x_mean(example_frame):
    frame = example_frame
    x_mean = np.float64(8.5)
    noise = frame.add_noise(x_mean=x_mean, noise_type='chi2')
    assert noise.shape == frame.shape
    assert np.issubdtype(noise.dtype, np.floating)
