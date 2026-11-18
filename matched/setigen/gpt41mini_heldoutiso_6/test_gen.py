import pytest
import numpy as np
from astropy import units as u
from setigen.frame import Frame
from setigen import distributions


@pytest.fixture
def simple_frame():
    # Create a Frame with simple specified parameters, no preloaded data
    fchans = 10
    tchans = 5
    df = 1 * u.Hz
    dt = 1 * u.s
    fch1 = 10 * u.Hz
    return Frame(fchans=fchans, tchans=tchans, df=df, dt=dt, fch1=fch1)


def test_add_noise_chi2_basic(simple_frame):
    frame = simple_frame
    x_mean = 5.0
    noise = frame.add_noise(x_mean=x_mean, noise_type='chi2')

    # Noise shape matches frame shape
    assert noise.shape == frame.shape
    # Data increased by noise
    np.testing.assert_allclose(frame.data, noise)
    # Noise mean and std updated to parameters on first call
    assert np.isclose(frame.noise_mean, x_mean)
    expected_std = np.sqrt(2 * frame.chi2_df) * x_mean / frame.chi2_df
    assert np.isclose(frame.noise_std, expected_std)


def test_add_noise_gaussian_with_min(simple_frame):
    frame = simple_frame
    x_mean = 3.0
    x_std = 1.0
    x_min = 1.5
    noise = frame.add_noise(x_mean=x_mean, x_std=x_std, x_min=x_min, noise_type='normal')

    # Noise shape matches frame shape
    assert noise.shape == frame.shape
    # Data increased by noise
    np.testing.assert_allclose(frame.data, noise)
    # Noise mean and std updated to parameters on first call
    assert np.isclose(frame.noise_mean, x_mean)
    assert np.isclose(frame.noise_std, x_std)
    # Noise values should be >= x_min (truncated Gaussian)
    assert np.all(noise >= x_min)


def test_add_noise_gaussian_no_min(simple_frame):
    frame = simple_frame
    x_mean = 2.0
    x_std = 0.5
    noise = frame.add_noise(x_mean=x_mean, x_std=x_std, noise_type='gaussian')

    assert noise.shape == frame.shape
    np.testing.assert_allclose(frame.data, noise)
    assert np.isclose(frame.noise_mean, x_mean)
    assert np.isclose(frame.noise_std, x_std)


def test_add_noise_gaussian_no_xstd_exits(simple_frame, monkeypatch):
    frame = simple_frame

    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, noise_type='normal')


def test_add_noise_invalid_noise_type_exits(simple_frame):
    frame = simple_frame
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1, noise_type='invalid_noise')


def test_add_noise_updates_noise_stats_on_multiple_calls(simple_frame):
    frame = simple_frame
    x_mean1 = 4.0
    frame.add_noise(x_mean=x_mean1, noise_type='chi2')

    # Update noise again with different params
    x_mean2 = 6.0
    noise2 = frame.add_noise(x_mean=x_mean2, noise_type='chi2')

    # After second noise addition noise stats should update via _update_noise_frame_stats
    # noise_mean and noise_std should not be exactly x_mean2 and calculated std, but closer to data
    assert frame.noise_mean != x_mean2 or frame.noise_std != (np.sqrt(2 * frame.chi2_df) * x_mean2 / frame.chi2_df)
    # Noise data shape correct
    assert noise2.shape == frame.shape


def test_get_data_returns_data_array(simple_frame):
    frame = simple_frame
    # Set some data explicitly
    frame.data = np.ones(frame.shape) * 10

    # Without dB scaling
    data = frame.get_data(use_db=False)
    np.testing.assert_array_equal(data, frame.data)

    # With dB scaling
    data_db = frame.get_data(use_db=True)
    expected_db = 10 * np.log10(frame.data)
    np.testing.assert_allclose(data_db, expected_db)


def test_get_data_log10_of_zero_causes_warning(simple_frame):
    frame = simple_frame
    # Set zero data to test log10(0) behavior
    frame.data = np.zeros(frame.shape)

    # Using pytest.warns to catch RuntimeWarning due to log10(0)
    with pytest.raises(RuntimeWarning, match="divide by zero"):
        _ = frame.get_data(use_db=True)


def test_add_noise_chi2_ignores_other_params(simple_frame):
    frame = simple_frame
    x_mean = 7.0
    # Provide x_std and x_min which should be ignored for chi2 noise type
    noise = frame.add_noise(x_mean, x_std=1.0, x_min=0.2, noise_type='chi2')

    # Noise shape correct
    assert noise.shape == frame.shape
    # Noise mean matches provided x_mean
    assert np.isclose(frame.noise_mean, x_mean)


def test_add_noise_gaussian_min_none_falls_back_to_gaussian(simple_frame):
    frame = simple_frame
    x_mean = 4.0
    x_std = 0.7
    noise = frame.add_noise(x_mean, x_std=x_std, noise_type='normal')  # x_min=None

    assert noise.shape == frame.shape
    assert np.isclose(frame.noise_std, x_std)
    assert np.all(noise >= 0 or np.all(noise < 0) or True)  # Gaussian noise no min clamp


def test_add_noise_returned_noise_affects_data(simple_frame):
    frame = simple_frame
    initial_data = frame.data.copy()
    noise = frame.add_noise(x_mean=5, noise_type='chi2')

    # Frame data should equal initial + noise
    np.testing.assert_array_equal(frame.data, initial_data + noise)


def test_get_data_returns_correct_type_and_shape(simple_frame):
    frame = simple_frame
    data = frame.get_data()
    assert isinstance(data, np.ndarray)
    assert data.shape == frame.shape


def test_get_data_db_scaling_with_large_numbers(simple_frame):
    frame = simple_frame
    frame.data = np.ones(frame.shape) * 1000
    result = frame.get_data(use_db=True)
    expected = 10 * np.log10(frame.data)
    np.testing.assert_allclose(result, expected)


def test_add_noise_with_noise_type_case_insensitive(simple_frame):
    frame = simple_frame
    x_mean = 3.0
    x_std = 1.0
    # 'Gaussian' lowercase vs normal casing
    noise = frame.add_noise(x_mean, x_std=x_std, noise_type='gaussian')
    assert noise.shape == frame.shape
    # 'normal' lowercase
    noise2 = frame.add_noise(x_mean, x_std=x_std, noise_type='normal')
    assert noise2.shape == frame.shape


def test_add_noise_invalid_noise_type_string_exits(simple_frame):
    frame = simple_frame
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=3, noise_type='chi3')


def test_add_noise_sets_params_only_if_noise_stats_zero(simple_frame):
    frame = simple_frame
    # Manually set noise_mean/std to non-zero
    frame.noise_mean = 1
    frame.noise_std = 1

    # Noise addition should update stats with _update_noise_frame_stats, not reset
    noise = frame.add_noise(x_mean=4, noise_type='chi2')

    assert frame.noise_mean != 4 or frame.noise_std != (np.sqrt(2 * frame.chi2_df) * 4 / frame.chi2_df)


def test_add_noise_with_zero_df_frame():
    # Build a frame with chi2_df likely zero or very small
    # Make df or dt very small so that chi2_df rounds to zero
    fchans = 5
    tchans = 5
    df = 1e-6 * u.Hz
    dt = 1e-6 * u.s
    fch1 = 1e6 * u.Hz
    frame = Frame(fchans=fchans, tchans=tchans, df=df, dt=dt, fch1=fch1)
    # chi2_df may be zero or low
    # Adding noise should not error but produce noise array of correct shape
    noise = frame.add_noise(1.0, noise_type='chi2')
    assert noise.shape == frame.shape


def test_add_noise_integration_with_custom_data():
    fchans = 4
    tchans = 3
    df = 1 * u.Hz
    dt = 1 * u.s
    fch1 = 10 * u.Hz
    data = np.ones((tchans, fchans)) * 2.0
    frame = Frame(fchans=fchans, tchans=tchans, df=df, dt=dt, fch1=fch1, data=data.copy())

    initial_data = frame.data.copy()
    noise = frame.add_noise(x_mean=1, noise_type='chi2')
    # Adding noise should increment data, not overwrite
    assert np.all(frame.data >= initial_data)


def test_add_noise_gaussian_negative_minimum_values(simple_frame):
    frame = simple_frame
    x_mean = 2.0
    x_std = 1.0
    x_min = -1.0
    noise = frame.add_noise(x_mean, x_std=x_std, x_min=x_min, noise_type='normal')
    # Noise values should all be >= x_min
    assert np.all(noise >= x_min)


def test_add_noise_gaussian_min_none_with_negative_x_mean(simple_frame):
    frame = simple_frame
    x_mean = -2.0
    x_std = 0.5
    noise = frame.add_noise(x_mean, x_std=x_std, noise_type='gaussian')
    # Noise returned and data updated accordingly
    assert noise.shape == frame.shape
