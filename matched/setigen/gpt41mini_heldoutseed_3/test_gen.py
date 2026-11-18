import numpy as np
import pytest
from astropy import units as u
from setigen.frame import Frame
from setigen import distributions


@pytest.fixture
def frame():
    # A simple frame with known parameters
    fchans = 16
    tchans = 8
    df = 1 * u.Hz
    dt = 1 * u.s
    fch1 = 100 * u.Hz
    return Frame(fchans=fchans, tchans=tchans, df=df, dt=dt, fch1=fch1)


def test_add_noise_chi2_basic(frame):
    noise = frame.add_noise(x_mean=5)
    # The returned noise should have the same shape as the frame
    assert noise.shape == frame.shape
    # Data has increased from zeros
    assert np.all(frame.data >= 0)
    # noise_mean and noise_std updated correctly
    assert frame.noise_mean == 5
    # x_std calculated from chi2 formula sqrt(2*df)*x_mean/df
    expected_std = np.sqrt(2 * frame.chi2_df) * 5 / frame.chi2_df
    assert pytest.approx(frame.noise_std, rel=1e-9) == expected_std


def test_add_noise_gaussian_without_min(frame):
    noise = frame.add_noise(x_mean=5, x_std=1, noise_type='gaussian')
    assert noise.shape == frame.shape
    # Values should vary around x_mean=5
    mean_noise = np.mean(noise)
    assert abs(mean_noise - 5) < 3  # within few std devs
    # noise parameters should be set
    assert frame.noise_mean == 5
    assert frame.noise_std == 1


def test_add_noise_gaussian_with_min(frame):
    noise = frame.add_noise(x_mean=5, x_std=1, x_min=4, noise_type='normal')
    assert noise.shape == frame.shape
    # All values should be >= 4 since truncated_gaussian applies min
    assert np.all(noise >= 4)
    assert frame.noise_mean == 5
    assert frame.noise_std == 1


def test_add_noise_gaussian_missing_x_std(frame):
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1, noise_type='normal')


def test_add_noise_invalid_type(frame):
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1, noise_type='invalid_type')


def test_add_noise_updates_stats_after_initial(frame):
    # Add noise first time
    frame.add_noise(x_mean=5)
    old_mean, old_std = frame.noise_mean, frame.noise_std
    # Add noise second time, stats should update (not remain fixed)
    noise = frame.add_noise(x_mean=7)
    # noise_mean and noise_std should not be the same as old
    assert (frame.noise_mean != old_mean) or (frame.noise_std != old_std)
    assert noise.shape == frame.shape


def test_get_data_default_and_db(frame):
    # Set some positive data for meaningful log10
    frame.data += 10
    data_linear = frame.get_data()
    data_db = frame.get_data(use_db=True)
    assert np.allclose(data_linear, frame.data)
    # Log10 values shape match original
    assert data_db.shape == frame.data.shape
    # db values != linear values generally
    assert not np.allclose(data_db, frame.data)


def test_get_data_use_db_with_zero_or_negative_data(frame):
    frame.data = np.zeros(frame.shape)
    # np.log10(0) is -inf, check result contains -inf
    data_db = frame.get_data(use_db=True)
    assert np.isneginf(data_db).any()
    # Set negative values (should still be handled without crashing)
    frame.data = -np.abs(np.random.randn(*frame.shape))
    data_db = frame.get_data(use_db=True)
    # log10 of negative is nan; check there are nans
    assert np.isnan(data_db).any()


def test_add_noise_chi2_then_get_data(frame):
    # Add chi2 noise and then verify data returned matches
    noise = frame.add_noise(10, noise_type='chi2')
    retrieved_data = frame.get_data()
    assert np.allclose(frame.data, retrieved_data)
    # noise returned matches frame.data - original zeros (approximately)
    assert np.allclose(noise, retrieved_data, rtol=1)


def test_add_noise_with_initial_zero_stats(frame):
    # noise_mean and noise_std should initially be zero
    assert frame.noise_mean == 0
    assert frame.noise_std == 0
    noise = frame.add_noise(2)
    # After adding noise, noise_mean and noise_std should be set to param values
    assert frame.noise_mean == 2
    assert frame.noise_std > 0


def test_add_noise_with_existing_noise_stats(frame):
    frame.noise_mean = 1.0
    frame.noise_std = 0.5
    old_data = frame.data.copy()
    noise = frame.add_noise(3)
    # noise added to data
    assert not np.array_equal(frame.data, old_data)
    # stats should update (not equal to old noise stats)
    assert frame.noise_mean != 1.0 or frame.noise_std != 0.5


def test_add_noise_with_min_none_no_min_addition(frame):
    # Gaussian noise, with x_min=None should call gaussian not truncated_gaussian
    noise = frame.add_noise(x_mean=5, x_std=1, x_min=None, noise_type='normal')
    assert noise.shape == frame.shape


def test_add_noise_multiple_calls_accumulate(frame):
    frame.add_noise(1)
    data1 = frame.data.copy()
    frame.add_noise(1)
    data2 = frame.data.copy()
    # Data should increase after second addition
    assert np.all(data2 >= data1)


def test_add_noise_zero_x_mean(frame):
    noise = frame.add_noise(0)
    assert np.all(noise >= 0)
    assert frame.noise_mean == 0
    # std should be zero (or close to)
    assert frame.noise_std == 0


def test_add_noise_with_integer_noise_type(frame):
    # noise_type is case sensitive, integers invalid; should raise SystemExit
    with pytest.raises(SystemExit):
        frame.add_noise(1, noise_type=123)


def test_add_noise_with_noise_type_uppercase(frame):
    # noise_type not recognized in uppercase: raises SystemExit
    with pytest.raises(SystemExit):
        frame.add_noise(1, noise_type='CHI2')


def test_add_noise_with_float_x_std_zero(frame):
    # x_std=0 is valid in gaussian, should produce noise all equal to x_mean
    noise = frame.add_noise(x_mean=5, x_std=0, noise_type='gaussian')
    assert np.allclose(noise, 5)
    assert frame.noise_mean == 5
    assert frame.noise_std == 0


def test_get_data_no_modification(frame):
    # Data initially zeros
    data1 = frame.get_data()
    data2 = frame.get_data(use_db=False)
    # Both return the same array (numpy zeros)
    assert np.array_equal(data1, data2)


def test_add_noise_chi2_returns_correct_shape_and_type(frame):
    noise = frame.add_noise(10)
    assert isinstance(noise, np.ndarray)
    assert noise.shape == frame.shape
    assert np.issubdtype(noise.dtype, np.floating)


def test_add_noise_gaussian_negative_x_std(frame):
    # Provide negative x_std: gaussian allows but results may be unexpected, 
    # test it does not crash (distribution.gaussian may handle this)
    noise = frame.add_noise(x_mean=5, x_std=-1, noise_type='gaussian')
    assert noise.shape == frame.shape
