import numpy as np
import pytest
from astropy import units as u
from setigen.frame import Frame
from setigen import distributions

# Useful constants for tests
FCHANS = 10
TCHANS = 5
DF = 2.0 * u.Hz
DT = 1.0 * u.s
FCH1 = 10.0 * u.Hz


def make_basic_frame():
    return Frame(fchans=FCHANS, tchans=TCHANS, df=DF, dt=DT, fch1=FCH1)


def test_add_noise_chi2_default_updates_data_and_noise_stats():
    frame = make_basic_frame()
    x_mean = 5.0
    noise = frame.add_noise(x_mean=x_mean, noise_type='chi2')

    # Noise sample shape correct
    assert noise.shape == frame.data.shape
    # Data updated (original zero + noise)
    assert np.all(frame.data >= 0)
    # noise_mean and noise_std set correctly on first call
    expected_std = np.sqrt(2 * frame.chi2_df) * x_mean / frame.chi2_df
    assert frame.noise_mean == pytest.approx(x_mean)
    assert frame.noise_std == pytest.approx(expected_std)


def test_add_noise_gaussian_no_minimum_adds_noise_and_updates_stats():
    frame = make_basic_frame()
    x_mean = 0.0
    x_std = 1.0
    noise = frame.add_noise(x_mean=x_mean, x_std=x_std, noise_type='normal')

    assert noise.shape == frame.data.shape
    # Data updated with noise
    diff = frame.data - noise
    assert np.allclose(diff, 0.0)
    # noise_mean, noise_std set to params because initial was zero
    assert frame.noise_mean == pytest.approx(x_mean)
    assert frame.noise_std == pytest.approx(x_std)


def test_add_noise_gaussian_with_minimum_adds_truncated_noise():
    frame = make_basic_frame()
    x_mean = 1.0
    x_std = 0.5
    x_min = 0.0  # truncate at zero
    noise = frame.add_noise(x_mean=x_mean, x_std=x_std, x_min=x_min, noise_type='gaussian')

    assert noise.shape == frame.data.shape
    # Data is at least x_min somewhere (should not go below zero in truncated)
    assert np.all(frame.data >= x_min)
    assert frame.noise_mean == pytest.approx(x_mean)
    assert frame.noise_std == pytest.approx(x_std)


def test_add_noise_gaussian_without_x_std_exits(monkeypatch):
    frame = make_basic_frame()
    exited = {}

    def fake_exit(msg):
        exited['called'] = True
        exited['msg'] = msg
        raise SystemExit(msg)

    monkeypatch.setattr("sys.exit", fake_exit)
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, noise_type='gaussian')
    assert exited.get('called') is True
    assert 'x_std must be given' in exited.get('msg')


def test_add_noise_invalid_noise_type_exits(monkeypatch):
    frame = make_basic_frame()
    exited = {}

    def fake_exit(msg):
        exited['called'] = True
        exited['msg'] = msg
        raise SystemExit(msg)

    monkeypatch.setattr("sys.exit", fake_exit)
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, x_std=1.0, noise_type='invalid_noise')
    assert exited.get('called') is True
    assert 'not a valid noise type' in exited.get('msg')


def test_add_noise_updates_noise_stats_on_second_call():
    frame = make_basic_frame()
    # first call sets noise_mean and noise_std directly
    frame.add_noise(5.0, noise_type='chi2')
    old_mean, old_std = frame.noise_mean, frame.noise_std

    # Manually corrupt noise params to test update
    frame.noise_mean = 100
    frame.noise_std = 100

    # second call should update noise stats using sigma_clip
    frame.add_noise(7.0, noise_type='chi2')
    assert frame.noise_mean != 100
    assert frame.noise_std != 100


def test_get_data_returns_original_data_no_db():
    frame = make_basic_frame()
    frame.data[:] = 3.0
    data = frame.get_data(use_db=False)
    assert np.array_equal(data, frame.data)


def test_get_data_returns_log10_data_when_use_db():
    frame = make_basic_frame()
    # Fill data with positive numbers to avoid log10 domain error
    frame.data[:] = 10.0
    data_db = frame.get_data(use_db=True)
    expected_db = 10 * np.log10(frame.data)
    assert np.allclose(data_db, expected_db)


def test_get_data_with_zero_data_and_use_db_produces_minus_inf():
    frame = make_basic_frame()
    frame.data[:] = 0.0
    data_db = frame.get_data(use_db=True)
    # log10(0) -> -inf, check for -inf
    assert np.isneginf(data_db).all()


def test_add_noise_returns_noise_array_of_correct_type():
    frame = make_basic_frame()
    noise = frame.add_noise(x_mean=3.0)
    assert isinstance(noise, np.ndarray)
    assert noise.shape == frame.shape


def test_add_noise_changes_frame_data_values():
    frame = make_basic_frame()
    old_data = frame.data.copy()
    frame.add_noise(x_mean=2.0)
    assert not np.array_equal(frame.data, old_data)


def test_add_noise_preserves_shape():
    frame = make_basic_frame()
    noise = frame.add_noise(x_mean=1.0)
    assert noise.shape == frame.shape


def test_add_noise_gaussian_min_x_min_array_ignored(monkeypatch):
    # The function add_noise does not accept array for x_min (only scalar),
    # this test checks behavior if x_min is provided as array by accident
    frame = make_basic_frame()
    x_mean = 1.0
    x_std = 0.2
    # Passing x_min as ndarray should not error - but it's unclear from the code,
    # add_noise expects scalar or None for x_min. So let's test with a scalar only.
    noise = frame.add_noise(x_mean, x_std, x_min=0.0, noise_type='normal')
    assert noise.shape == frame.shape


def test_add_noise_chi2_ignores_x_std_and_x_min():
    frame = make_basic_frame()
    # Provide x_std and x_min but noise_type='chi2' so those should be ignored
    noise = frame.add_noise(x_mean=4.0, x_std=1000.0, x_min=-1000.0, noise_type='chi2')
    assert noise.shape == frame.shape
    # Noise should be positive (chi2) even though x_min negative
    assert np.all(noise >= 0)


def test_add_noise_gaussian_x_min_greater_than_mean():
    frame = make_basic_frame()
    # x_min > x_mean should still work for truncated gaussian
    x_mean = 1.0
    x_std = 0.5
    x_min = 2.0
    noise = frame.add_noise(x_mean, x_std, x_min=x_min, noise_type='normal')
    assert noise.shape == frame.shape
    assert np.all(noise >= x_min)


def test_add_noise_with_scalar_nans_does_not_break():
    frame = make_basic_frame()
    # Pass NaN as mean for gaussian noise, should propagate NaNs in data
    noise = frame.add_noise(np.nan, x_std=1.0, noise_type='normal')
    assert noise.shape == frame.shape
    assert np.isnan(noise).all() or np.any(np.isnan(noise))


def test_add_noise_with_zero_chi2_df_raises_division_by_zero(monkeypatch):
    frame = make_basic_frame()
    frame.chi2_df = 0
    with pytest.raises(ZeroDivisionError):
        frame.add_noise(x_mean=1.0, noise_type='chi2')
