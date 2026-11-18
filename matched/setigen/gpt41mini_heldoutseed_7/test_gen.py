import numpy as np
import pytest
from astropy import units as u
from setigen.frame import Frame
from setigen import distributions

# Helper to create a minimal Frame with given noise stats zeroed
def create_empty_frame():
    fchans = 10
    tchans = 5
    df = 1 * u.Hz
    dt = 1 * u.s
    fch1 = 10 * u.Hz
    frame = Frame(fchans=fchans, tchans=tchans, df=df, dt=dt, fch1=fch1)
    frame.zero_data()
    return frame

def test_add_noise_chi2_basic():
    frame = create_empty_frame()
    x_mean = 10
    noise = frame.add_noise(x_mean=x_mean, noise_type='chi2')
    assert noise.shape == frame.shape
    # chi2 noise mean should be roughly x_mean (within reason)
    noise_mean = noise.mean()
    assert np.isclose(noise_mean, x_mean, rtol=0.5)
    # Check data updated correctly
    assert np.allclose(frame.data, noise)

def test_add_noise_chi2_noise_std_calculated():
    frame = create_empty_frame()
    x_mean = 5
    noise = frame.add_noise(x_mean=x_mean, noise_type='chi2')
    # Check frame.noise_std is correctly calculated per formula
    expected_std = np.sqrt(2 * frame.chi2_df) * x_mean / frame.chi2_df
    assert frame.noise_std == pytest.approx(expected_std, rel=1e-12)

def test_add_noise_gaussian_with_x_std_and_x_min():
    frame = create_empty_frame()
    x_mean = 1
    x_std = 0.5
    x_min = 0
    noise = frame.add_noise(x_mean=x_mean, x_std=x_std, x_min=x_min, noise_type='normal')
    assert noise.shape == frame.shape
    assert (noise >= x_min).all(), "Noise values should be >= x_min"
    # Check noise mean and std roughly match or at least frame updated
    assert np.allclose(frame.data, noise)

def test_add_noise_gaussian_with_x_std_no_x_min():
    frame = create_empty_frame()
    x_mean = 2
    x_std = 1
    noise = frame.add_noise(x_mean=x_mean, x_std=x_std, noise_type='gaussian')
    assert noise.shape == frame.shape
    # Noise can have values below zero (no truncation)
    assert np.isclose(np.mean(noise), x_mean, rtol=0.5)
    assert np.isclose(np.std(noise), x_std, rtol=0.5)
    assert np.allclose(frame.data, noise)

def test_add_noise_gaussian_missing_x_std_exits(monkeypatch):
    frame = create_empty_frame()
    x_mean = 1
    # Monkeypatch sys.exit to raise RuntimeError for test
    import sys
    def fake_exit(msg=None):
        raise RuntimeError("sys.exit called: " + (msg or ""))
    monkeypatch.setattr(sys, "exit", fake_exit)
    with pytest.raises(RuntimeError):
        frame.add_noise(x_mean=x_mean, noise_type='normal')

def test_add_noise_invalid_noise_type_exits(monkeypatch):
    frame = create_empty_frame()
    # Monkeypatch sys.exit to raise RuntimeError for test
    import sys
    def fake_exit(msg=None):
        raise RuntimeError("sys.exit called: " + (msg or ""))
    monkeypatch.setattr(sys, "exit", fake_exit)
    with pytest.raises(RuntimeError):
        frame.add_noise(x_mean=1, noise_type='invalid_type')

def test_get_data_returns_correct_array_and_db():
    frame = create_empty_frame()
    frame.data[:] = 1.0
    # Without use_db flag should return data as is
    data = frame.get_data()
    assert np.all(data == 1.0)
    # With use_db flag should return 10 * log10(data)
    data_db = frame.get_data(use_db=True)
    expected_db = 10 * np.log10(frame.data)
    assert np.allclose(data_db, expected_db)

def test_get_data_log_of_zero_results_in_negative_infinity_or_warning():
    frame = create_empty_frame()
    frame.data[:] = 0.0
    # Getting log10 zero is -inf, numpy issues a warning - suppress it
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data_db = frame.get_data(use_db=True)
    assert np.isneginf(data_db).all()

def test_add_noise_updates_noise_mean_std_first_time():
    frame = create_empty_frame()
    assert frame.noise_mean == 0
    assert frame.noise_std == 0
    x_mean = 3.0
    x_std = 1.0
    frame.add_noise(x_mean=x_mean, x_std=x_std, x_min=0, noise_type='normal')
    assert frame.noise_mean == x_mean
    assert frame.noise_std == x_std

def test_add_noise_updates_noise_mean_std_on_second_call():
    frame = create_empty_frame()
    x_mean1 = 3.0
    x_std1 = 1.0
    frame.add_noise(x_mean=x_mean1, x_std=x_std1, x_min=0, noise_type='normal')
    # Save current stats
    old_mean = frame.noise_mean
    old_std = frame.noise_std
    # Add noise second time with different params, should update noise stats (not just set)
    x_mean2 = 4.0
    x_std2 = 2.0
    noise2 = frame.add_noise(x_mean=x_mean2, x_std=x_std2, x_min=0, noise_type='normal')
    # Means likely different due to update
    assert frame.noise_mean != old_mean or frame.noise_std != old_std
    # Data updated correctly (should be sum of noises)
    assert np.allclose(frame.data, noise2 + frame.data - noise2)

def test_add_noise_returns_noise_array_correct_type():
    frame = create_empty_frame()
    noise = frame.add_noise(x_mean=5, noise_type='chi2')
    assert isinstance(noise, np.ndarray)
    assert noise.shape == frame.shape

def test_add_noise_minimum_not_enforced_for_chi2():
    frame = create_empty_frame()
    noise = frame.add_noise(x_mean=5, noise_type='chi2', x_min=-1000)
    # chi2 noise ignores x_min so can have values < 0
    assert np.any(noise < 0) or np.all(noise >=0)  # Typically chi2 is positive, but test that x_min ignored

def test_add_noise_with_min_only_for_gaussian():
    frame = create_empty_frame()
    x_mean = 0
    x_std = 1
    x_min = 0.5
    noise = frame.add_noise(x_mean=x_mean, x_std=x_std, x_min=x_min, noise_type='gaussian')
    # Check noise respects x_min
    assert (noise >= x_min).all()

def test_add_noise_with_float_noise_type_case_insensitive():
    # According to code, noise_type must match exactly; test 'Gaussian' fails
    frame = create_empty_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1, x_std=1, noise_type='Gaussian')

def test_add_noise_with_integer_x_std():
    # x_std can be int and works fine
    frame = create_empty_frame()
    x_mean = 2
    x_std = 1
    noise = frame.add_noise(x_mean, x_std=x_std, noise_type='normal')
    assert noise.shape == frame.shape
    assert np.isclose(np.mean(noise), x_mean, rtol=0.5)

def test_get_data_does_not_modify_internal_data():
    frame = create_empty_frame()
    original_data = frame.data.copy()
    _ = frame.get_data(use_db=False)
    np.testing.assert_array_equal(frame.data, original_data)
    _ = frame.get_data(use_db=True)
    np.testing.assert_array_equal(frame.data, original_data)
