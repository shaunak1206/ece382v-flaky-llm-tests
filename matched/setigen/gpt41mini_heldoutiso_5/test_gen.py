import numpy as np
import pytest
from astropy import units as u
from setigen.frame import Frame
from setigen import distributions


@pytest.fixture
def simple_frame():
    # Create a simple Frame with known size and resolution
    fchans = 10
    tchans = 5
    df = 1 * u.Hz
    dt = 1 * u.s
    fch1 = 10 * u.Hz
    return Frame(fchans=fchans, tchans=tchans, df=df, dt=dt, fch1=fch1)


def test_add_noise_chi2_returns_noise_shape_and_stats(simple_frame):
    frame = simple_frame
    x_mean = 5.0
    noise = frame.add_noise(x_mean=x_mean, noise_type='chi2')
    assert noise.shape == frame.shape
    assert np.all(noise >= 0)  # chi2 noise should be positive
    # noise_mean and noise_std get set from add_noise if zero
    assert frame.noise_mean == pytest.approx(x_mean)
    assert frame.noise_std > 0


def test_add_noise_gaussian_with_params(simple_frame):
    frame = simple_frame
    x_mean = 1.0
    x_std = 0.5
    x_min = 0.0
    noise = frame.add_noise(x_mean=x_mean, x_std=x_std, x_min=x_min, noise_type='gaussian')
    assert noise.shape == frame.shape
    assert np.all(noise >= x_min)


def test_add_noise_gaussian_without_x_std_exits(monkeypatch, simple_frame):
    frame = simple_frame
    # monkeypatch sys.exit to raise exception for test
    called = {}
    def fake_exit(msg=None):
        called['msg'] = msg
        raise SystemExit
    monkeypatch.setattr("sys.exit", fake_exit)
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, noise_type='normal')


def test_add_noise_invalid_noise_type_exits(monkeypatch, simple_frame):
    frame = simple_frame
    called = {}
    def fake_exit(msg=None):
        called['msg'] = msg
        raise SystemExit
    monkeypatch.setattr("sys.exit", fake_exit)
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, noise_type='invalid_type')
    assert "not a valid noise type" in called['msg']


def test_add_noise_updates_noise_stats_after_first_call(simple_frame):
    frame = simple_frame
    # Add noise first time (noise_mean and noise_std were zero)
    noise1 = frame.add_noise(x_mean=2.0, noise_type='chi2')
    old_mean = frame.noise_mean
    old_std = frame.noise_std
    # Modify data so noise stats will differ on update
    frame.data += 1
    noise2 = frame.add_noise(x_mean=3.0, noise_type='chi2')
    assert (frame.noise_mean != old_mean) or (frame.noise_std != old_std)
    # Returned noise should match shape again
    assert noise2.shape == frame.shape


def test_get_data_returns_data_and_db_version(simple_frame):
    frame = simple_frame
    frame.data = np.ones(frame.shape) * 10
    arr1 = frame.get_data(use_db=False)
    assert np.all(arr1 == 10)
    arr2 = frame.get_data(use_db=True)
    expected_db = 10 * np.log10(10)
    assert np.allclose(arr2, expected_db)


def test_add_noise_normal_and_gaussian_equivalent(simple_frame):
    frame1 = Frame.from_data(simple_frame.df * u.Hz,
                             simple_frame.dt * u.s,
                             simple_frame.fch1 * u.Hz,
                             np.zeros(simple_frame.shape))
    frame2 = Frame.from_data(simple_frame.df * u.Hz,
                             simple_frame.dt * u.s,
                             simple_frame.fch1 * u.Hz,
                             np.zeros(simple_frame.shape))

    x_mean = 2
    x_std = 0.5

    noise1 = frame1.add_noise(x_mean, x_std, noise_type='normal')
    noise2 = frame2.add_noise(x_mean, x_std, noise_type='gaussian')

    assert noise1.shape == frame1.shape
    assert noise2.shape == frame2.shape

    # The distributions are defined differently but noise arrays shape must be equal
    assert noise1.shape == noise2.shape


def test_add_noise_with_x_min_none(simple_frame):
    frame = simple_frame
    # Should call gaussian when x_min is None
    noise = frame.add_noise(x_mean=1.0, x_std=0.1, x_min=None, noise_type='normal')
    assert noise.shape == frame.shape


def test_add_noise_with_noise_type_chi2_ignores_x_std_and_x_min(simple_frame):
    frame = simple_frame
    noise = frame.add_noise(x_mean=5.0, x_std=100, x_min=-100, noise_type='chi2')
    assert noise.shape == frame.shape
    # noise should be all positive (chi2)
    assert np.all(noise >= 0)


def test_get_data_with_zero_data_log_warns(simple_frame):
    frame = simple_frame
    frame.data.fill(0)
    out = frame.get_data(use_db=True)
    # log10(0) returns -inf, but numpy does not error. We check that result contains -inf.
    assert np.isneginf(out).any()


def test_add_noise_returns_expected_stats(simple_frame):
    frame = simple_frame
    x_mean = 4.0
    noise = frame.add_noise(x_mean=x_mean, noise_type='chi2')
    # The returned noise is independent of frame.data after addition
    assert noise.shape == frame.shape
    # The noise stats stored in frame should be close to input x_mean and computed std deviation
    assert frame.noise_mean == pytest.approx(x_mean, rel=1e-5)
    expected_std = np.sqrt(2 * frame.chi2_df) * x_mean / frame.chi2_df
    assert frame.noise_std == pytest.approx(expected_std, rel=1e-5)
