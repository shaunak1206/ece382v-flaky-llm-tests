import numpy as np
import pytest
from astropy import units as u
from astropy.stats import sigma_clip

import setigen.frame as frame_module
from setigen.frame import Frame


FCHANS = 16
TCHANS = 8
DF = 2.0  # Hz
DT = 1.0  # s
# Chi-squared degrees of freedom: 4 * round(DF * DT) = 8
CHI2_DF = 8


def make_frame(data=None):
    return Frame(fchans=FCHANS,
                 tchans=TCHANS,
                 df=DF * u.Hz,
                 dt=DT * u.s,
                 fch1=8 * u.GHz,
                 data=data)


class CallRecorder:
    """Stand-in for a distributions function. Records its arguments and
    returns a constant array of the requested shape filled with the mean."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        mean = args[0]
        shape = args[-1]
        return np.full(shape, mean, dtype=float)


def _fail(*args, **kwargs):
    raise AssertionError("This distribution should not have been called")


def test_chi2_calls_distribution_with_mean_df_and_shape(monkeypatch):
    rec = CallRecorder()
    monkeypatch.setattr(frame_module.distributions, 'chi2', rec)
    frame = make_frame()

    frame.add_noise(x_mean=10)

    assert rec.calls == [(10, CHI2_DF, (TCHANS, FCHANS))]


def test_chi2_returns_noise_with_frame_shape(monkeypatch):
    monkeypatch.setattr(frame_module.distributions, 'chi2', CallRecorder())
    frame = make_frame()

    noise = frame.add_noise(x_mean=3)

    assert noise.shape == (TCHANS, FCHANS)


def test_chi2_adds_noise_to_zero_frame(monkeypatch):
    monkeypatch.setattr(frame_module.distributions, 'chi2', CallRecorder())
    frame = make_frame()

    noise = frame.add_noise(x_mean=3)

    np.testing.assert_array_equal(frame.get_data(), noise)


def test_chi2_adds_noise_to_existing_data(monkeypatch):
    monkeypatch.setattr(frame_module.distributions, 'chi2', CallRecorder())
    frame = make_frame(data=np.ones((TCHANS, FCHANS)))

    frame.add_noise(x_mean=3)

    np.testing.assert_array_equal(frame.data, np.full((TCHANS, FCHANS), 4.0))


def test_chi2_accumulates_across_calls(monkeypatch):
    monkeypatch.setattr(frame_module.distributions, 'chi2', CallRecorder())
    frame = make_frame()

    frame.add_noise(x_mean=1)
    frame.add_noise(x_mean=2)

    np.testing.assert_array_equal(frame.data, np.full((TCHANS, FCHANS), 3.0))


def test_chi2_ignores_x_std_and_x_min(monkeypatch):
    rec = CallRecorder()
    monkeypatch.setattr(frame_module.distributions, 'chi2', rec)
    monkeypatch.setattr(frame_module.distributions, 'gaussian', _fail)
    monkeypatch.setattr(frame_module.distributions, 'truncated_gaussian', _fail)
    frame = make_frame()

    frame.add_noise(x_mean=10, x_std=100, x_min=50, noise_type='chi2')

    assert rec.calls == [(10, CHI2_DF, (TCHANS, FCHANS))]


def test_chi2_sets_noise_stats_from_ideal_variance_on_first_call(monkeypatch):
    monkeypatch.setattr(frame_module.distributions, 'chi2', CallRecorder())
    frame = make_frame()

    frame.add_noise(x_mean=4)

    # x_std = sqrt(2 * df) * mean / df = sqrt(16) * 4 / 8 = 2
    assert frame.noise_mean == pytest.approx(4)
    assert frame.noise_std == pytest.approx(2.0)


def test_chi2_second_call_recomputes_stats_with_sigma_clip(monkeypatch):
    monkeypatch.setattr(frame_module.distributions, 'chi2', CallRecorder())
    frame = make_frame()

    frame.add_noise(x_mean=10)
    frame.add_noise(x_mean=10)

    clipped = sigma_clip(frame.data, sigma=3, maxiters=5, masked=False)
    assert frame.noise_mean == pytest.approx(np.mean(clipped))
    assert frame.noise_std == pytest.approx(np.std(clipped))


def test_chi2_returns_noise_that_was_added(monkeypatch):
    monkeypatch.setattr(frame_module.distributions, 'chi2', CallRecorder())
    frame = make_frame(data=np.full((TCHANS, FCHANS), 2.0))

    noise = frame.add_noise(x_mean=5)

    np.testing.assert_array_equal(frame.data - 2.0, noise)


@pytest.mark.parametrize('noise_type', ['normal', 'gaussian'])
def test_gaussian_without_min_calls_gaussian(monkeypatch, noise_type):
    rec = CallRecorder()
    monkeypatch.setattr(frame_module.distributions, 'gaussian', rec)
    monkeypatch.setattr(frame_module.distributions, 'truncated_gaussian', _fail)
    frame = make_frame()

    frame.add_noise(x_mean=5, x_std=2, noise_type=noise_type)

    assert rec.calls == [(5, 2, (TCHANS, FCHANS))]


@pytest.mark.parametrize('noise_type', ['normal', 'gaussian'])
def test_gaussian_with_min_calls_truncated_gaussian(monkeypatch, noise_type):
    rec = CallRecorder()
    monkeypatch.setattr(frame_module.distributions, 'truncated_gaussian', rec)
    monkeypatch.setattr(frame_module.distributions, 'gaussian', _fail)
    frame = make_frame()

    frame.add_noise(x_mean=5, x_std=2, x_min=1, noise_type=noise_type)

    assert rec.calls == [(5, 2, 1, (TCHANS, FCHANS))]


def test_gaussian_sets_noise_stats_to_params_on_first_call(monkeypatch):
    monkeypatch.setattr(frame_module.distributions, 'gaussian', CallRecorder())
    frame = make_frame()

    frame.add_noise(x_mean=5, x_std=2, noise_type='normal')

    assert frame.noise_mean == 5
    assert frame.noise_std == 2


def test_gaussian_adds_returned_noise_to_data(monkeypatch):
    monkeypatch.setattr(frame_module.distributions, 'gaussian', CallRecorder())
    frame = make_frame()

    noise = frame.add_noise(x_mean=5, x_std=2, noise_type='normal')

    np.testing.assert_array_equal(frame.get_data(), noise)


def test_gaussian_without_x_std_exits_and_leaves_data_unchanged(monkeypatch):
    monkeypatch.setattr(frame_module.distributions, 'gaussian', _fail)
    frame = make_frame()

    with pytest.raises(SystemExit, match='x_std must be given'):
        frame.add_noise(x_mean=5, noise_type='normal')

    np.testing.assert_array_equal(frame.data, np.zeros((TCHANS, FCHANS)))


def test_invalid_noise_type_exits(monkeypatch):
    monkeypatch.setattr(frame_module.distributions, 'chi2', _fail)
    frame = make_frame()

    with pytest.raises(SystemExit, match='poisson is not a valid noise type'):
        frame.add_noise(x_mean=5, x_std=2, noise_type='poisson')


def test_get_data_default_returns_underlying_data(monkeypatch):
    frame = make_frame(data=np.full((TCHANS, FCHANS), 7.0))

    assert frame.get_data() is frame.data


def test_get_data_use_db_false_returns_linear_data():
    data = np.random.RandomState(0).rand(TCHANS, FCHANS) + 0.5
    frame = make_frame(data=data)

    np.testing.assert_array_equal(frame.get_data(use_db=False), data)


def test_get_data_use_db_converts_to_decibels():
    frame = make_frame(data=np.full((TCHANS, FCHANS), 100.0))

    result = frame.get_data(use_db=True)

    np.testing.assert_allclose(result, np.full((TCHANS, FCHANS), 20.0))


def test_get_data_use_db_does_not_modify_stored_data():
    data = np.full((TCHANS, FCHANS), 10.0)
    frame = make_frame(data=data.copy())

    frame.get_data(use_db=True)

    np.testing.assert_array_equal(frame.data, data)


def test_get_data_use_db_of_zero_is_negative_infinity():
    frame = make_frame()

    with np.errstate(divide='ignore'):
        result = frame.get_data(use_db=True)

    assert np.all(np.isneginf(result))


def test_get_data_shape_matches_frame_dimensions():
    frame = make_frame()

    assert frame.get_data().shape == (TCHANS, FCHANS)
    assert frame.get_data(use_db=True).shape == (TCHANS, FCHANS)


def test_get_data_reflects_add_noise_result(monkeypatch):
    monkeypatch.setattr(frame_module.distributions, 'chi2', CallRecorder())
    frame = make_frame(data=np.ones((TCHANS, FCHANS)))

    frame.add_noise(x_mean=3)
    result = frame.get_data(use_db=True)

    np.testing.assert_allclose(result, np.full((TCHANS, FCHANS), 10 * np.log10(4.0)))


def test_get_data_from_data_preserves_values():
    data = np.random.RandomState(1).rand(TCHANS, FCHANS)
    frame = Frame.from_data(DF * u.Hz, DT * u.s, 8 * u.GHz, data)

    np.testing.assert_array_equal(frame.get_data(), data)
