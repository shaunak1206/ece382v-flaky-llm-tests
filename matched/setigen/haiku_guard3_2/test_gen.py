import numpy as np
import pytest
from astropy import units as u
from astropy.stats import sigma_clip

from setigen import distributions
from setigen.frame import Frame


def make_frame(fchans=8, tchans=4, data=None):
    # df * dt = 1 gives chi2_df = 4 * round(1) = 4
    return Frame(fchans=fchans,
                 tchans=tchans,
                 df=1 * u.Hz,
                 dt=1 * u.s,
                 data=data)


def test_add_noise_chi2_returns_noise_with_frame_shape():
    np.random.seed(1001)
    frame = make_frame(fchans=8, tchans=4)
    noise = frame.add_noise(x_mean=10)
    assert noise.shape == frame.shape == (4, 8)


def test_add_noise_chi2_adds_returned_noise_to_zero_data():
    np.random.seed(1002)
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_array_equal(frame.get_data(), noise)
    # The returned array must be a separate object from the frame data
    assert noise is not frame.get_data()


def test_add_noise_chi2_is_reproducible_with_seed():
    np.random.seed(1003)
    frame_a = make_frame()
    noise_a = frame_a.add_noise(x_mean=5)

    np.random.seed(1003)
    frame_b = make_frame()
    noise_b = frame_b.add_noise(x_mean=5)

    np.testing.assert_array_equal(noise_a, noise_b)
    np.testing.assert_array_equal(frame_a.get_data(), frame_b.get_data())


def test_add_noise_chi2_sample_mean_matches_x_mean():
    np.random.seed(1004)
    frame = make_frame(fchans=200, tchans=200)
    x_mean = 10.0
    noise = frame.add_noise(x_mean=x_mean)
    assert np.mean(noise) == pytest.approx(x_mean, rel=0.03)


def test_add_noise_chi2_ignores_x_std():
    np.random.seed(1005)
    frame_a = make_frame()
    noise_a = frame_a.add_noise(x_mean=3)

    np.random.seed(1005)
    frame_b = make_frame()
    noise_b = frame_b.add_noise(x_mean=3, x_std=50)

    np.testing.assert_array_equal(noise_a, noise_b)


def test_add_noise_first_call_sets_noise_stats_from_parameters():
    np.random.seed(1006)
    frame = make_frame()
    x_mean = 7.0
    frame.add_noise(x_mean=x_mean)
    noise_mean, noise_std = frame.get_noise_stats()
    expected_std = np.sqrt(2 * frame.chi2_df) * x_mean / frame.chi2_df
    assert noise_mean == pytest.approx(x_mean)
    assert noise_std == pytest.approx(expected_std)


def test_add_noise_second_call_recomputes_noise_stats_from_data():
    np.random.seed(1007)
    frame = make_frame(fchans=32, tchans=32)
    frame.add_noise(x_mean=4)
    frame.add_noise(x_mean=6)

    clipped = sigma_clip(frame.get_data(), sigma=3, maxiters=5, masked=False)
    noise_mean, noise_std = frame.get_noise_stats()
    assert noise_mean == pytest.approx(np.mean(clipped))
    assert noise_std == pytest.approx(np.std(clipped))


def test_add_noise_accumulates_onto_preloaded_data():
    np.random.seed(1008)
    preload = np.full((4, 8), 2.0)
    frame = make_frame(data=preload.copy())
    noise = frame.add_noise(x_mean=5)
    np.testing.assert_allclose(frame.get_data(), preload + noise)


def test_add_noise_returns_sum_of_multiple_calls():
    np.random.seed(1009)
    frame = make_frame()
    noise_1 = frame.add_noise(x_mean=2)
    noise_2 = frame.add_noise(x_mean=3)
    np.testing.assert_allclose(frame.get_data(), noise_1 + noise_2)


def test_add_noise_gaussian_sample_moments():
    np.random.seed(1010)
    frame = make_frame(fchans=200, tchans=200)
    x_mean, x_std = 20.0, 2.0
    noise = frame.add_noise(x_mean=x_mean, x_std=x_std, noise_type='gaussian')
    assert noise.shape == frame.shape
    assert np.mean(noise) == pytest.approx(x_mean, abs=0.1)
    assert np.std(noise) == pytest.approx(x_std, rel=0.05)


def test_add_noise_normal_matches_gaussian_alias():
    np.random.seed(1011)
    frame_a = make_frame()
    noise_a = frame_a.add_noise(x_mean=1, x_std=0.5, noise_type='gaussian')

    np.random.seed(1011)
    frame_b = make_frame()
    noise_b = frame_b.add_noise(x_mean=1, x_std=0.5, noise_type='normal')

    np.testing.assert_array_equal(noise_a, noise_b)


def test_add_noise_truncated_gaussian_respects_minimum():
    np.random.seed(1012)
    frame = make_frame(fchans=50, tchans=50)
    x_min = 0.5
    frame.add_noise(x_mean=0, x_std=1, x_min=x_min, noise_type='gaussian')
    assert np.all(frame.get_data() >= x_min)


def test_add_noise_gaussian_without_x_std_exits():
    np.random.seed(1013)
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=5, noise_type='gaussian')


def test_add_noise_invalid_noise_type_exits():
    np.random.seed(1014)
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=5, noise_type='poisson')


def test_add_noise_chi2_uses_patched_distribution(monkeypatch):
    np.random.seed(1015)
    calls = []

    def fake_chi2(x_mean, df, shape):
        calls.append((x_mean, df, shape))
        return np.full(shape, 2.0)

    monkeypatch.setattr(distributions, 'chi2', fake_chi2)

    frame = make_frame()
    noise = frame.add_noise(x_mean=5)

    assert calls == [(5, 4, (4, 8))]
    np.testing.assert_array_equal(noise, np.full((4, 8), 2.0))
    np.testing.assert_array_equal(frame.get_data(), np.full((4, 8), 2.0))
    _, noise_std = frame.get_noise_stats()
    assert noise_std == pytest.approx(np.sqrt(8) * 5 / 4)


def test_get_data_returns_internal_array_by_default():
    np.random.seed(1016)
    frame = make_frame()
    assert frame.get_data() is frame.data


def test_get_data_use_db_converts_to_decibels():
    np.random.seed(1017)
    data = np.array([[1.0, 10.0], [100.0, 1000.0]])
    frame = make_frame(fchans=2, tchans=2, data=data)
    np.testing.assert_allclose(frame.get_data(use_db=True),
                               np.array([[0.0, 10.0], [20.0, 30.0]]))


def test_get_data_use_db_does_not_modify_stored_data():
    np.random.seed(1018)
    data = np.array([[1.0, 10.0], [100.0, 1000.0]])
    frame = make_frame(fchans=2, tchans=2, data=data.copy())
    frame.get_data(use_db=True)
    np.testing.assert_array_equal(frame.data, data)


def test_get_data_reflects_noise_added_after_creation():
    np.random.seed(1019)
    frame = make_frame()
    before = frame.get_data().copy()
    noise = frame.add_noise(x_mean=4)
    after = frame.get_data()
    np.testing.assert_allclose(after - before, noise)
    assert not np.array_equal(before, after)


def test_get_data_returns_preloaded_array_unchanged():
    np.random.seed(1020)
    data = np.arange(8, dtype=float).reshape(2, 4)
    frame = Frame.from_data(df=1 * u.Hz, dt=1 * u.s, fch1=8 * u.GHz, data=data)
    assert frame.get_data() is data
    np.testing.assert_array_equal(frame.get_data(), data)
