import numpy as np
import pytest
from astropy import units as u
from astropy.stats import sigma_clip

from setigen.frame import Frame


def make_frame(tchans=8, fchans=16, df=1.0, dt=1.0, data=None):
    """Create a Frame with simple resolutions (chi2_df = 4 * round(df * dt))."""
    return Frame(fchans=fchans,
                 tchans=tchans,
                 df=df * u.Hz,
                 dt=dt * u.s,
                 fch1=8 * u.GHz,
                 data=data)


# ---------------------------------------------------------------------------
# Frame.add_noise
# ---------------------------------------------------------------------------

def test_add_noise_chi2_returns_frame_shape():
    np.random.seed(0)
    frame = make_frame(tchans=8, fchans=16)
    noise = frame.add_noise(x_mean=10)
    assert noise.shape == frame.shape == (8, 16)


def test_add_noise_chi2_adds_returned_noise_to_zero_data():
    np.random.seed(1)
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_array_equal(frame.get_data(), noise)


def test_add_noise_preserves_existing_data():
    np.random.seed(2)
    initial = np.full((8, 16), 3.0)
    frame = make_frame(data=initial.copy())
    noise = frame.add_noise(x_mean=5)
    np.testing.assert_allclose(frame.get_data(), initial + noise)


def test_add_noise_chi2_mean_matches_x_mean():
    np.random.seed(3)
    frame = make_frame(tchans=100, fchans=200)
    noise = frame.add_noise(x_mean=10)
    assert np.mean(noise) == pytest.approx(10, abs=0.5)


def test_add_noise_chi2_is_nonnegative():
    np.random.seed(4)
    frame = make_frame(tchans=50, fchans=50)
    noise = frame.add_noise(x_mean=7)
    assert np.all(noise >= 0)


def test_add_noise_chi2_ignores_x_std_and_x_min():
    frame_a = make_frame()
    frame_b = make_frame()
    np.random.seed(5)
    noise_a = frame_a.add_noise(x_mean=10)
    np.random.seed(5)
    noise_b = frame_b.add_noise(x_mean=10, x_std=1000, x_min=500)
    np.testing.assert_array_equal(noise_a, noise_b)


def test_add_noise_chi2_sets_noise_stats_from_params_on_zero_frame():
    np.random.seed(6)
    frame = make_frame(df=1.0, dt=1.0)  # chi2_df = 4
    x_mean = 10
    frame.add_noise(x_mean=x_mean)
    expected_std = np.sqrt(2 * frame.chi2_df) * x_mean / frame.chi2_df
    noise_mean, noise_std = frame.get_noise_stats()
    assert noise_mean == pytest.approx(x_mean)
    assert noise_std == pytest.approx(expected_std)


def test_add_noise_invalid_noise_type_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1, noise_type='poisson')


def test_add_noise_normal_without_x_std_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1, noise_type='normal')


def test_add_noise_normal_matches_requested_mean_and_std():
    np.random.seed(7)
    frame = make_frame(tchans=100, fchans=200)
    noise = frame.add_noise(x_mean=3, x_std=2, noise_type='normal')
    assert np.mean(noise) == pytest.approx(3, abs=0.1)
    assert np.std(noise) == pytest.approx(2, rel=0.05)


def test_add_noise_gaussian_alias_matches_normal():
    frame_a = make_frame()
    frame_b = make_frame()
    np.random.seed(8)
    noise_a = frame_a.add_noise(x_mean=0, x_std=1, noise_type='normal')
    np.random.seed(8)
    noise_b = frame_b.add_noise(x_mean=0, x_std=1, noise_type='gaussian')
    np.testing.assert_array_equal(noise_a, noise_b)


def test_add_noise_truncated_gaussian_respects_x_min():
    np.random.seed(9)
    frame = make_frame(tchans=50, fchans=50)
    noise = frame.add_noise(x_mean=0, x_std=1, x_min=0.5, noise_type='normal')
    assert np.min(noise) >= 0.5


def test_add_noise_gaussian_without_x_min_can_be_negative():
    np.random.seed(10)
    frame = make_frame(tchans=100, fchans=200)
    noise = frame.add_noise(x_mean=0, x_std=1, noise_type='normal')
    assert np.min(noise) < 0


def test_add_noise_normal_sets_noise_stats_from_params_on_zero_frame():
    np.random.seed(11)
    frame = make_frame()
    frame.add_noise(x_mean=5, x_std=2, noise_type='normal')
    assert frame.get_noise_stats() == (5, 2)


def test_add_noise_second_call_updates_stats_from_sigma_clipped_data():
    np.random.seed(12)
    frame = make_frame(tchans=50, fchans=50)
    frame.add_noise(x_mean=10)
    frame.add_noise(x_mean=10)
    clipped = sigma_clip(frame.get_data(), sigma=3, maxiters=5, masked=False)
    noise_mean, noise_std = frame.get_noise_stats()
    np.testing.assert_allclose(noise_mean, np.mean(clipped))
    np.testing.assert_allclose(noise_std, np.std(clipped))


def test_add_noise_accumulates_across_calls():
    np.random.seed(13)
    frame = make_frame()
    noise1 = frame.add_noise(x_mean=4)
    noise2 = frame.add_noise(x_mean=4)
    np.testing.assert_allclose(frame.get_data(), noise1 + noise2)


# ---------------------------------------------------------------------------
# Frame.get_data
# ---------------------------------------------------------------------------

def test_get_data_default_returns_linear_data():
    data = np.arange(1, 13, dtype=float).reshape(3, 4)
    frame = Frame.from_data(df=1 * u.Hz, dt=1 * u.s, fch1=8 * u.GHz, data=data)
    np.testing.assert_array_equal(frame.get_data(), data)


def test_get_data_returns_same_underlying_array():
    frame = make_frame()
    assert frame.get_data() is frame.data


def test_get_data_default_matches_use_db_false():
    np.random.seed(20)
    frame = make_frame()
    frame.add_noise(x_mean=5)
    np.testing.assert_array_equal(frame.get_data(),
                                  frame.get_data(use_db=False))


def test_get_data_shape_matches_frame_shape():
    frame = make_frame(tchans=5, fchans=7)
    assert frame.get_data().shape == (5, 7)
    assert frame.get_data(use_db=True).shape == (5, 7)


def test_get_data_use_db_matches_10_log10():
    np.random.seed(21)
    frame = make_frame()
    frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.get_data(use_db=True),
                               10 * np.log10(frame.data))


def test_get_data_use_db_known_values():
    data = np.array([[1.0, 10.0],
                     [100.0, 1000.0]])
    frame = Frame.from_data(df=1 * u.Hz, dt=1 * u.s, fch1=8 * u.GHz, data=data)
    np.testing.assert_allclose(frame.get_data(use_db=True),
                               np.array([[0.0, 10.0],
                                         [20.0, 30.0]]))


def test_get_data_use_db_zero_data_gives_negative_infinity():
    frame = make_frame()
    with np.errstate(divide='ignore'):
        out = frame.get_data(use_db=True)
    assert np.all(np.isneginf(out))


def test_get_data_does_not_modify_frame_data():
    data = np.full((4, 4), 2.0)
    frame = make_frame(tchans=4, fchans=4, data=data.copy())
    frame.get_data(use_db=True)
    np.testing.assert_array_equal(frame.data, data)


def test_get_data_reflects_add_noise_changes():
    frame = make_frame()
    assert np.all(frame.get_data() == 0)
    noise = frame.add_noise(x_mean=6)
    np.testing.assert_array_equal(frame.get_data(), noise)
