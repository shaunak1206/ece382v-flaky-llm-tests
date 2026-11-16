import numpy as np
import pytest
from astropy import units as u

from setigen.frame import Frame


def make_frame(fchans=16, tchans=8, df=1.0 * u.Hz, dt=1.0 * u.s,
               fch1=8 * u.GHz, data=None):
    """Create a Frame with explicit dimensions and resolution."""
    return Frame(fchans=fchans,
                 tchans=tchans,
                 df=df,
                 dt=dt,
                 fch1=fch1,
                 data=data)


@pytest.fixture(autouse=True)
def seed_rng():
    """Make every test deterministic."""
    np.random.seed(12345)
    yield


# ---------------------------------------------------------------------------
# Frame.add_noise: chi-squared (default) noise
# ---------------------------------------------------------------------------

def test_add_noise_chi2_returns_noise_with_frame_shape():
    frame = make_frame(fchans=16, tchans=8)
    noise = frame.add_noise(x_mean=10)
    assert noise.shape == frame.shape == (8, 16)


def test_add_noise_chi2_adds_returned_noise_to_zero_frame():
    frame = make_frame(fchans=16, tchans=8)
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_array_equal(frame.get_data(), noise)


def test_add_noise_chi2_accumulates_onto_existing_data():
    data = np.ones((8, 16))
    frame = make_frame(fchans=16, tchans=8, data=data.copy())
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.get_data() - noise, np.ones((8, 16)))


def test_add_noise_chi2_mean_approximates_x_mean():
    frame = make_frame(fchans=256, tchans=64)
    frame.add_noise(x_mean=10)
    assert np.mean(frame.get_data()) == pytest.approx(10, abs=0.3)


def test_add_noise_chi2_ignores_x_std():
    frame_a = make_frame(fchans=32, tchans=16)
    frame_b = make_frame(fchans=32, tchans=16)

    np.random.seed(999)
    noise_a = frame_a.add_noise(x_mean=3)

    np.random.seed(999)
    noise_b = frame_b.add_noise(x_mean=3, x_std=1000)

    np.testing.assert_array_equal(noise_a, noise_b)


def test_add_noise_chi2_sets_noise_stats_on_first_call():
    frame = make_frame(fchans=16, tchans=8)  # chi2_df = 4
    x_mean = 10
    frame.add_noise(x_mean=x_mean)

    expected_std = np.sqrt(2 * 4) * x_mean / 4
    noise_mean, noise_std = frame.get_noise_stats()
    assert noise_mean == pytest.approx(x_mean)
    assert noise_std == pytest.approx(expected_std)


def test_add_noise_chi2_std_uses_chi2_df_from_resolution():
    # df * dt = 2 -> chi2_df = 8
    frame = make_frame(fchans=16, tchans=8, df=2.0 * u.Hz, dt=1.0 * u.s)
    assert frame.chi2_df == 8

    x_mean = 6
    frame.add_noise(x_mean=x_mean)

    expected_std = np.sqrt(2 * 8) * x_mean / 8
    _, noise_std = frame.get_noise_stats()
    assert noise_std == pytest.approx(expected_std)


def test_add_noise_second_call_recomputes_noise_stats_from_data():
    frame = make_frame(fchans=256, tchans=64)
    x_mean = 5
    frame.add_noise(x_mean=x_mean)
    frame.add_noise(x_mean=x_mean)

    # Two independent chi-squared noise contributions with mean 5 each
    noise_mean, _ = frame.get_noise_stats()
    assert noise_mean == pytest.approx(2 * x_mean, rel=0.05)


# ---------------------------------------------------------------------------
# Frame.add_noise: Gaussian noise
# ---------------------------------------------------------------------------

def test_add_noise_normal_has_requested_moments():
    frame = make_frame(fchans=256, tchans=64)
    frame.add_noise(x_mean=5, x_std=2, noise_type='normal')

    data = frame.get_data()
    assert np.mean(data) == pytest.approx(5, abs=0.1)
    assert np.std(data) == pytest.approx(2, rel=0.05)


def test_add_noise_gaussian_alias_matches_normal():
    frame_a = make_frame(fchans=32, tchans=16)
    frame_b = make_frame(fchans=32, tchans=16)

    np.random.seed(2024)
    noise_a = frame_a.add_noise(x_mean=4, x_std=1, noise_type='normal')

    np.random.seed(2024)
    noise_b = frame_b.add_noise(x_mean=4, x_std=1, noise_type='gaussian')

    np.testing.assert_array_equal(noise_a, noise_b)


def test_add_noise_normal_without_x_std_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=5, noise_type='normal')


def test_add_noise_normal_sets_noise_stats_to_parameters_on_first_call():
    frame = make_frame(fchans=16, tchans=8)
    frame.add_noise(x_mean=5, x_std=2, noise_type='normal')

    noise_mean, noise_std = frame.get_noise_stats()
    assert noise_mean == pytest.approx(5)
    assert noise_std == pytest.approx(2)


def test_add_noise_truncated_gaussian_respects_x_min():
    frame = make_frame(fchans=64, tchans=32)
    noise = frame.add_noise(x_mean=1, x_std=2, x_min=0, noise_type='normal')

    assert np.all(noise >= 0)
    assert np.all(frame.get_data() >= 0)


def test_add_noise_gaussian_without_x_min_can_go_negative():
    frame = make_frame(fchans=64, tchans=32)
    noise = frame.add_noise(x_mean=0, x_std=1, noise_type='normal')

    assert np.min(noise) < 0


# ---------------------------------------------------------------------------
# Frame.add_noise: invalid input
# ---------------------------------------------------------------------------

def test_add_noise_invalid_noise_type_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=5, x_std=1, noise_type='poisson')


# ---------------------------------------------------------------------------
# Frame.get_data
# ---------------------------------------------------------------------------

def test_get_data_returns_underlying_array_by_default():
    frame = make_frame()
    data = frame.get_data()
    assert data is frame.data


def test_get_data_shape_matches_frame_shape():
    frame = make_frame(fchans=24, tchans=12)
    assert frame.get_data().shape == (12, 24)


def test_get_data_use_db_returns_10_log10_of_data():
    data = np.array([[1.0, 10.0],
                     [100.0, 1000.0]])
    frame = Frame.from_data(df=1.0 * u.Hz,
                            dt=1.0 * u.s,
                            fch1=8 * u.GHz,
                            data=data.copy())
    result = frame.get_data(use_db=True)
    np.testing.assert_allclose(result, [[0.0, 10.0],
                                        [20.0, 30.0]])


def test_get_data_use_db_zero_values_become_negative_infinity():
    frame = make_frame(fchans=4, tchans=2)
    with np.errstate(divide='ignore'):
        result = frame.get_data(use_db=True)
    assert np.all(np.isneginf(result))


def test_get_data_use_db_does_not_modify_underlying_data():
    data = np.array([[1.0, 10.0], [100.0, 1000.0]])
    frame = Frame.from_data(df=1.0 * u.Hz,
                            dt=1.0 * u.s,
                            fch1=8 * u.GHz,
                            data=data.copy())
    before = frame.data.copy()
    frame.get_data(use_db=True)
    np.testing.assert_array_equal(frame.data, before)


def test_get_data_reflects_noise_added_by_add_noise():
    frame = make_frame(fchans=16, tchans=8)
    noise = frame.add_noise(x_mean=7)
    np.testing.assert_array_equal(frame.get_data(), noise)
    np.testing.assert_array_equal(frame.get_data(use_db=False), frame.data)
