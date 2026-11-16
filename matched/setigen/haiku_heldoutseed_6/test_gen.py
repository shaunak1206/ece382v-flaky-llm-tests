import numpy as np
import pytest
from astropy import units as u

from setigen.frame import Frame


@pytest.fixture(autouse=True)
def _seed_rng():
    """Make every test deterministic."""
    np.random.seed(12345)
    yield


def make_frame(fchans=256, tchans=128, df=1.0, dt=1.0, data=None):
    """Build a Frame with df/dt given in Hz and s."""
    kwargs = dict(fchans=fchans,
                  tchans=tchans,
                  df=df * u.Hz,
                  dt=dt * u.s,
                  fch1=8 * u.GHz)
    if data is not None:
        kwargs['data'] = data
    return Frame(**kwargs)


# ---------------------------------------------------------------------------
# Frame.add_noise
# ---------------------------------------------------------------------------

def test_add_noise_chi2_returns_array_matching_shape():
    frame = make_frame(fchans=64, tchans=32)
    noise = frame.add_noise(x_mean=10)
    assert isinstance(noise, np.ndarray)
    assert noise.shape == (32, 64)
    assert frame.data.shape == (32, 64)


def test_add_noise_chi2_adds_returned_noise_to_frame_data():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_array_equal(frame.get_data(), noise)


def test_add_noise_chi2_empirical_mean_matches_x_mean():
    frame = make_frame(fchans=256, tchans=256)
    noise = frame.add_noise(x_mean=10)
    assert np.mean(noise) == pytest.approx(10, rel=0.02)


def test_add_noise_chi2_std_matches_theoretical_value():
    # df=1 Hz, dt=1 s -> chi2_df = 4 -> std = sqrt(8) * x_mean / 4
    frame = make_frame(fchans=256, tchans=256, df=1.0, dt=1.0)
    x_mean = 10
    noise = frame.add_noise(x_mean=x_mean)
    expected_std = np.sqrt(2 * 4) * x_mean / 4
    assert np.std(noise) == pytest.approx(expected_std, rel=0.02)


def test_add_noise_chi2_uses_resolution_dependent_dof():
    # df=2 Hz, dt=1 s -> chi2_df = 8 -> std = sqrt(16) * x_mean / 8 = x_mean / 2
    frame = make_frame(fchans=256, tchans=256, df=2.0, dt=1.0)
    x_mean = 10
    noise = frame.add_noise(x_mean=x_mean)
    assert frame.chi2_df == 8
    assert np.std(noise) == pytest.approx(x_mean / 2, rel=0.02)


def test_add_noise_chi2_ignores_x_std_argument():
    frame_a = make_frame()
    frame_b = make_frame()

    np.random.seed(7)
    noise_a = frame_a.add_noise(x_mean=5)

    np.random.seed(7)
    noise_b = frame_b.add_noise(x_mean=5, x_std=1000, x_min=-50)

    np.testing.assert_array_equal(noise_a, noise_b)


def test_add_noise_chi2_is_reproducible_with_seed():
    frame_a = make_frame()
    frame_b = make_frame()

    np.random.seed(99)
    noise_a = frame_a.add_noise(x_mean=3)

    np.random.seed(99)
    noise_b = frame_b.add_noise(x_mean=3)

    np.testing.assert_array_equal(noise_a, noise_b)


def test_add_noise_normal_empirical_mean_and_std():
    frame = make_frame(fchans=256, tchans=256)
    noise = frame.add_noise(x_mean=20, x_std=4, noise_type='normal')
    assert np.mean(noise) == pytest.approx(20, rel=0.02)
    assert np.std(noise) == pytest.approx(4, rel=0.02)


def test_add_noise_gaussian_alias_matches_normal():
    frame_a = make_frame()
    frame_b = make_frame()

    np.random.seed(2024)
    noise_a = frame_a.add_noise(x_mean=0, x_std=1, noise_type='normal')

    np.random.seed(2024)
    noise_b = frame_b.add_noise(x_mean=0, x_std=1, noise_type='gaussian')

    np.testing.assert_array_equal(noise_a, noise_b)


def test_add_noise_normal_without_x_std_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=5, noise_type='normal')


def test_add_noise_invalid_noise_type_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=5, x_std=1, noise_type='poisson')


def test_add_noise_normal_with_x_min_respects_lower_bound():
    frame = make_frame(fchans=128, tchans=128)
    noise = frame.add_noise(x_mean=0, x_std=5, x_min=-1, noise_type='normal')
    assert np.all(noise >= -1)


def test_add_noise_accumulates_onto_existing_data():
    original = np.full((32, 64), 7.0)
    frame = make_frame(fchans=64, tchans=32, data=original.copy())
    noise = frame.add_noise(x_mean=2, x_std=1, noise_type='normal')
    np.testing.assert_allclose(frame.get_data(), original + noise)


def test_add_noise_successive_calls_sum_into_data():
    frame = make_frame()
    noise1 = frame.add_noise(x_mean=4)
    noise2 = frame.add_noise(x_mean=6)
    np.testing.assert_allclose(frame.get_data(), noise1 + noise2)


def test_add_noise_first_call_sets_chi2_noise_stats_from_parameters():
    frame = make_frame(fchans=64, tchans=32, df=1.0, dt=1.0)
    x_mean = 10
    frame.add_noise(x_mean=x_mean)
    noise_mean, noise_std = frame.get_noise_stats()
    expected_std = np.sqrt(2 * 4) * x_mean / 4
    assert noise_mean == pytest.approx(x_mean)
    assert noise_std == pytest.approx(expected_std)


def test_add_noise_first_call_sets_normal_noise_stats_from_parameters():
    frame = make_frame()
    frame.add_noise(x_mean=12, x_std=3, noise_type='normal')
    noise_mean, noise_std = frame.get_noise_stats()
    assert noise_mean == pytest.approx(12)
    assert noise_std == pytest.approx(3)


def test_add_noise_second_call_recomputes_noise_stats_from_data():
    frame = make_frame(fchans=256, tchans=256)
    frame.add_noise(x_mean=10)
    frame.add_noise(x_mean=20)
    noise_mean, _ = frame.get_noise_stats()
    # Combined data has mean ~30 after the two chi-squared additions
    assert noise_mean == pytest.approx(30, rel=0.02)


# ---------------------------------------------------------------------------
# Frame.get_data
# ---------------------------------------------------------------------------

def test_get_data_default_returns_frame_data_object():
    frame = make_frame()
    assert frame.get_data() is frame.data


def test_get_data_use_db_converts_to_decibels():
    data = np.array([[1.0, 10.0],
                     [100.0, 1000.0]])
    frame = make_frame(fchans=2, tchans=2, data=data)
    expected = np.array([[0.0, 10.0],
                         [20.0, 30.0]])
    np.testing.assert_allclose(frame.get_data(use_db=True), expected)


def test_get_data_use_db_does_not_modify_frame_data():
    data = np.array([[1.0, 10.0],
                     [100.0, 1000.0]])
    frame = make_frame(fchans=2, tchans=2, data=data.copy())
    _ = frame.get_data(use_db=True)
    np.testing.assert_array_equal(frame.data, data)


def test_get_data_returns_live_reference_to_frame_data():
    frame = make_frame(fchans=2, tchans=2, data=np.zeros((2, 2)))
    returned = frame.get_data()
    returned[0, 0] = 42.0
    assert frame.data[0, 0] == 42.0


def test_get_data_zero_values_give_negative_infinity_in_db():
    frame = make_frame(fchans=2, tchans=2, data=np.zeros((2, 2)))
    with np.errstate(divide='ignore'):
        result = frame.get_data(use_db=True)
    assert np.all(np.isneginf(result))


def test_get_data_after_add_noise_matches_returned_noise():
    frame = make_frame(fchans=64, tchans=32, data=np.zeros((32, 64)))
    noise = frame.add_noise(x_mean=8)
    np.testing.assert_allclose(frame.get_data(), noise)
    with np.errstate(divide='ignore'):
        np.testing.assert_allclose(frame.get_data(use_db=True),
                                   10 * np.log10(noise))


def test_get_data_from_data_constructor_roundtrip():
    data = np.arange(12, dtype=float).reshape(3, 4) + 1.0
    frame = Frame.from_data(df=1 * u.Hz,
                            dt=1 * u.s,
                            fch1=8 * u.GHz,
                            data=data)
    np.testing.assert_array_equal(frame.get_data(), data)
