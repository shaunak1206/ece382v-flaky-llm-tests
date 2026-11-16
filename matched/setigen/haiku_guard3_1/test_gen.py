import numpy as np
import pytest
from astropy import units as u
from astropy.stats import sigma_clip

from setigen import distributions
from setigen.frame import Frame


def make_frame(tchans=4, fchans=8, data=None):
    # df * dt = 1, so chi2_df = 4 * round(1) = 4
    return Frame(fchans=fchans,
                 tchans=tchans,
                 df=1 * u.Hz,
                 dt=1 * u.s,
                 fch1=8 * u.GHz,
                 data=data)


# ---------------------------------------------------------------------------
# Frame.add_noise
# ---------------------------------------------------------------------------

def test_add_noise_chi2_returns_noise_with_frame_shape():
    np.random.seed(1001)
    frame = make_frame(tchans=5, fchans=7)
    noise = frame.add_noise(x_mean=10)
    assert noise.shape == (5, 7)
    assert frame.get_data().shape == (5, 7)


def test_add_noise_chi2_adds_noise_to_existing_data():
    np.random.seed(1002)
    base = np.ones((4, 8))
    frame = make_frame(tchans=4, fchans=8, data=base.copy())
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.get_data(), base + noise, rtol=1e-12, atol=1e-12)


def test_add_noise_chi2_values_are_nonnegative():
    # A scaled chi-squared sample is nonnegative for every random outcome.
    np.random.seed(1003)
    frame = make_frame(tchans=20, fchans=20)
    noise = frame.add_noise(x_mean=5)
    assert np.all(noise >= 0)


def test_add_noise_chi2_sets_noise_stats_on_first_call():
    np.random.seed(1004)
    frame = make_frame()
    x_mean = 10.0
    frame.add_noise(x_mean=x_mean)
    expected_std = np.sqrt(2 * frame.chi2_df) * x_mean / frame.chi2_df
    assert frame.noise_mean == pytest.approx(x_mean, rel=1e-12)
    assert frame.noise_std == pytest.approx(expected_std, rel=1e-12)


def test_add_noise_chi2_ignores_x_std_argument():
    np.random.seed(1005)
    frame = make_frame()
    x_mean = 10.0
    frame.add_noise(x_mean=x_mean, x_std=123.0)
    expected_std = np.sqrt(2 * frame.chi2_df) * x_mean / frame.chi2_df
    assert frame.noise_std == pytest.approx(expected_std, rel=1e-12)


def test_add_noise_chi2_sample_mean_close_to_x_mean():
    np.random.seed(1006)
    frame = make_frame(tchans=200, fchans=200)
    x_mean = 10.0
    noise = frame.add_noise(x_mean=x_mean)
    # Standard error is about 0.035 here, so 0.5 is a very wide margin.
    assert np.mean(noise) == pytest.approx(x_mean, abs=0.5)


def test_add_noise_gaussian_sample_moments_close_to_params():
    np.random.seed(1007)
    frame = make_frame(tchans=200, fchans=200)
    noise = frame.add_noise(x_mean=5.0, x_std=2.0, noise_type='gaussian')
    assert np.mean(noise) == pytest.approx(5.0, abs=0.1)
    assert np.std(noise) == pytest.approx(2.0, abs=0.1)


def test_add_noise_normal_and_gaussian_are_equivalent_with_same_seed():
    frame_a = make_frame()
    np.random.seed(1008)
    noise_a = frame_a.add_noise(x_mean=3.0, x_std=1.0, noise_type='normal')

    frame_b = make_frame()
    np.random.seed(1008)
    noise_b = frame_b.add_noise(x_mean=3.0, x_std=1.0, noise_type='gaussian')

    np.testing.assert_allclose(noise_a, noise_b, rtol=0, atol=0)


def test_add_noise_normal_sets_noise_stats_from_params():
    np.random.seed(1009)
    frame = make_frame()
    frame.add_noise(x_mean=4.0, x_std=1.5, noise_type='normal')
    assert frame.noise_mean == pytest.approx(4.0, rel=1e-12)
    assert frame.noise_std == pytest.approx(1.5, rel=1e-12)


def test_add_noise_truncated_gaussian_respects_x_min():
    # Every sample must be at or above x_min, for any random outcome.
    np.random.seed(1010)
    frame = make_frame(tchans=30, fchans=30)
    x_min = 0.0
    noise = frame.add_noise(x_mean=1.0, x_std=2.0, x_min=x_min, noise_type='normal')
    assert np.all(noise >= x_min - 1e-12)
    assert np.all(frame.get_data() >= x_min - 1e-12)


def test_add_noise_normal_without_x_std_exits():
    np.random.seed(1011)
    frame = make_frame()
    with pytest.raises(SystemExit) as excinfo:
        frame.add_noise(x_mean=1.0, noise_type='normal')
    assert 'x_std must be given' in str(excinfo.value.code)


def test_add_noise_invalid_noise_type_exits():
    np.random.seed(1012)
    frame = make_frame()
    with pytest.raises(SystemExit) as excinfo:
        frame.add_noise(x_mean=1.0, noise_type='poisson')
    assert 'not a valid noise type' in str(excinfo.value.code)


def test_add_noise_second_call_recomputes_stats_from_data():
    np.random.seed(1013)
    frame = make_frame(tchans=10, fchans=10)
    frame.add_noise(x_mean=10.0)
    frame.add_noise(x_mean=10.0)
    clipped = sigma_clip(frame.get_data(), sigma=3, maxiters=5, masked=False)
    assert frame.noise_mean == pytest.approx(np.mean(clipped), rel=1e-12)
    assert frame.noise_std == pytest.approx(np.std(clipped), rel=1e-12)


def test_add_noise_accumulates_across_calls():
    np.random.seed(1014)
    frame = make_frame()
    noise1 = frame.add_noise(x_mean=5.0)
    noise2 = frame.add_noise(x_mean=5.0)
    np.testing.assert_allclose(frame.get_data(), noise1 + noise2,
                               rtol=1e-12, atol=1e-12)


def test_add_noise_uses_patched_chi2_distribution(monkeypatch):
    np.random.seed(1015)

    calls = []

    def stub_chi2(x_mean, df, shape):
        calls.append((x_mean, df, shape))
        return np.full(shape, 2.0)

    monkeypatch.setattr(distributions, 'chi2', stub_chi2)

    frame = make_frame(tchans=3, fchans=5)
    noise = frame.add_noise(x_mean=7.0)

    assert calls == [(7.0, frame.chi2_df, (3, 5))]
    np.testing.assert_array_equal(noise, np.full((3, 5), 2.0))
    np.testing.assert_array_equal(frame.get_data(), np.full((3, 5), 2.0))


# ---------------------------------------------------------------------------
# Frame.get_data
# ---------------------------------------------------------------------------

def test_get_data_default_returns_underlying_array():
    frame = make_frame()
    assert frame.get_data() is frame.data


def test_get_data_use_db_false_returns_underlying_array():
    frame = make_frame()
    assert frame.get_data(use_db=False) is frame.data


def test_get_data_use_db_converts_to_decibels():
    data = np.array([[1.0, 10.0], [100.0, 1000.0]])
    frame = make_frame(tchans=2, fchans=2, data=data)
    expected = np.array([[0.0, 10.0], [20.0, 30.0]])
    np.testing.assert_allclose(frame.get_data(use_db=True), expected,
                               rtol=1e-12, atol=1e-12)


def test_get_data_use_db_positional_argument():
    data = np.array([[1.0, 10.0], [100.0, 1000.0]])
    frame = make_frame(tchans=2, fchans=2, data=data)
    np.testing.assert_allclose(frame.get_data(True),
                               10 * np.log10(data), rtol=1e-12, atol=1e-12)


def test_get_data_use_db_does_not_modify_stored_data():
    data = np.array([[1.0, 10.0], [100.0, 1000.0]])
    frame = make_frame(tchans=2, fchans=2, data=data.copy())
    frame.get_data(use_db=True)
    np.testing.assert_array_equal(frame.data, data)


def test_get_data_use_db_zero_values_give_negative_infinity():
    frame = make_frame(tchans=2, fchans=2)  # all zeros
    with np.errstate(divide='ignore'):
        out = frame.get_data(use_db=True)
    assert np.all(np.isneginf(out))


def test_get_data_preserves_time_by_frequency_orientation():
    data = np.arange(12, dtype=float).reshape(3, 4)
    frame = make_frame(tchans=3, fchans=4, data=data)
    out = frame.get_data()
    assert out.shape == (3, 4)
    assert out[1, 2] == 6.0
    np.testing.assert_array_equal(out, data)


def test_get_data_reflects_added_noise():
    np.random.seed(1016)
    frame = make_frame()
    noise = frame.add_noise(x_mean=3.0)
    np.testing.assert_allclose(frame.get_data(), noise, rtol=1e-12, atol=1e-12)


def test_get_data_use_db_matches_log_of_noisy_data():
    np.random.seed(1017)
    frame = make_frame(tchans=6, fchans=6)
    frame.add_noise(x_mean=8.0)
    data = frame.get_data()
    with np.errstate(divide='ignore'):
        expected = 10 * np.log10(data)
        out = frame.get_data(use_db=True)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)
