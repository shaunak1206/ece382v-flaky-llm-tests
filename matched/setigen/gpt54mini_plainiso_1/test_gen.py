import numpy as np
import pytest
from astropy import units as u

import setigen.frame as frame_module
from setigen.frame import Frame


@pytest.fixture
def simple_frame():
    return Frame(fchans=4, tchans=3, df=2 * u.Hz, dt=1 * u.s, fch1=10 * u.Hz)


def test_get_data_returns_internal_array(simple_frame):
    data = simple_frame.get_data()
    assert data is simple_frame.data
    assert np.all(data == 0)


def test_get_data_use_db_converts_to_decibels(simple_frame):
    simple_frame.data = np.array([[1.0, 10.0], [100.0, 0.1]])
    expected = 10 * np.log10(simple_frame.data)
    out = simple_frame.get_data(use_db=True)
    assert np.allclose(out, expected)


def test_get_data_use_db_with_zeros_produces_negative_infinity(simple_frame):
    simple_frame.data = np.zeros((3, 4))
    out = simple_frame.get_data(use_db=True)
    assert out.shape == (3, 4)
    assert np.isneginf(out).all()


def test_add_noise_chi2_calls_distribution_and_updates_data_and_stats(simple_frame, monkeypatch):
    calls = {}

    def fake_chi2(x_mean, chi2_df, shape):
        calls["args"] = (x_mean, chi2_df, shape)
        return np.full(shape, 2.5)

    monkeypatch.setattr(frame_module.distributions, "chi2", fake_chi2)

    noise = simple_frame.add_noise(x_mean=5.0, noise_type="chi2")

    assert calls["args"] == (5.0, simple_frame.chi2_df, simple_frame.shape)
    assert np.all(noise == 2.5)
    assert np.all(simple_frame.data == 2.5)
    assert simple_frame.noise_mean == 5.0
    assert simple_frame.noise_std == pytest.approx(np.sqrt(2 * simple_frame.chi2_df) * 5.0 / simple_frame.chi2_df)


def test_add_noise_normal_calls_gaussian_and_updates_data_and_stats(simple_frame, monkeypatch):
    calls = {}

    def fake_gaussian(x_mean, x_std, shape):
        calls["args"] = (x_mean, x_std, shape)
        return np.full(shape, 7.0)

    monkeypatch.setattr(frame_module.distributions, "gaussian", fake_gaussian)

    noise = simple_frame.add_noise(x_mean=1.5, x_std=0.25, noise_type="normal")

    assert calls["args"] == (1.5, 0.25, simple_frame.shape)
    assert np.all(noise == 7.0)
    assert np.all(simple_frame.data == 7.0)
    assert simple_frame.noise_mean == 1.5
    assert simple_frame.noise_std == 0.25


def test_add_noise_gaussian_alias_uses_gaussian_distribution(simple_frame, monkeypatch):
    called = {}

    def fake_gaussian(x_mean, x_std, shape):
        called["args"] = (x_mean, x_std, shape)
        return np.ones(shape)

    monkeypatch.setattr(frame_module.distributions, "gaussian", fake_gaussian)

    noise = simple_frame.add_noise(x_mean=3.0, x_std=1.0, noise_type="gaussian")

    assert called["args"] == (3.0, 1.0, simple_frame.shape)
    assert np.array_equal(noise, np.ones(simple_frame.shape))


def test_add_noise_normal_with_x_min_uses_truncated_gaussian(simple_frame, monkeypatch):
    called = {}

    def fake_truncated_gaussian(x_mean, x_std, x_min, shape):
        called["args"] = (x_mean, x_std, x_min, shape)
        return np.full(shape, 4.0)

    monkeypatch.setattr(frame_module.distributions, "truncated_gaussian", fake_truncated_gaussian)

    noise = simple_frame.add_noise(x_mean=2.0, x_std=0.5, x_min=1.0, noise_type="normal")

    assert called["args"] == (2.0, 0.5, 1.0, simple_frame.shape)
    assert np.all(noise == 4.0)
    assert np.all(simple_frame.data == 4.0)


def test_add_noise_invalid_noise_type_exits(simple_frame):
    with pytest.raises(SystemExit):
        simple_frame.add_noise(x_mean=1.0, x_std=1.0, noise_type="invalid")


def test_add_noise_normal_without_x_std_exits(simple_frame):
    with pytest.raises(SystemExit):
        simple_frame.add_noise(x_mean=1.0, x_std=None, noise_type="normal")


def test_add_noise_updates_noise_stats_when_existing_stats_are_nonzero(simple_frame, monkeypatch):
    simple_frame.noise_mean = 2.0
    simple_frame.noise_std = 3.0

    def fake_gaussian(x_mean, x_std, shape):
        return np.full(shape, 1.0)

    monkeypatch.setattr(frame_module.distributions, "gaussian", fake_gaussian)
    monkeypatch.setattr(simple_frame, "_update_noise_frame_stats", lambda: setattr(simple_frame, "_updated", True))

    noise = simple_frame.add_noise(x_mean=10.0, x_std=1.0, noise_type="normal")

    assert np.all(noise == 1.0)
    assert simple_frame.data.sum() == pytest.approx(np.prod(simple_frame.shape))
    assert getattr(simple_frame, "_updated", False) is True
    assert simple_frame.noise_mean == 2.0
    assert simple_frame.noise_std == 3.0


def test_add_noise_from_obs_uses_saved_parameters_when_none_provided(simple_frame, monkeypatch):
    loaded = np.array([[10.0, 1.0, 0.1], [20.0, 2.0, 0.2]], dtype=float)
    monkeypatch.setattr(frame_module.np, "load", lambda path: loaded)

    def fake_gaussian(x_mean, x_std, shape):
        return np.full(shape, x_mean + x_std)

    monkeypatch.setattr(frame_module.distributions, "gaussian", fake_gaussian)
    monkeypatch.setattr(frame_module.np.random, "randint", lambda n: 1)

    noise = simple_frame.add_noise_from_obs(noise_type="normal", share_index=True)

    scale_factor = simple_frame.dt / 1.4316557653333333
    expected_mean = loaded[1, 0] * scale_factor
    expected_std = loaded[1, 1] * scale_factor
    assert np.all(noise == expected_mean + expected_std)
    assert simple_frame.noise_mean == expected_mean
    assert simple_frame.noise_std == expected_std


def test_add_noise_from_obs_share_index_length_mismatch_raises(simple_frame):
    with pytest.raises(IndexError):
        simple_frame.add_noise_from_obs(
            x_mean_array=np.array([1.0, 2.0]),
            x_std_array=np.array([0.1]),
            x_min_array=np.array([0.0, 0.0]),
            noise_type="normal",
            share_index=True,
        )


def test_add_noise_from_obs_gaussian_without_min_uses_shared_index(simple_frame, monkeypatch):
    monkeypatch.setattr(frame_module.np.random, "randint", lambda n: 0)

    def fake_gaussian(x_mean, x_std, shape):
        return np.full(shape, x_mean - x_std)

    monkeypatch.setattr(frame_module.distributions, "gaussian", fake_gaussian)

    noise = simple_frame.add_noise_from_obs(
        x_mean_array=np.array([3.0, 5.0]),
        x_std_array=np.array([0.5, 0.7]),
        noise_type="normal",
        share_index=True,
    )

    assert np.all(noise == 2.5)
    assert simple_frame.noise_mean == 3.0
    assert simple_frame.noise_std == 0.5


def test_add_noise_from_obs_chi2_uses_mean_array_only(simple_frame, monkeypatch):
    monkeypatch.setattr(frame_module.np.random, "choice", lambda arr: arr[-1])

    def fake_chi2(x_mean, chi2_df, shape):
        return np.full(shape, x_mean)

    monkeypatch.setattr(frame_module.distributions, "chi2", fake_chi2)

    noise = simple_frame.add_noise_from_obs(
        x_mean_array=np.array([1.0, 2.0, 3.0]),
        x_std_array=np.array([10.0, 20.0, 30.0]),
        x_min_array=np.array([0.0, 0.0, 0.0]),
        noise_type="chi2",
    )

    assert np.all(noise == 3.0)
    assert simple_frame.noise_mean == 3.0
    assert simple_frame.noise_std == pytest.approx(np.sqrt(2 * simple_frame.chi2_df) * 3.0 / simple_frame.chi2_df)
