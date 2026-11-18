import numpy as np
import pytest
from astropy import units as u

from setigen.frame import Frame


def make_frame():
    return Frame(fchans=4, tchans=3, df=1 * u.Hz, dt=1 * u.s, fch1=10 * u.Hz)


def test_get_data_returns_underlying_array():
    frame = make_frame()
    frame.data[:] = np.arange(frame.data.size).reshape(frame.shape)
    out = frame.get_data()
    assert out is frame.data
    assert np.array_equal(out, frame.data)


def test_get_data_db_converts_to_decibels():
    frame = make_frame()
    frame.data[:] = 10.0
    out = frame.get_data(use_db=True)
    expected = 10 * np.log10(frame.data)
    assert np.allclose(out, expected)


def test_get_data_db_on_ones_is_zero():
    frame = make_frame()
    frame.data[:] = 1.0
    out = frame.get_data(use_db=True)
    assert np.allclose(out, 0.0)


def test_add_noise_chi2_uses_distribution_and_updates_data(monkeypatch):
    frame = make_frame()

    called = {}

    def fake_chi2(x_mean, df, shape):
        called["args"] = (x_mean, df, shape)
        return np.full(shape, 2.5)

    monkeypatch.setattr("setigen.frame.distributions.chi2", fake_chi2)

    noise = frame.add_noise(x_mean=3.0, noise_type="chi2")

    assert np.array_equal(noise, np.full(frame.shape, 2.5))
    assert np.array_equal(frame.data, np.full(frame.shape, 2.5))
    assert called["args"] == (3.0, frame.chi2_df, frame.shape)
    assert frame.noise_mean == 3.0
    assert frame.noise_std == pytest.approx(
        np.sqrt(2 * frame.chi2_df) * 3.0 / frame.chi2_df
    )


def test_add_noise_normal_uses_gaussian_when_no_min(monkeypatch):
    frame = make_frame()

    def fake_gaussian(x_mean, x_std, shape):
        assert (x_mean, x_std, shape) == (1.5, 0.25, frame.shape)
        return np.ones(shape)

    monkeypatch.setattr("setigen.frame.distributions.gaussian", fake_gaussian)

    noise = frame.add_noise(x_mean=1.5, x_std=0.25, noise_type="normal")
    assert np.array_equal(noise, np.ones(frame.shape))
    assert np.array_equal(frame.data, np.ones(frame.shape))
    assert frame.noise_mean == 1.5
    assert frame.noise_std == 0.25


def test_add_noise_gaussian_alias_uses_gaussian(monkeypatch):
    frame = make_frame()

    def fake_gaussian(x_mean, x_std, shape):
        return np.full(shape, x_mean + x_std)

    monkeypatch.setattr("setigen.frame.distributions.gaussian", fake_gaussian)

    noise = frame.add_noise(x_mean=2.0, x_std=3.0, noise_type="gaussian")
    assert np.array_equal(noise, np.full(frame.shape, 5.0))
    assert frame.noise_mean == 2.0
    assert frame.noise_std == 3.0


def test_add_noise_normal_with_min_uses_truncated_gaussian(monkeypatch):
    frame = make_frame()

    def fake_trunc(x_mean, x_std, x_min, shape):
        assert (x_mean, x_std, x_min, shape) == (4.0, 1.0, 0.5, frame.shape)
        return np.full(shape, 7.0)

    monkeypatch.setattr("setigen.frame.distributions.truncated_gaussian", fake_trunc)

    noise = frame.add_noise(x_mean=4.0, x_std=1.0, x_min=0.5, noise_type="normal")
    assert np.array_equal(noise, np.full(frame.shape, 7.0))
    assert frame.noise_mean == 4.0
    assert frame.noise_std == 1.0


def test_add_noise_invalid_noise_type_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, x_std=1.0, noise_type="bogus")


def test_add_noise_normal_missing_x_std_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, noise_type="normal")


def test_add_noise_updates_noise_stats_when_existing_noise_present(monkeypatch):
    frame = make_frame()
    frame.noise_mean = 1.0
    frame.noise_std = 2.0

    def fake_gaussian(x_mean, x_std, shape):
        return np.full(shape, 3.0)

    called = {"updated": False}

    def fake_update():
        called["updated"] = True
        frame.noise_mean = 9.0
        frame.noise_std = 8.0

    monkeypatch.setattr("setigen.frame.distributions.gaussian", fake_gaussian)
    monkeypatch.setattr(frame, "_update_noise_frame_stats", fake_update)

    noise = frame.add_noise(x_mean=1.5, x_std=0.25, noise_type="normal")
    assert np.array_equal(noise, np.full(frame.shape, 3.0))
    assert called["updated"] is True
    assert frame.noise_mean == 9.0
    assert frame.noise_std == 8.0


def test_add_noise_share_index_length_mismatch_raises_index_error():
    frame = make_frame()
    with pytest.raises(IndexError):
        frame.add_noise(
            x_mean_array=np.array([1.0, 2.0]),
            x_std_array=np.array([0.1]),
            x_min_array=np.array([0.0, 0.0]),
            share_index=True,
            noise_type="normal",
        )


def test_add_noise_share_index_false_uses_sample_from_obs_for_gaussian(monkeypatch):
    frame = make_frame()
    mean_arr = np.array([1.0, 2.0, 3.0])
    std_arr = np.array([0.1, 0.2, 0.3])

    def fake_sample_gaussian_params(x_mean_array, x_std_array):
        assert np.array_equal(x_mean_array, mean_arr)
        assert np.array_equal(x_std_array, std_arr)
        return 2.2, 0.22

    def fake_gaussian(x_mean, x_std, shape):
        assert (x_mean, x_std, shape) == (2.2, 0.22, frame.shape)
        return np.full(shape, 6.0)

    monkeypatch.setattr("setigen.frame.sample_from_obs.sample_gaussian_params", fake_sample_gaussian_params)
    monkeypatch.setattr("setigen.frame.distributions.gaussian", fake_gaussian)

    noise = frame.add_noise(
        x_mean_array=mean_arr,
        x_std_array=std_arr,
        share_index=False,
        noise_type="normal",
    )
    assert np.array_equal(noise, np.full(frame.shape, 6.0))
    assert frame.noise_mean == 2.2
    assert frame.noise_std == 0.22


def test_add_noise_share_index_false_with_min_uses_truncated_gaussian(monkeypatch):
    frame = make_frame()

    def fake_sample_gaussian_params(x_mean_array, x_std_array, x_min_array):
        return 4.4, 0.44, 0.04

    def fake_trunc(x_mean, x_std, x_min, shape):
        assert (x_mean, x_std, x_min, shape) == (4.4, 0.44, 0.04, frame.shape)
        return np.full(shape, 11.0)

    monkeypatch.setattr("setigen.frame.sample_from_obs.sample_gaussian_params", fake_sample_gaussian_params)
    monkeypatch.setattr("setigen.frame.distributions.truncated_gaussian", fake_trunc)

    noise = frame.add_noise(
        x_mean_array=np.array([1.0, 2.0]),
        x_std_array=np.array([0.1, 0.2]),
        x_min_array=np.array([0.01, 0.02]),
        share_index=False,
        noise_type="normal",
    )
    assert np.array_equal(noise, np.full(frame.shape, 11.0))
    assert frame.noise_mean == 4.4
    assert frame.noise_std == 0.44


def test_add_noise_with_existing_zero_stats_sets_noise_mean_and_std_from_parameters(monkeypatch):
    frame = make_frame()

    def fake_gaussian(x_mean, x_std, shape):
        return np.zeros(shape)

    monkeypatch.setattr("setigen.frame.distributions.gaussian", fake_gaussian)

    frame.zero_data()
    noise = frame.add_noise(x_mean=5.0, x_std=1.25, noise_type="normal")
    assert np.array_equal(noise, np.zeros(frame.shape))
    assert frame.noise_mean == 5.0
    assert frame.noise_std == 1.25
