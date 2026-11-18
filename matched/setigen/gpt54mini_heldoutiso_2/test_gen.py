import sys
import types
import numpy as np
import pytest

from setigen.frame import Frame


def make_frame(fchans=4, tchans=3, df=1.0, dt=2.0, fch1=10.0, data=None):
    return Frame(fchans=fchans, tchans=tchans, df=df, dt=dt, fch1=fch1, data=data)


def test_get_data_returns_internal_array_without_db():
    frame = make_frame()
    frame.data[:] = np.arange(frame.data.size).reshape(frame.shape)

    out = frame.get_data()
    assert out is frame.data
    np.testing.assert_array_equal(out, frame.data)


def test_get_data_db_converts_to_log10_scale():
    frame = make_frame()
    frame.data[:] = 10.0

    out = frame.get_data(use_db=True)
    np.testing.assert_allclose(out, np.full(frame.shape, 10.0))


def test_get_data_db_handles_nonuniform_values():
    frame = make_frame()
    frame.data[:] = np.array(
        [
            [1.0, 10.0, 100.0, 1000.0],
            [2.0, 20.0, 200.0, 2000.0],
            [3.0, 30.0, 300.0, 3000.0],
        ]
    )

    out = frame.get_data(use_db=True)
    expected = 10 * np.log10(frame.data)
    np.testing.assert_allclose(out, expected)


def test_add_noise_chi2_adds_noise_and_returns_it(monkeypatch):
    frame = make_frame(df=2.0, dt=2.0)
    noise = np.full(frame.shape, 7.5)

    called = {}

    def fake_chi2(x_mean, df, shape):
        called["args"] = (x_mean, df, shape)
        return noise

    monkeypatch.setattr("setigen.frame.distributions.chi2", fake_chi2)

    out = frame.add_noise(x_mean=5.0, noise_type="chi2")

    assert called["args"] == (5.0, frame.chi2_df, frame.shape)
    np.testing.assert_array_equal(out, noise)
    np.testing.assert_array_equal(frame.data, noise)
    assert frame.noise_mean == 5.0
    assert frame.noise_std == np.sqrt(2 * frame.chi2_df) * 5.0 / frame.chi2_df


def test_add_noise_chi2_updates_existing_noise_stats(monkeypatch):
    frame = make_frame()
    frame.noise_mean = 1.0
    frame.noise_std = 2.0

    noise = np.ones(frame.shape)

    monkeypatch.setattr("setigen.frame.distributions.chi2", lambda x_mean, df, shape: noise)
    updated = {}

    def fake_update():
        updated["called"] = True
        frame.noise_mean = 11.0
        frame.noise_std = 22.0

    monkeypatch.setattr(frame, "_update_noise_frame_stats", fake_update)

    frame.add_noise(x_mean=3.0, noise_type="chi2")

    assert updated["called"] is True
    assert frame.noise_mean == 11.0
    assert frame.noise_std == 22.0


def test_add_noise_normal_uses_gaussian_when_no_min(monkeypatch):
    frame = make_frame()
    expected = np.full(frame.shape, 4.2)

    called = {}

    def fake_gaussian(x_mean, x_std, shape):
        called["args"] = (x_mean, x_std, shape)
        return expected

    monkeypatch.setattr("setigen.frame.distributions.gaussian", fake_gaussian)

    out = frame.add_noise(x_mean=1.5, x_std=0.7, noise_type="normal")

    assert called["args"] == (1.5, 0.7, frame.shape)
    np.testing.assert_array_equal(out, expected)
    assert frame.noise_mean == 1.5
    assert frame.noise_std == 0.7


def test_add_noise_gaussian_alias_uses_gaussian(monkeypatch):
    frame = make_frame()
    expected = np.zeros(frame.shape)

    monkeypatch.setattr("setigen.frame.distributions.gaussian", lambda x_mean, x_std, shape: expected)

    out = frame.add_noise(x_mean=0.0, x_std=1.0, noise_type="gaussian")

    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(frame.data, expected)


def test_add_noise_normal_with_min_uses_truncated_gaussian(monkeypatch):
    frame = make_frame()
    expected = np.full(frame.shape, 9.0)

    called = {}

    def fake_trunc(x_mean, x_std, x_min, shape):
        called["args"] = (x_mean, x_std, x_min, shape)
        return expected

    monkeypatch.setattr("setigen.frame.distributions.truncated_gaussian", fake_trunc)

    out = frame.add_noise(x_mean=2.0, x_std=3.0, x_min=1.0, noise_type="normal")

    assert called["args"] == (2.0, 3.0, 1.0, frame.shape)
    np.testing.assert_array_equal(out, expected)
    assert frame.noise_mean == 2.0
    assert frame.noise_std == 3.0


def test_add_noise_normal_requires_x_std():
    frame = make_frame()

    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, noise_type="normal")


def test_add_noise_invalid_noise_type_exits():
    frame = make_frame()

    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, x_std=2.0, noise_type="not-a-type")


def test_add_noise_updates_frame_data_in_place(monkeypatch):
    frame = make_frame()
    frame.data[:] = 5.0
    noise = np.arange(frame.data.size, dtype=float).reshape(frame.shape)

    monkeypatch.setattr("setigen.frame.distributions.gaussian", lambda x_mean, x_std, shape: noise)

    frame.add_noise(x_mean=0.0, x_std=1.0, noise_type="normal")

    np.testing.assert_array_equal(frame.data, 5.0 + noise)


def test_add_noise_from_obs_uses_shared_index_and_validates_lengths(monkeypatch):
    frame = make_frame()
    x_mean_array = np.array([1.0, 2.0, 3.0])
    x_std_array = np.array([0.1, 0.2, 0.3])
    x_min_array = np.array([0.0, 0.0, 0.0])

    monkeypatch.setattr(np.random, "randint", lambda n: 1)
    monkeypatch.setattr("setigen.frame.distributions.truncated_gaussian",
                        lambda x_mean, x_std, x_min, shape: np.full(shape, x_mean + x_std + x_min))

    out = frame.add_noise_from_obs(
        x_mean_array=x_mean_array,
        x_std_array=x_std_array,
        x_min_array=x_min_array,
        share_index=True,
        noise_type="normal",
    )

    np.testing.assert_array_equal(out, np.full(frame.shape, 2.2))
    assert frame.noise_mean == 2.0
    assert frame.noise_std == 0.2


def test_add_noise_from_obs_share_index_length_mismatch_raises():
    frame = make_frame()

    with pytest.raises(IndexError):
        frame.add_noise_from_obs(
            x_mean_array=np.array([1.0, 2.0]),
            x_std_array=np.array([0.1]),
            x_min_array=np.array([0.0, 0.0]),
            share_index=True,
            noise_type="normal",
        )


def test_add_noise_from_obs_share_index_false_uses_sample_gaussian_params(monkeypatch):
    frame = make_frame()
    x_mean_array = np.array([1.0, 2.0])
    x_std_array = np.array([0.1, 0.2])
    x_min_array = np.array([0.0, 0.0])

    sampled = {}

    def fake_sample_gaussian_params(mean_arr, std_arr, min_arr=None):
        sampled["args"] = (mean_arr, std_arr, min_arr)
        return 8.0, 0.5, 0.0

    monkeypatch.setattr("setigen.frame.sample_from_obs.sample_gaussian_params", fake_sample_gaussian_params)
    monkeypatch.setattr("setigen.frame.distributions.truncated_gaussian",
                        lambda x_mean, x_std, x_min, shape: np.full(shape, x_mean))

    out = frame.add_noise_from_obs(
        x_mean_array=x_mean_array,
        x_std_array=x_std_array,
        x_min_array=x_min_array,
        share_index=False,
        noise_type="normal",
    )

    assert sampled["args"][0] is x_mean_array
    assert sampled["args"][1] is x_std_array
    assert sampled["args"][2] is x_min_array
    np.testing.assert_array_equal(out, np.full(frame.shape, 8.0))


def test_add_noise_from_obs_defaults_loads_sample_params_and_scales(monkeypatch):
    frame = make_frame(dt=2.8633115306666666)

    sample_noise_params = np.array(
        [
            [10.0, 1.0, 0.5],
            [20.0, 2.0, 1.0],
        ]
    )

    def fake_load(path):
        assert path.endswith("assets/sample_noise_params.npy")
        return sample_noise_params

    monkeypatch.setattr(np, "load", fake_load)
    monkeypatch.setattr(np.random, "choice", lambda arr: arr[0])
    monkeypatch.setattr("setigen.frame.distributions.chi2", lambda x_mean, df, shape: np.full(shape, x_mean))

    out = frame.add_noise_from_obs(noise_type="chi2")

    np.testing.assert_array_equal(out, np.full(frame.shape, 20.0))
    assert frame.noise_mean == 20.0
    assert frame.noise_std == np.sqrt(2 * frame.chi2_df) * 20.0 / frame.chi2_df
