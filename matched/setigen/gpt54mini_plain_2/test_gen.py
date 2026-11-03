import numpy as np
import pytest

from setigen.frame import Frame
import setigen.frame as frame_module


def make_frame():
    return Frame(
        fchans=4,
        tchans=3,
        df=1,
        dt=1,
        fch1=10,
    )


def test_get_data_returns_internal_array_without_db():
    frame = make_frame()
    frame.data[:] = np.arange(frame.data.size).reshape(frame.shape)

    out = frame.get_data()

    assert out is frame.data
    np.testing.assert_array_equal(out, frame.data)


def test_get_data_use_db_applies_10_log10():
    frame = make_frame()
    frame.data[:] = 10.0

    out = frame.get_data(use_db=True)

    np.testing.assert_allclose(out, np.full(frame.shape, 10.0))


def test_get_data_use_db_handles_nonpositive_values_as_numpy_does():
    frame = make_frame()
    frame.data[:] = np.array(
        [
            [1.0, 0.0, -1.0, 10.0],
            [2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0],
        ]
    )

    out = frame.get_data(use_db=True)

    assert np.isfinite(out[0, 0])
    assert np.isneginf(out[0, 1])
    assert np.isnan(out[0, 2])


def test_add_noise_chi2_updates_data_and_returns_noise(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 2.5)

    def fake_chi2(x_mean, df, shape):
        assert x_mean == 7
        assert df == frame.chi2_df
        assert shape == frame.shape
        return noise

    monkeypatch.setattr(frame_module.distributions, "chi2", fake_chi2)

    out = frame.add_noise(x_mean=7, noise_type="chi2")

    np.testing.assert_array_equal(out, noise)
    np.testing.assert_array_equal(frame.data, noise)
    assert frame.noise_mean == 7
    assert frame.noise_std == np.sqrt(2 * frame.chi2_df) * 7 / frame.chi2_df


def test_add_noise_chi2_uses_existing_stats_branch(monkeypatch):
    frame = make_frame()
    frame.noise_mean = 1.0
    frame.noise_std = 2.0
    noise = np.ones(frame.shape)

    monkeypatch.setattr(frame_module.distributions, "chi2", lambda x_mean, df, shape: noise)
    called = {"update": False}

    def fake_update():
        called["update"] = True

    monkeypatch.setattr(frame, "_update_noise_frame_stats", fake_update)

    frame.add_noise(x_mean=3, noise_type="chi2")

    assert called["update"] is True
    assert frame.noise_mean == 1.0
    assert frame.noise_std == 2.0


def test_add_noise_normal_requires_x_std(monkeypatch):
    frame = make_frame()

    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, noise_type="normal")


def test_add_noise_invalid_noise_type_exits():
    frame = make_frame()

    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, x_std=1.0, noise_type="not-a-type")


def test_add_noise_normal_calls_gaussian_when_no_min(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 4.2)

    def fake_gaussian(x_mean, x_std, shape):
        assert x_mean == 5.0
        assert x_std == 1.5
        assert shape == frame.shape
        return noise

    monkeypatch.setattr(frame_module.distributions, "gaussian", fake_gaussian)

    out = frame.add_noise(x_mean=5.0, x_std=1.5, noise_type="gaussian")

    np.testing.assert_array_equal(out, noise)
    np.testing.assert_array_equal(frame.data, noise)
    assert frame.noise_mean == 5.0
    assert frame.noise_std == 1.5


def test_add_noise_normal_calls_truncated_gaussian_with_min(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 3.3)

    def fake_trunc(x_mean, x_std, x_min, shape):
        assert x_mean == 2.0
        assert x_std == 0.5
        assert x_min == 1.0
        assert shape == frame.shape
        return noise

    monkeypatch.setattr(frame_module.distributions, "truncated_gaussian", fake_trunc)

    out = frame.add_noise(x_mean=2.0, x_std=0.5, x_min=1.0, noise_type="normal")

    np.testing.assert_array_equal(out, noise)
    np.testing.assert_array_equal(frame.data, noise)
    assert frame.noise_mean == 2.0
    assert frame.noise_std == 0.5


def test_add_noise_normal_with_existing_stats_triggers_update(monkeypatch):
    frame = make_frame()
    frame.noise_mean = 1.0
    frame.noise_std = 2.0
    noise = np.ones(frame.shape)

    monkeypatch.setattr(frame_module.distributions, "gaussian", lambda x_mean, x_std, shape: noise)

    called = {"update": False}

    def fake_update():
        called["update"] = True
        frame.noise_mean = 9.0
        frame.noise_std = 8.0

    monkeypatch.setattr(frame, "_update_noise_frame_stats", fake_update)

    frame.add_noise(x_mean=5.0, x_std=1.0, noise_type="normal")

    assert called["update"] is True
    assert frame.noise_mean == 9.0
    assert frame.noise_std == 8.0


def test_add_noise_zero_initial_stats_sets_mean_and_std_to_params(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 1.0)

    monkeypatch.setattr(frame_module.distributions, "gaussian", lambda x_mean, x_std, shape: noise)

    frame.add_noise(x_mean=11.0, x_std=2.0, noise_type="normal")

    assert frame.noise_mean == 11.0
    assert frame.noise_std == 2.0


def test_add_noise_accumulates_onto_existing_data(monkeypatch):
    frame = make_frame()
    frame.data[:] = 5.0
    noise = np.full(frame.shape, 2.0)

    monkeypatch.setattr(frame_module.distributions, "chi2", lambda x_mean, df, shape: noise)

    frame.add_noise(x_mean=1.0, noise_type="chi2")

    np.testing.assert_array_equal(frame.data, np.full(frame.shape, 7.0))


def test_add_noise_from_obs_share_index_length_mismatch_raises(monkeypatch):
    frame = make_frame()

    with pytest.raises(IndexError):
        frame.add_noise_from_obs(
            x_mean_array=np.array([1.0, 2.0]),
            x_std_array=np.array([0.1]),
            x_min_array=np.array([0.0, 0.0]),
            share_index=True,
            noise_type="normal",
        )


def test_add_noise_from_obs_share_index_uses_same_random_index(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 6.0)

    monkeypatch.setattr(frame_module.np.random, "randint", lambda n: 1)

    captured = {}

    def fake_trunc(x_mean, x_std, x_min, shape):
        captured["args"] = (x_mean, x_std, x_min, shape)
        return noise

    monkeypatch.setattr(frame_module.distributions, "truncated_gaussian", fake_trunc)

    out = frame.add_noise_from_obs(
        x_mean_array=np.array([10.0, 20.0, 30.0]),
        x_std_array=np.array([1.0, 2.0, 3.0]),
        x_min_array=np.array([0.0, 4.0, 5.0]),
        share_index=True,
        noise_type="normal",
    )

    np.testing.assert_array_equal(out, noise)
    assert captured["args"] == (20.0, 2.0, 4.0, frame.shape)
    assert frame.noise_mean == 20.0
    assert frame.noise_std == 2.0


def test_add_noise_from_obs_share_index_false_uses_sample_gaussian_params(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 8.0)

    called = {"sample": False}

    def fake_sample_gaussian_params(x_mean_array, x_std_array, x_min_array=None):
        called["sample"] = True
        assert x_min_array is None
        return 1.1, 2.2

    monkeypatch.setattr(frame_module.sample_from_obs, "sample_gaussian_params", fake_sample_gaussian_params)
    monkeypatch.setattr(frame_module.distributions, "gaussian", lambda x_mean, x_std, shape: noise)

    out = frame.add_noise_from_obs(
        x_mean_array=np.array([1.0, 2.0]),
        x_std_array=np.array([3.0, 4.0]),
        share_index=False,
        noise_type="normal",
    )

    assert called["sample"] is True
    np.testing.assert_array_equal(out, noise)
    assert frame.noise_mean == 1.1
    assert frame.noise_std == 2.2


def test_add_noise_from_obs_chi2_uses_sampled_mean(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 9.0)

    monkeypatch.setattr(frame_module.np.random, "choice", lambda arr: arr[2])

    captured = {}

    def fake_chi2(x_mean, df, shape):
        captured["args"] = (x_mean, df, shape)
        return noise

    monkeypatch.setattr(frame_module.distributions, "chi2", fake_chi2)

    out = frame.add_noise_from_obs(
        x_mean_array=np.array([1.0, 2.0, 3.0]),
        noise_type="chi2",
    )

    np.testing.assert_array_equal(out, noise)
    assert captured["args"] == (3.0, frame.chi2_df, frame.shape)
    assert frame.noise_mean == 3.0
    assert frame.noise_std == np.sqrt(2 * frame.chi2_df) * 3.0 / frame.chi2_df
