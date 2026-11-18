import numpy as np
import pytest

from setigen.frame import Frame


def make_frame():
    return Frame(fchans=4, tchans=3, df=1, dt=1, fch1=10)


def test_get_data_returns_internal_array_reference():
    frame = make_frame()
    data = frame.get_data()
    assert data is frame.data


def test_get_data_with_db_false_returns_same_values():
    frame = make_frame()
    frame.data[:] = np.array([[1.0, 2.0, 3.0, 4.0],
                              [5.0, 6.0, 7.0, 8.0],
                              [9.0, 10.0, 11.0, 12.0]])
    out = frame.get_data(use_db=False)
    assert np.array_equal(out, frame.data)


def test_get_data_with_db_true_applies_log10_scaling():
    frame = make_frame()
    frame.data[:] = 10.0
    out = frame.get_data(use_db=True)
    assert np.allclose(out, 10 * np.log10(10.0))


def test_get_data_with_db_true_preserves_shape():
    frame = make_frame()
    frame.data[:] = np.arange(frame.data.size).reshape(frame.shape) + 1
    out = frame.get_data(use_db=True)
    assert out.shape == frame.shape


def test_add_noise_chi2_adds_returned_noise_and_updates_data(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 2.5)

    called = {}

    def fake_chi2(x_mean, df, shape):
        called["args"] = (x_mean, df, shape)
        return noise

    monkeypatch.setattr("setigen.distributions.chi2", fake_chi2)

    returned = frame.add_noise(x_mean=7.0, noise_type="chi2")

    assert np.array_equal(returned, noise)
    assert np.array_equal(frame.data, noise)
    assert called["args"] == (7.0, frame.chi2_df, frame.shape)
    assert frame.noise_mean == 7.0
    assert frame.noise_std == np.sqrt(2 * frame.chi2_df) * 7.0 / frame.chi2_df


def test_add_noise_chi2_only_uses_x_mean(monkeypatch):
    frame = make_frame()
    noise = np.ones(frame.shape)

    def fake_chi2(x_mean, df, shape):
        return noise

    monkeypatch.setattr("setigen.distributions.chi2", fake_chi2)

    returned = frame.add_noise(x_mean=3.0, x_std=999.0, x_min=-5.0, noise_type="chi2")

    assert np.array_equal(returned, noise)
    assert np.array_equal(frame.data, noise)


def test_add_noise_normal_uses_gaussian_when_no_min(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 1.25)
    called = {}

    def fake_gaussian(x_mean, x_std, shape):
        called["args"] = (x_mean, x_std, shape)
        return noise

    monkeypatch.setattr("setigen.distributions.gaussian", fake_gaussian)

    returned = frame.add_noise(x_mean=2.0, x_std=0.5, noise_type="normal")

    assert np.array_equal(returned, noise)
    assert called["args"] == (2.0, 0.5, frame.shape)
    assert np.array_equal(frame.data, noise)
    assert frame.noise_mean == 2.0
    assert frame.noise_std == 0.5


def test_add_noise_gaussian_alias_uses_gaussian(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 4.0)

    def fake_gaussian(x_mean, x_std, shape):
        return noise

    monkeypatch.setattr("setigen.distributions.gaussian", fake_gaussian)

    returned = frame.add_noise(x_mean=1.5, x_std=0.25, noise_type="gaussian")

    assert np.array_equal(returned, noise)
    assert np.array_equal(frame.data, noise)


def test_add_noise_normal_with_x_min_uses_truncated_gaussian(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 3.3)
    called = {}

    def fake_truncated_gaussian(x_mean, x_std, x_min, shape):
        called["args"] = (x_mean, x_std, x_min, shape)
        return noise

    monkeypatch.setattr("setigen.distributions.truncated_gaussian", fake_truncated_gaussian)

    returned = frame.add_noise(x_mean=2.0, x_std=0.5, x_min=0.0, noise_type="normal")

    assert np.array_equal(returned, noise)
    assert called["args"] == (2.0, 0.5, 0.0, frame.shape)
    assert np.array_equal(frame.data, noise)


def test_add_noise_invalid_noise_type_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, noise_type="invalid")


def test_add_noise_normal_missing_x_std_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, noise_type="normal")


def test_add_noise_updates_existing_noise_stats_instead_of_leaving_defaults(monkeypatch):
    frame = make_frame()
    frame.noise_mean = 5.0
    frame.noise_std = 6.0
    noise = np.full(frame.shape, 1.0)

    def fake_gaussian(x_mean, x_std, shape):
        return noise

    monkeypatch.setattr("setigen.distributions.gaussian", fake_gaussian)
    monkeypatch.setattr(frame, "_update_noise_frame_stats", lambda: setattr(frame, "noise_mean", 123.0) or setattr(frame, "noise_std", 456.0))

    frame.add_noise(x_mean=2.0, x_std=0.5, noise_type="normal")

    assert frame.noise_mean == 123.0
    assert frame.noise_std == 456.0


def test_add_noise_from_obs_chi2_uses_sampled_mean(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 7.7)

    monkeypatch.setattr(np.random, "choice", lambda arr: arr[1])

    called = {}

    def fake_chi2(x_mean, df, shape):
        called["args"] = (x_mean, df, shape)
        return noise

    monkeypatch.setattr("setigen.distributions.chi2", fake_chi2)

    x_mean_array = np.array([1.0, 2.0, 3.0])
    x_std_array = np.array([0.1, 0.2, 0.3])
    x_min_array = np.array([0.0, 0.0, 0.0])

    returned = frame.add_noise_from_obs(x_mean_array=x_mean_array,
                                        x_std_array=x_std_array,
                                        x_min_array=x_min_array,
                                        noise_type="chi2")

    assert np.array_equal(returned, noise)
    assert called["args"] == (2.0, frame.chi2_df, frame.shape)
    assert np.array_equal(frame.data, noise)


def test_add_noise_from_obs_gaussian_share_index_selects_same_index(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 9.9)

    monkeypatch.setattr(np.random, "randint", lambda n: 1)

    called = {}

    def fake_gaussian(x_mean, x_std, shape):
        called["args"] = (x_mean, x_std, shape)
        return noise

    monkeypatch.setattr("setigen.distributions.gaussian", fake_gaussian)

    x_mean_array = np.array([1.0, 2.0, 3.0])
    x_std_array = np.array([0.1, 0.2, 0.3])

    returned = frame.add_noise_from_obs(x_mean_array=x_mean_array,
                                        x_std_array=x_std_array,
                                        share_index=True,
                                        noise_type="normal")

    assert np.array_equal(returned, noise)
    assert called["args"] == (2.0, 0.2, frame.shape)
    assert np.array_equal(frame.data, noise)


def test_add_noise_from_obs_gaussian_share_index_length_mismatch_raises(monkeypatch):
    frame = make_frame()

    x_mean_array = np.array([1.0, 2.0])
    x_std_array = np.array([0.1])
    x_min_array = np.array([0.0, 0.0])

    with pytest.raises(IndexError):
        frame.add_noise_from_obs(x_mean_array=x_mean_array,
                                 x_std_array=x_std_array,
                                 x_min_array=x_min_array,
                                 share_index=True,
                                 noise_type="normal")


def test_add_noise_from_obs_gaussian_without_share_index_uses_sample_helper(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 8.8)

    called = {}

    def fake_sample_gaussian_params(x_mean_array, x_std_array, x_min_array=None):
        called["args"] = (x_mean_array, x_std_array, x_min_array)
        if x_min_array is None:
            return 4.4, 0.44
        return 4.4, 0.44, 0.04

    def fake_gaussian(x_mean, x_std, shape):
        called["gaussian"] = (x_mean, x_std, shape)
        return noise

    monkeypatch.setattr("setigen.sample_from_obs.sample_gaussian_params", fake_sample_gaussian_params)
    monkeypatch.setattr("setigen.distributions.gaussian", fake_gaussian)

    x_mean_array = np.array([1.0, 2.0, 3.0])
    x_std_array = np.array([0.1, 0.2, 0.3])

    returned = frame.add_noise_from_obs(x_mean_array=x_mean_array,
                                        x_std_array=x_std_array,
                                        share_index=False,
                                        noise_type="normal")

    assert np.array_equal(returned, noise)
    assert called["args"][0] is x_mean_array
    assert called["args"][1] is x_std_array
    assert called["args"][2] is None
    assert called["gaussian"] == (4.4, 0.44, frame.shape)
    assert np.array_equal(frame.data, noise)
