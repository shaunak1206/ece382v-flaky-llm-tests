import numpy as np
import pytest

from nonlinshrink.nonlinshrink import shrink_cov


def make_data(n, p):
    return np.random.randn(n, p)


def test_output_shape_and_type_low_dim():
    np.random.seed(0)
    data = make_data(100, 5)
    out = shrink_cov(data)
    assert isinstance(out, np.ndarray)
    assert out.shape == (5, 5)
    assert np.all(np.isfinite(out))


def test_output_is_symmetric():
    np.random.seed(1)
    data = make_data(80, 6)
    out = shrink_cov(data)
    np.testing.assert_allclose(out, out.T, rtol=1e-8, atol=1e-10)


def test_output_positive_definite_low_dim():
    np.random.seed(2)
    data = make_data(120, 8)
    out = shrink_cov(data)
    eig = np.linalg.eigvalsh((out + out.T) / 2)
    assert np.all(eig > 0)


def test_one_dimensional_input_raises_assertion():
    np.random.seed(3)
    with pytest.raises(AssertionError):
        shrink_cov(np.random.randn(50))


def test_three_dimensional_input_raises_assertion():
    np.random.seed(4)
    with pytest.raises(AssertionError):
        shrink_cov(np.random.randn(20, 3, 2))


def test_too_few_samples_default_k_raises():
    np.random.seed(5)
    # 12 rows with default demeaning gives effective sample size 11 < 12
    with pytest.raises(AssertionError):
        shrink_cov(make_data(12, 3))
    with pytest.raises(AssertionError):
        shrink_cov(make_data(11, 3))


def test_minimum_sample_size_accepted():
    np.random.seed(6)
    # 13 rows with default k=1 -> n=12, accepted
    out = shrink_cov(make_data(13, 3))
    assert out.shape == (3, 3)
    # 12 rows with k=0 -> n=12, accepted
    out0 = shrink_cov(make_data(12, 2), k=0)
    assert out0.shape == (2, 2)
    assert np.all(np.isfinite(out0))


def test_too_few_samples_with_large_k_raises():
    np.random.seed(7)
    with pytest.raises(AssertionError):
        shrink_cov(make_data(20, 3), k=9)


def test_singular_matrix_raises_value_error():
    np.random.seed(8)
    base = make_data(60, 2)
    data = np.hstack([base, base[:, :1]])  # duplicated column -> singular
    with pytest.raises(ValueError):
        shrink_cov(data)


def test_zero_column_raises_value_error():
    np.random.seed(9)
    data = make_data(60, 3)
    data[:, 1] = 0.0
    with pytest.raises(ValueError):
        shrink_cov(data)


def test_high_dimensional_case_p_greater_than_n():
    np.random.seed(10)
    data = make_data(15, 30)
    out = shrink_cov(data)
    assert out.shape == (30, 30)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, out.T, rtol=1e-8, atol=1e-10)
    eig = np.linalg.eigvalsh((out + out.T) / 2)
    assert eig.min() > -1e-8


def test_input_not_modified():
    np.random.seed(11)
    data = make_data(50, 4)
    copy = data.copy()
    shrink_cov(data)
    np.testing.assert_array_equal(data, copy)
    shrink_cov(data, k=0)
    np.testing.assert_array_equal(data, copy)


def test_default_k_matches_prior_demeaning_with_k1():
    np.random.seed(12)
    data = make_data(70, 5) + 3.0
    demeaned = data - data.mean(axis=0)
    a = shrink_cov(data)
    b = shrink_cov(demeaned, k=1)
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8)


def test_translation_invariance_with_default_k():
    np.random.seed(13)
    data = make_data(70, 5)
    a = shrink_cov(data)
    b = shrink_cov(data + 10.0)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)


def test_k0_does_not_demean():
    np.random.seed(14)
    data = make_data(70, 4) + 5.0
    a = shrink_cov(data)
    b = shrink_cov(data, k=0)
    assert not np.allclose(a, b, rtol=1e-3, atol=1e-3)


def test_scale_equivariance():
    np.random.seed(15)
    data = make_data(90, 5)
    c = 3.0
    a = shrink_cov(data)
    b = shrink_cov(c * data)
    np.testing.assert_allclose(b, c ** 2 * a, rtol=1e-6, atol=1e-8)


def test_rotation_equivariance():
    np.random.seed(16)
    data = make_data(100, 5)
    q, _ = np.linalg.qr(np.random.randn(5, 5))
    a = shrink_cov(data)
    b = shrink_cov(data.dot(q))
    np.testing.assert_allclose(b, q.T.dot(a).dot(q), rtol=1e-5, atol=1e-7)


def test_row_permutation_invariance():
    np.random.seed(17)
    data = make_data(60, 4)
    perm = np.random.permutation(60)
    a = shrink_cov(data)
    b = shrink_cov(data[perm])
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8)


def test_commutes_with_sample_covariance():
    np.random.seed(18)
    data = make_data(100, 6)
    centered = data - data.mean(axis=0)
    sample = centered.T.dot(centered) / (100 - 1)
    out = shrink_cov(data)
    np.testing.assert_allclose(sample.dot(out), out.dot(sample),
                               rtol=1e-6, atol=1e-8)


def test_close_to_sample_covariance_for_large_n():
    np.random.seed(19)
    data = make_data(3000, 3)
    centered = data - data.mean(axis=0)
    sample = centered.T.dot(centered) / (3000 - 1)
    out = shrink_cov(data)
    np.testing.assert_allclose(out, sample, atol=0.15)
