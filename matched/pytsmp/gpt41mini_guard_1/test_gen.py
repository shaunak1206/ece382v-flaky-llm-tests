import numpy as np
import pytest
from pytsmp import utils


# --- Tests for sliding_dot_product ---


def test_sliding_dot_product_values_basic():
    Q = np.array([1, 2, 3])
    T = np.array([1, 2, 3, 4, 5])
    result = utils.sliding_dot_product(Q, T)
    # Manually computed dot products:
    # T[0:3] * Q = 1*3 + 2*2 + 3*1 reversed Q = [3,2,1]
    # But sliding dot product uses reversed Q so sliding dot product at position i = dot(Q[::-1], T[i:i+len(Q)])
    # We'll just trust the output shape and check with manual dot product:
    expected = np.array([
        np.dot(T[0:3], Q[::-1]),
        np.dot(T[1:4], Q[::-1]),
        np.dot(T[2:5], Q[::-1]),
    ])
    assert np.allclose(result, expected, rtol=1e-12, atol=1e-15)


def test_sliding_dot_product_length_error():
    Q = np.array([1, 2, 3, 4])
    T = np.array([1, 2, 3])  # shorter than Q
    with pytest.raises(ValueError):
        utils.sliding_dot_product(Q, T)


def test_sliding_dot_product_identical_series():
    T = np.linspace(0, 1, 10)
    Q = T[:3]
    result = utils.sliding_dot_product(Q, T)
    # The first sliding dot product must equal dot(Q[::-1], T[0:len(Q)])
    expected0 = np.dot(Q[::-1], T[0:len(Q)])
    assert np.isclose(result[0], expected0, rtol=1e-12, atol=1e-15)
    # Result length should be len(T)-len(Q)+1
    assert result.shape == (len(T) - len(Q) + 1,)


def test_sliding_dot_product_output_type_and_shape():
    Q = np.arange(4, dtype=float)
    T = np.arange(10, dtype=float)
    res = utils.sliding_dot_product(Q, T)
    # Check dtype is float (due to FFT irfft output)
    assert issubclass(res.dtype.type, np.floating)
    # Check shape
    assert res.shape == (len(T) - len(Q) + 1,)


def test_sliding_dot_product_zero_query():
    Q = np.zeros(3)
    T = np.arange(10)
    res = utils.sliding_dot_product(Q, T)
    # Dot product with zero query must be all zeros
    assert np.allclose(res, 0, atol=1e-15)


# --- Tests for rolling_avg_sd ---


def test_rolling_avg_sd_basic():
    t = np.array([1, 2, 3, 4, 5])
    window = 3
    avg, sd = utils.rolling_avg_sd(t, window)
    expected_avg = np.array([2, 3, 4])
    expected_sd = np.array([
        np.std(t[0:3], ddof=0),
        np.std(t[1:4], ddof=0),
        np.std(t[2:5], ddof=0),
    ])
    assert np.allclose(avg, expected_avg, rtol=1e-12, atol=1e-15)
    assert np.allclose(sd, expected_sd, rtol=1e-12, atol=1e-15)


def test_rolling_avg_sd_window_too_big():
    t = np.array([1, 2])
    window = 3
    with pytest.raises(ValueError):
        utils.rolling_avg_sd(t, window)


def test_rolling_avg_sd_constant_series():
    t = np.ones(5)
    window = 3
    avg, sd = utils.rolling_avg_sd(t, window)
    # Average should be ones
    assert np.allclose(avg, 1.0, rtol=1e-12, atol=1e-15)
    # Standard deviation should be zeros (numerically no negative)
    assert np.allclose(sd, 0.0, rtol=1e-12, atol=1e-15)


def test_rolling_avg_sd_output_shapes():
    t = np.arange(10)
    window = 4
    avg, sd = utils.rolling_avg_sd(t, window)
    expected_length = len(t) - window + 1
    assert avg.shape == (expected_length,)
    assert sd.shape == (expected_length,)


# --- Tests for calculate_distance_profile ---


def test_calculate_distance_profile_basic():
    # prepare inputs for calculate_distance_profile
    m = 3
    QT = np.array([3.0, 6.0, 9.0])
    mean_Q = 2.0
    sigma_Q = 1.0
    mean_T = np.array([1.0, 2.0, 3.0])
    sigma_T = np.array([1.0, 1.0, 1.0])
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    # Calculate expected distances manually:
    dist_manual = np.sqrt(np.maximum(
        2 * m * (1 - (QT - m * mean_Q * mean_T) / (m * sigma_Q * sigma_T)),
        0
    ))
    assert np.allclose(D, dist_manual, rtol=1e-12, atol=1e-15)


def test_calculate_distance_profile_sigma_q_zero_const_series():
    m = 4
    QT = np.array([0.0, 0.0])
    mean_Q = 1.0
    sigma_Q = 0.0
    mean_T = np.array([1.0, 1.0])
    sigma_T = np.array([0.1, 0.0])
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T, epsilon=1e-8)
    # When sigma_Q < epsilon (const Q) and sigma_T < epsilon (const subsequence), distance is zero
    # otherwise distance is sqrt(m)
    expected = np.array([np.sqrt(m), 0.0])
    assert np.allclose(D, expected, rtol=1e-12, atol=1e-15)


def test_calculate_distance_profile_sigma_t_zero():
    m = 5
    QT = np.array([5.0, 4.5])
    mean_Q = 1.0
    sigma_Q = 0.5
    mean_T = np.array([1.0, 1.0])
    sigma_T = np.array([0.0, 0.001])  # one zero, one small positive
    epsilon = 1e-3
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T, epsilon=epsilon)
    expected = np.array([np.sqrt(m), np.sqrt(
        max(0, 2 * m * (1 - (QT[1] - m * mean_Q * mean_T[1]) / (m * sigma_Q * sigma_T[1])))
    )])
    assert np.allclose(D, expected, rtol=1e-12, atol=1e-15)


def test_calculate_distance_profile_dimension_mismatch_raises():
    m = 3
    QT = np.array([1.0, 2.0])
    mean_Q = 1.0
    sigma_Q = 2.0
    mean_T = np.array([1.0])
    sigma_T = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)


def test_calculate_distance_profile_meanQ_sigmaQ_array_length_mismatch():
    m = 3
    QT = np.array([1.0, 2.0])
    mean_Q = np.array([1.0])
    sigma_Q = np.array([1.0, 2.0])
    mean_T = np.array([1.0, 2.0])
    sigma_T = np.array([1.0, 1.0])
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)


def test_calculate_distance_profile_values_positive_definite_inside_sqrt():
    m = 3
    # Select values so inside sqrt is zero or positive, no negative underflow
    QT = np.array([3.0, 3.0])
    mean_Q = 2.0
    sigma_Q = 1.0
    mean_T = np.array([1.0, 2.0])
    sigma_T = np.array([1.0, 1.0])
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    # All distance values should be >= 0 (due to sqrt)
    assert np.all(D >= 0)
    # No NaNs or infs
    assert np.all(np.isfinite(D))


# --- Combined tests for sliding_dot_product + rolling_avg_sd + calculate_distance_profile ---

def test_mass_algorithm_consistency():
    # Compare utils.mass to manually calling components
    T = np.arange(20, dtype=float)
    Q = np.array([1.0, 2.0, 3.0, 2.5])
    # Calculate pieces manually
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, len(Q))
    mean_Q = np.mean(Q)
    sigma_Q = np.std(Q)
    expected_D = utils.calculate_distance_profile(QT, len(Q), mean_Q, sigma_Q, mean_T, sigma_T)
    # Now calculate via mass (which calls same functions)
    mass_D = utils.mass(Q, T)
    assert mass_D.shape == expected_D.shape
    assert np.allclose(mass_D, expected_D, rtol=1e-12, atol=1e-15)


def test_mass_raises_on_short_T():
    Q = np.arange(5)
    T = np.arange(4)
    with pytest.raises(ValueError):
        utils.mass(Q, T)


def test_mass_with_constant_query():
    T = np.arange(10)
    Q = np.ones(3)
    # sigma_Q=0, triggers constant sequence path in distance profile
    D = utils.mass(Q, T)
    # distances non-negative and length correct
    assert D.shape == (len(T) - len(Q) + 1,)
    assert np.all(D >= 0)
    # Check that the value sqrt(m) appears in the result (expected for const query)
    assert np.any(np.isclose(D, np.sqrt(len(Q)), atol=1e-15))


def test_rolling_avg_sd_matches_rolling_sum():
    # For a random small series, check rolling_avg_sd returns average consistent with rolling_sum
    t = np.arange(10, dtype=float)
    w = 4
    sum_ = utils.rolling_sum(t, w)
    avg, _ = utils.rolling_avg_sd(t, w)
    expected_avg = sum_ / w
    assert np.allclose(avg, expected_avg, rtol=1e-12, atol=1e-15)


def test_rolling_avg_sd_sd_non_negative():
    # Standard deviations returned should never be negative
    t = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    window = 3
    _, sd = utils.rolling_avg_sd(t, window)
    assert np.all(sd >= 0)


def test_sliding_dot_product_padding_ignored():
    # The internal padding in sliding_dot_product should not change output length
    Q = np.array([1, 2, 3], dtype=float)
    T = np.arange(11, dtype=float)  # odd length to trigger pad
    res = utils.sliding_dot_product(Q, T)
    expected_length = len(T) - len(Q) + 1
    assert len(res) == expected_length


# Additional sanity tests


def test_sliding_dot_product_constant_series():
    Q = np.ones(4)
    T = np.ones(10)
    res = utils.sliding_dot_product(Q, T)
    # For constant series, dot product = sum(Q) * value of each subsequence (which is window length)
    expected_value = np.sum(Q) * np.ones(len(T) - len(Q) + 1)
    assert np.allclose(res, expected_value, rtol=1e-12, atol=1e-15)


def test_calculate_distance_profile_zero_std_dev_T():
    m = 4
    QT = np.array([8, 8])
    mean_Q = 2
    sigma_Q = 1
    mean_T = np.array([2, 2])
    sigma_T = np.array([0, 0])  # zero std deviation in subsequences means constant subsequence
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    # Distance should be either 0 or sqrt(m), here sigma_T < epsilon so should be sqrt(m)
    assert np.allclose(D, np.sqrt(m), rtol=1e-12, atol=1e-15)
