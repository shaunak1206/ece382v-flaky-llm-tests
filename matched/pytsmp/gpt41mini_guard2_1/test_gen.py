import numpy as np
import pytest
from pytsmp import utils


def test_sliding_dot_product_basic_known_example():
    # Q shorter than T, simple numeric series
    Q = np.array([1, 2, 3], dtype=float)
    T = np.array([1, 2, 3, 4, 5], dtype=float)
    # Manual dot products:
    # [1*1+2*2+3*3, 2*1+3*2+4*3, 3*1+4*2+5*3] = [14, 20, 26]
    expected = np.array([14, 20, 26], dtype=float)
    result = utils.sliding_dot_product(Q, T)
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-12)


def test_sliding_dot_product_value_error_when_Q_longer_than_T():
    Q = np.array([1, 2, 3])
    T = np.array([1])
    with pytest.raises(ValueError):
        utils.sliding_dot_product(Q, T)


def test_sliding_dot_product_result_length_and_type():
    Q = np.arange(4, dtype=float) + 1  # length 4
    T = np.arange(10, dtype=float) + 1  # length 10
    result = utils.sliding_dot_product(Q, T)
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(T) - len(Q) + 1,)


def test_rolling_avg_sd_basic_known_values():
    t = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    window = 3
    # rolling average: [ (1+2+3)/3, (2+3+4)/3, (3+4+5)/3, (4+5+6)/3 ] = [2, 3, 4, 5]
    expected_avg = np.array([2, 3, 4, 5], dtype=float)
    # rolling std: sqrt of variance of window segments:
    # std([1,2,3]) = ~0.81649658, [2,3,4]=0.81649658, [3,4,5]=0.81649658, [4,5,6]=0.81649658
    expected_sd = np.array([0.81649658] * 4)
    avg, sd = utils.rolling_avg_sd(t, window)
    np.testing.assert_allclose(avg, expected_avg, rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(sd, expected_sd, rtol=1e-7, atol=1e-12)


def test_rolling_avg_sd_raises_value_error_if_window_too_large():
    t = np.array([1, 2], dtype=float)
    window = 3
    with pytest.raises(ValueError):
        utils.rolling_avg_sd(t, window)


def test_rolling_avg_sd_constant_series_zero_sd():
    t = np.ones(5)
    window = 3
    avg, sd = utils.rolling_avg_sd(t, window)
    expected_avg = np.ones(3)
    expected_sd = np.zeros(3)
    np.testing.assert_array_equal(avg, expected_avg)
    np.testing.assert_array_equal(sd, expected_sd)


def test_calculate_distance_profile_dimension_mismatch_raises():
    # Length mismatch between QT and mean_T, sigma_T
    QT = np.array([1, 2])
    mean_T = np.array([1])
    sigma_T = np.array([1])
    m = 2
    mean_Q = 0
    sigma_Q = 1
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)


def test_calculate_distance_profile_dimension_mismatch_arrays_raises():
    # mean_Q and sigma_Q are arrays with inconsistent lengths
    QT = np.array([1, 2, 3])
    mean_T = np.array([1, 2, 3])
    sigma_T = np.array([1, 2, 3])
    m = 3
    mean_Q = np.array([1, 2])
    sigma_Q = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)


def test_calculate_distance_profile_constant_sequences_zero_distance():
    m = 3
    length = 5
    QT = np.zeros(length)
    mean_T = np.zeros(length)
    sigma_T = np.zeros(length)
    mean_Q = 0.0
    sigma_Q = 0.0
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    # Since both Q and T subseqs std < epsilon, distances should be zero vector
    np.testing.assert_allclose(D, 0)


def test_calculate_distance_profile_constant_query_nonconstant_subseq():
    m = 3
    length = 5
    QT = np.ones(length)
    mean_T = np.array([1, 2, 3, 4, 5])
    sigma_T = np.ones(length) * 0.5  # > epsilon
    mean_Q = 0.0
    sigma_Q = 0.0
    # distances should be sqrt(m) for all because sigma_Q < epsilon but sigma_T >= epsilon
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    expected = np.full(length, np.sqrt(m))
    np.testing.assert_allclose(D, expected)


def test_calculate_distance_profile_nonconstant_query_constant_subseq():
    m = 3
    length = 5
    QT = np.ones(length)
    mean_T = np.array([1, 2, 3, 4, 5])
    sigma_T = np.zeros(length)
    mean_Q = 0.0
    sigma_Q = 1.0
    # Because sigma_T < epsilon but sigma_Q >= epsilon
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    expected = np.full(length, np.sqrt(m))
    np.testing.assert_allclose(D, expected)


def test_calculate_distance_profile_general_case_values():
    # Setup non-constant Q and T
    m = 2
    QT = np.array([3, 7, 11])
    mean_T = np.array([1, 2, 3])
    sigma_T = np.array([0.5, 0.5, 0.5])
    mean_Q = 2
    sigma_Q = 1

    # Manually computed distance profile D_i = sqrt(2 * m * (1 - (QT_i - m*mean_Q*mean_T_i)/(m*sigma_Q*sigma_T_i)))
    # = sqrt(4 * (1 - (QT_i - 2*2*mean_T_i) / (2*1* sigma_T_i))) = sqrt(4*(1-(QT_i -4*mean_T_i)/(2*sigma_T_i)))
    expected = []
    for i in range(len(QT)):
        numerator = QT[i] - m * mean_Q * mean_T[i]
        denom = m * sigma_Q * sigma_T[i]
        val = 2 * m * (1 - numerator / denom)
        val = max(val, 0)
        expected.append(np.sqrt(val))
    expected = np.array(expected)
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T, epsilon=1e-7)
    np.testing.assert_allclose(D, expected)


def test_mass_basic_distance_profile(monkeypatch):
    # Seed RNGs as precaution (mass calls sliding_dot_product, rolling_avg_sd)
    import random
    import numpy
    import torch

    random.seed(123)
    np.random.seed(123)
    try:
        import torch
        torch.manual_seed(123)
    except ImportError:
        pass

    Q = np.array([1, 2, 3], dtype=float)
    T = np.array([1, 2, 3, 4, 5], dtype=float)

    D = utils.mass(Q, T)
    expected = utils.calculate_distance_profile(
        utils.sliding_dot_product(Q, T),
        len(Q),
        np.mean(Q),
        np.std(Q),
        *utils.rolling_avg_sd(T, len(Q))
    )
    np.testing.assert_allclose(D, expected)


def test_mass_value_error_when_Q_longer_than_T():
    Q = np.array([1, 2, 3, 4])
    T = np.array([1, 2])
    with pytest.raises(ValueError):
        utils.mass(Q, T)


def test_sliding_dot_product_output_real_and_positive_length():
    Q = np.array([1, 0, -1])
    T = np.array(np.sin(np.linspace(0, 2 * np.pi, 10)))
    result = utils.sliding_dot_product(Q, T)
    assert isinstance(result, np.ndarray)
    assert np.isrealobj(result)
    assert result.shape == (len(T) - len(Q) + 1,)


def test_sliding_dot_product_against_direct_dot_product():
    Q = np.array([3, 1, 2])
    T = np.array([1, 4, 2, 8, 5], dtype=float)

    expected = np.array([np.dot(Q, T[i:i + len(Q)]) for i in range(len(T) - len(Q) + 1)])
    result = utils.sliding_dot_product(Q, T)
    np.testing.assert_allclose(result, expected)


def test_rolling_avg_sd_output_properties():
    t = np.linspace(0, 10, 100)
    window_size = 10
    avg, sd = utils.rolling_avg_sd(t, window_size)
    assert avg.shape == (len(t) - window_size + 1,)
    assert sd.shape == (len(t) - window_size + 1,)
    # standard deviation non-negative always
    assert np.all(sd >= 0)


def test_calculate_distance_profile_array_sigmaQ_sigmaMeanQ_scalar_mix():
    # mean_Q and sigma_Q scalar, mean_T and sigma_T arrays, dimensions consistent
    m = 3
    QT = np.array([1, 2, 3])
    mean_T = np.array([0, 0, 0])
    sigma_T = np.array([1, 1, 1])
    mean_Q = 0.0
    sigma_Q = 1.0
    result = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    assert result.shape == (3,)


def test_calculate_distance_profile_with_zero_sigmaQ_and_positive_sigmaT():
    m = 3
    length = 3
    QT = np.array([0, 0, 0], dtype=float)
    mean_T = np.array([1, 2, 3], dtype=float)
    sigma_T = np.array([0.1, 0.1, 0.1], dtype=float)
    mean_Q = 0.0
    sigma_Q = 0.0
    expected = np.full(length, np.sqrt(m))
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    np.testing.assert_allclose(D, expected)


def test_calculate_distance_profile_with_zero_sigmaT_and_positive_sigmaQ():
    m = 3
    length = 3
    QT = np.array([0, 0, 0], dtype=float)
    mean_T = np.array([1, 2, 3], dtype=float)
    sigma_T = np.array([0.0, 0.0, 0.0], dtype=float)
    mean_Q = 0.0
    sigma_Q = 0.5
    expected = np.full(length, np.sqrt(m))
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    np.testing.assert_allclose(D, expected)
