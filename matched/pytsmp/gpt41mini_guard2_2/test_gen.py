import numpy as np
import pytest
from pytsmp import utils


def test_sliding_dot_product_basic():
    Q = np.array([1, 2, 3])
    T = np.array([0, 1, 2, 3, 4])
    # Expected manual dot products:
    # subsequences of T length 3: [0,1,2], [1,2,3], [2,3,4]
    # dot products with Q: (1*0 + 2*1 + 3*2)=8,
    #                      (1*1 + 2*2 + 3*3)=14,
    #                      (1*2 + 2*3 + 3*4)=20
    result = utils.sliding_dot_product(Q, T)
    expected = np.array([8, 14, 20])
    assert np.allclose(result, expected, rtol=1e-12, atol=1e-15)


def test_sliding_dot_product_query_longer_error():
    Q = np.array([1, 2, 3, 4])
    T = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        utils.sliding_dot_product(Q, T)


def test_sliding_dot_product_length_and_type():
    Q = np.arange(5)
    T = np.arange(10)
    result = utils.sliding_dot_product(Q, T)
    # Result length should be len(T)-len(Q)+1
    assert result.shape == (len(T) - len(Q) + 1,)
    # Result dtype float
    assert np.issubdtype(result.dtype, np.floating)


@pytest.mark.parametrize("window_size", [1, 2, 5])
def test_rolling_avg_sd_basic(window_size):
    t = np.array([1, 3, 5, 7, 9, 11, 13], dtype=float)
    if window_size > len(t):
        with pytest.raises(ValueError):
            utils.rolling_avg_sd(t, window_size)
        return
    mean, sd = utils.rolling_avg_sd(t, window_size)
    # For a sliding window, check length
    expected_len = len(t) - window_size + 1
    assert mean.shape == (expected_len,)
    assert sd.shape == (expected_len,)
    # Check that mean matches np.mean on slices and sd matches np.std (ddof=0)
    for i in range(expected_len):
        win = t[i : i + window_size]
        assert np.isclose(mean[i], np.mean(win), rtol=1e-12, atol=1e-15)
        assert np.isclose(sd[i], np.std(win), rtol=1e-12, atol=1e-15)


def test_rolling_avg_sd_raises_on_shorter_window():
    t = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        utils.rolling_avg_sd(t, 5)


def test_calculate_distance_profile_shapes_and_values():
    # Generate fixed inputs
    m = 4
    QT = np.array([10.0, 20.0, 30.0])
    mean_Q = 5.0
    sigma_Q = 2.0
    mean_T = np.array([4.0, 5.0, 6.0])
    sigma_T = np.array([1.0, 2.0, 3.0])
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    assert D.shape == QT.shape
    # Distance profile should be non-negative
    assert np.all(D >= 0)
    # Check formula manually on first element
    v = (QT[0] - m * mean_Q * mean_T[0]) / (m * sigma_Q * sigma_T[0])
    distance_expected = np.sqrt(max(2 * m * (1 - v), 0))
    assert np.isclose(D[0], distance_expected, rtol=1e-12, atol=1e-15)


def test_calculate_distance_profile_input_length_mismatch():
    # QT length 3, mean_T length 2, sigma_T length 3
    QT = np.array([1, 2, 3])
    mean_T = np.array([1, 2])
    sigma_T = np.array([1, 2, 3])
    mean_Q = 0.0
    sigma_Q = 1.0
    m = 2
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    # mean_Q and sigma_Q length mismatch (arrays)
    mean_Q_arr = np.array([1, 2])
    sigma_Q_arr = np.array([1])
    mean_T = np.array([1, 2])
    sigma_T = np.array([1, 2])
    QT = np.array([1, 2])
    with pytest.raises(ValueError):
        utils.calculate_distance_profile(QT, m, mean_Q_arr, sigma_Q_arr, mean_T, sigma_T)


def test_calculate_distance_profile_constant_sigmaQ_and_sigmaT_cases():
    # Cases of sigma_Q and/or sigma_T below epsilon cause special distance values
    m = 3
    length = 4
    QT = np.ones(length) * 5
    mean_Q = 1.0
    epsilon = 1e-7

    # sigma_Q < epsilon, sigma_T < epsilon => zeros
    sigma_Q = 0.0
    sigma_T = np.zeros(length)
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, sigma_T, epsilon=epsilon)
    assert np.allclose(D, 0)

    # sigma_Q < epsilon, sigma_T >= epsilon => sqrt(m)
    sigma_Q = 0.0
    sigma_T = np.ones(length)
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, sigma_T, epsilon=epsilon)
    assert np.allclose(D, np.sqrt(m))

    # sigma_Q >= epsilon, sigma_T < epsilon => sqrt(m)
    sigma_Q = 1.0
    sigma_T = np.zeros(length)
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, sigma_T, epsilon=epsilon)
    assert np.allclose(D, np.sqrt(m))


def test_mass_distance_results_non_negative_and_correct_size():
    # fixed Q and T
    Q = np.array([1., 0., -1., 0.])
    T = np.array([0., 1., 0., -1., 0., 1., 0., -1., 0.])
    # length T=9, length Q=4, expected output length: 9-4+1=6
    D = utils.mass(Q, T)
    assert D.shape == (len(T) - len(Q) + 1,)
    assert np.all(D >= 0)


def test_mass_raises_when_Q_longer_than_T():
    Q = np.arange(5)
    T = np.arange(4)
    with pytest.raises(ValueError):
        utils.mass(Q, T)


def test_mass_values_vs_sliding_dot_and_distance_profile_seeded():
    np.random.seed(12345)
    Q = np.random.randn(10)
    T = np.random.randn(50)
    D_mass = utils.mass(Q, T)

    # Manual compute all pieces for distance profile
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, len(Q))
    mean_Q = np.mean(Q)
    sigma_Q = np.std(Q)
    D_manual = utils.calculate_distance_profile(QT, len(Q), mean_Q, sigma_Q, mean_T, sigma_T)

    assert np.allclose(D_mass, D_manual, rtol=1e-12, atol=1e-14)


def test_rolling_avg_sd_constant_series_sd_zero():
    # rolling sd should be zero for constant window slices
    t = np.ones(10)
    window_size = 5
    mean, sd = utils.rolling_avg_sd(t, window_size)
    assert np.allclose(sd, 0)
    assert np.allclose(mean, 1)


def test_sliding_dot_product_identical_series_autocorrelation():
    # Sliding dot product of Q with itself in T=Q repeated, autocorrelation at lag 0 should be highest
    Q = np.array([1, 2, 3, 4, 5], dtype=float)
    T = np.concatenate([Q, Q])
    result = utils.sliding_dot_product(Q, T)
    # length 2*len(Q) - len(Q) +1 = len(Q)+1 = 6
    assert result.shape == (len(Q) + 1,)
    # maximum dot product should be at position where Q aligns with itself exactly
    max_idx = np.argmax(result)
    assert max_idx == 0 or max_idx == len(Q)  # Exact matches at start or len(Q)
    # dot product at exact matches is sum of squares of Q
    dot_self = np.dot(Q, Q)
    assert np.isclose(result[max_idx], dot_self, rtol=1e-12)


def test_calculate_distance_profile_handles_negative_in_sqrt_correctly():
    # Compose values that cause negative inside sqrt, expect zero via maximum(.,0)
    m = 3
    length = 2
    # Set QT and means to cause negative argument inside sqrt
    QT = np.array([100.0, 100.0])
    mean_Q = 10.0
    sigma_Q = 2.0
    mean_T = np.array([10.0, 10.0])
    sigma_T = np.array([2.0, 2.0])

    # Manipulate QT to produce value inside sqrt < 0, by making (QT - m*mean_Q*mean_T)/(m*sigma_Q*sigma_T) > 1
    # For safety, directly call and check no nan and no error
    D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    assert np.all(D >= 0)
    assert not np.any(np.isnan(D))


def test_sliding_dot_product_zero_pad_behavior():
    Q = np.array([1, 1])
    T = np.array([1, 2, 3, 4, 5])
    # Check that sliding dot product and manual dot product with subsequences match ignoring tail padding effects
    result = utils.sliding_dot_product(Q, T)
    expected = np.array([3, 5, 7, 9])  # dot products: [1*1+1*2=3, 1*2+1*3=5, 1*3+1*4=7, 1*4+1*5=9]
    assert np.allclose(result, expected, rtol=1e-12, atol=1e-15)


def test_rolling_avg_sd_handles_float_and_int_inputs():
    for dtype in [float, int]:
        t = np.array([1, 2, 3, 4, 5, 6, 7], dtype=dtype)
        window_size = 3
        mean, sd = utils.rolling_avg_sd(t, window_size)
        assert mean.shape[0] == len(t) - window_size + 1
        assert sd.shape[0] == len(t) - window_size + 1
        # With float dtype results are floats
        if dtype == int:
            assert mean.dtype.kind in {'f', 'i'}
            assert sd.dtype.kind in {'f'}
        else:
            assert mean.dtype.kind == 'f'
            assert sd.dtype.kind == 'f'


def test_mass_vs_sliding_dot_and_distance_profile_with_seed2():
    # Different random seed to confirm reproducibility
    np.random.seed(2023)
    Q = np.random.randn(15)
    T = np.random.randn(100)
    D_mass = utils.mass(Q, T)

    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, len(Q))
    mean_Q = np.mean(Q)
    sigma_Q = np.std(Q)
    D_manual = utils.calculate_distance_profile(QT, len(Q), mean_Q, sigma_Q, mean_T, sigma_T)
    assert np.allclose(D_mass, D_manual, rtol=1e-12, atol=1e-14)
