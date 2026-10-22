import dataclasses
import math
from itertools import count

import pytest

from more_itertools.recipes import (
    Stats,
    convolve,
    dotproduct,
    factor,
    is_prime,
    matmul,
    multinomial,
    polynomial_derivative,
    polynomial_eval,
    polynomial_from_roots,
    random_derangement,
    reshape,
    running_max,
    running_mean,
    running_median,
    running_min,
    running_statistics,
    sieve,
    sum_of_squares,
    take,
    totient,
    transpose,
)


def test_convolve_matches_polynomial_multiplication():
    assert list(convolve([1, -1, -20], [1, -3])) == [1, -4, -17, 60]
    assert list(convolve([3, 4, 5], [1])) == [3, 4, 5]
    assert list(convolve([1, 2, 3, 4], [0.5, 0.5])) == [0.5, 1.5, 2.5, 3.5, 2.0]
    assert take(3, convolve(count(1), [1, 1])) == [1, 3, 5]


def test_polynomial_from_roots_and_eval():
    assert polynomial_from_roots([5, -4, 3]) == [1, -4, -17, 60]
    assert polynomial_from_roots([]) == [1]
    assert polynomial_from_roots([2]) == [1, -2]

    assert polynomial_eval([1, -4, -17, 60], 2.5) == 8.125
    assert polynomial_eval([1, 0, -1], 3) == 8
    assert polynomial_eval([7], 100) == 7
    assert polynomial_eval([], 2.5) == 0.0

    coeffs = polynomial_from_roots([1, 2, 3])
    for root in (1, 2, 3):
        assert polynomial_eval(coeffs, root) == 0
    assert polynomial_eval(coeffs, 4) == 6


def test_polynomial_derivative():
    assert polynomial_derivative([1, -4, -17, 60]) == [3, -8, -17]
    assert polynomial_derivative([2, 0, 0]) == [4, 0]
    assert polynomial_derivative([5]) == []
    assert polynomial_derivative([]) == []


def test_sum_of_squares_and_dotproduct():
    assert sum_of_squares([10, 20, 30]) == 1400
    assert sum_of_squares([]) == 0
    assert sum_of_squares([0.5, 1.5]) == pytest.approx(2.5)
    assert dotproduct([1, 2, 3], [4, 5, 6]) == 32
    assert dotproduct([10, 15, 12], [0.65, 0.80, 1.25]) == pytest.approx(33.5)
    assert dotproduct([], []) == 0


def test_sieve_generates_primes_below_n():
    assert list(sieve(30)) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    assert list(sieve(2)) == []
    assert list(sieve(3)) == [2]
    assert len(list(sieve(1000))) == 168
    assert list(sieve(100))[-1] == 97
    assert list(sieve(200)) == [n for n in range(200) if is_prime(n)]


def test_factor_returns_sorted_prime_factors():
    assert list(factor(360)) == [2, 2, 2, 3, 3, 5]
    assert list(factor(1)) == []
    assert list(factor(0)) == []
    assert list(factor(97)) == [97]
    assert list(factor(2**10)) == [2] * 10
    assert list(factor(10007 * 10009)) == [10007, 10009]
    assert list(factor(2**31 - 1)) == [2**31 - 1]
    assert list(factor(600851475143)) == [71, 839, 1471, 6857]
    for n in (12, 255, 1001, 65536, 999983 * 2):
        assert math.prod(factor(n)) == n


def test_totient_matches_bruteforce_definition():
    assert totient(9) == 6
    assert totient(1) == 1
    assert totient(10) == 4
    assert totient(13) == 12
    for n in range(1, 60):
        expected = sum(1 for k in range(1, n + 1) if math.gcd(n, k) == 1)
        assert totient(n) == expected


def test_is_prime_small_large_and_pseudoprimes():
    assert [n for n in range(30) if is_prime(n)] == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    assert not is_prime(-7)
    assert not is_prime(0)
    assert not is_prime(1)
    assert is_prime(1_000_000_007)
    assert is_prime(18_446_744_073_709_551_557)
    assert is_prime(2**61 - 1)
    assert not is_prime(2**64 + 1)
    for carmichael in (561, 41041, 3215031751):
        assert not is_prime(carmichael)
    primes_below_500 = set(sieve(500))
    assert all(is_prime(n) == (n in primes_below_500) for n in range(500))


def test_multinomial_counts_arrangements():
    assert multinomial(3, 4, 2) == 1260
    assert multinomial(5, 2, 2, 1, 1) == 83160
    assert multinomial(5, 7) == math.comb(12, 5)
    assert multinomial(1, 1, 1, 1, 1, 1, 1) == math.factorial(7)
    assert multinomial() == 1


def test_transpose_and_matmul():
    assert list(transpose([(1, 2, 3), (11, 22, 33)])) == [(1, 11), (2, 22), (3, 33)]
    assert list(transpose([])) == []
    with pytest.raises(ValueError):
        list(transpose([(1, 2), (3,)]))

    assert list(matmul([(7, 5), (3, 5)], [(2, 5), (7, 9)])) == [(49, 80), (41, 60)]
    identity = [(1, 0), (0, 1)]
    other = [(5, 6), (7, 8)]
    assert list(matmul(identity, other)) == other
    assert list(matmul([(1, 2, 3)], [(1,), (2,), (3,)])) == [(14,)]


def test_reshape_with_int_and_tuple_shapes():
    matrix = [(0, 1), (2, 3), (4, 5)]
    assert list(reshape(matrix, 3)) == [(0, 1, 2), (3, 4, 5)]
    assert list(reshape(matrix, (2, 3))) == [(0, 1, 2), (3, 4, 5)]
    assert list(reshape(matrix, (6,))) == [0, 1, 2, 3, 4, 5]
    assert list(reshape(matrix, (2, 1, 3, 1))) == [
        (((0,), (1,), (2,)),),
        (((3,), (4,), (5,)),),
    ]
    tensor = [[(1, 2), (3, 4)], [(5, 6), (7, 8)]]
    assert list(reshape(tensor, (8,))) == [1, 2, 3, 4, 5, 6, 7, 8]


def test_running_median_and_running_mean():
    data = [5.0, 9.0, 4.0, 12.0, 8.0, 9.0]
    assert list(running_median(data)) == [5.0, 7.0, 5.0, 7.0, 8.0, 8.5]
    assert list(running_median(data, maxlen=3)) == [5.0, 7.0, 5.0, 9.0, 8.0, 9.0]
    assert list(running_median([3, 1, 2], maxlen=1)) == [3, 1, 2]
    assert list(running_median([])) == []

    assert list(running_mean([40, 30, 50, 46, 39, 44])) == [
        40.0,
        35.0,
        40.0,
        41.5,
        41.0,
        41.5,
    ]
    assert list(running_mean([40, 30, 50, 46, 39, 44], maxlen=3)) == [
        40.0,
        35.0,
        40.0,
        42.0,
        45.0,
        43.0,
    ]
    assert list(running_mean([])) == []
    for func in (running_median, running_mean):
        with pytest.raises(ValueError):
            func([1, 2, 3], maxlen=0)
        with pytest.raises(ValueError):
            func([1, 2, 3], maxlen=-2)


def test_running_min_and_running_max():
    data = [4, 3, 7, 0, 8, 1, 6, 2, 9, 5]
    assert list(running_min(data)) == [4, 3, 3, 0, 0, 0, 0, 0, 0, 0]
    assert list(running_min(data, maxlen=3)) == [4, 3, 3, 0, 0, 0, 1, 1, 2, 2]
    assert list(running_max(data)) == [4, 4, 7, 7, 8, 8, 8, 8, 9, 9]
    assert list(running_max(data, maxlen=3)) == [4, 4, 7, 7, 8, 8, 8, 6, 9, 9]
    assert list(running_min(data, maxlen=1)) == data
    assert list(running_max(data, maxlen=1)) == data
    assert list(running_min([])) == []
    for func in (running_min, running_max):
        with pytest.raises(ValueError):
            func(data, maxlen=0)


def test_running_statistics_and_random_derangement():
    stats = list(running_statistics([4, 1, 3]))
    assert [s.size for s in stats] == [1, 2, 3]
    assert [s.minimum for s in stats] == [4, 1, 1]
    assert [s.maximum for s in stats] == [4, 4, 4]
    assert [s.median for s in stats] == [4, 2.5, 3]
    assert stats[-1].mean == pytest.approx(8 / 3)
    assert isinstance(stats[0], Stats)

    windowed = list(running_statistics([4, 1, 3], maxlen=2))
    assert [s.size for s in windowed] == [1, 2, 2]
    assert windowed[-1].minimum == 1
    assert windowed[-1].maximum == 3
    assert windowed[-1].median == 2.0
    assert windowed[-1].mean == 2.0

    with pytest.raises(dataclasses.FrozenInstanceError):
        stats[0].size = 99

    for _ in range(25):
        result = random_derangement(range(6))
        assert sorted(result) == list(range(6))
        assert all(item != index for index, item in enumerate(result))
    assert random_derangement([]) == ()
    assert random_derangement('ab') == ('b', 'a')
    with pytest.raises(IndexError):
        random_derangement([1])
