import re

import pytest

from tenacity import (
    RetryCallState,
    Retrying,
    retry_all,
    retry_always,
    retry_any,
    retry_if_exception,
    retry_if_exception_cause_type,
    retry_if_exception_message,
    retry_if_exception_type,
    retry_if_not_exception_message,
    retry_if_not_exception_type,
    retry_if_not_result,
    retry_if_result,
    retry_never,
    retry_unless_exception_type,
)


def make_state(exc=None, result=None, with_outcome=True):
    rs = RetryCallState(Retrying(), fn=None, args=(), kwargs={})
    if with_outcome:
        if exc is not None:
            rs.set_exception((type(exc), exc, None))
        else:
            rs.set_result(result)
    return rs


def test_retry_never_and_always_ignore_state():
    assert retry_never(make_state(exc=ValueError())) is False
    assert retry_never(make_state(result=1)) is False
    assert retry_always(make_state(exc=ValueError())) is True
    assert retry_always(make_state(result=1)) is True


def test_retry_if_exception_type_matches_types_and_ignores_results():
    default = retry_if_exception_type()
    assert default(make_state(exc=RuntimeError("x"))) is True
    assert default(make_state(result="fine")) is False

    only_value = retry_if_exception_type(ValueError)
    assert only_value(make_state(exc=ValueError())) is True
    assert only_value(make_state(exc=KeyError())) is False

    multi = retry_if_exception_type((ValueError, KeyError))
    assert multi(make_state(exc=KeyError())) is True
    assert multi(make_state(exc=TypeError())) is False
    # subclasses count as a match
    assert retry_if_exception_type(LookupError)(make_state(exc=KeyError())) is True


def test_retry_if_not_exception_type_inverts_match_for_failures_only():
    r = retry_if_not_exception_type(ValueError)
    assert r(make_state(exc=KeyError())) is True
    assert r(make_state(exc=ValueError())) is False
    assert r(make_state(result="fine")) is False


def test_retry_unless_exception_type_retries_on_success_too():
    r = retry_unless_exception_type(ValueError)
    assert r(make_state(result="fine")) is True
    assert r(make_state(exc=KeyError())) is True
    assert r(make_state(exc=ValueError())) is False
    with pytest.raises(RuntimeError):
        r(make_state(with_outcome=False))


def test_retry_if_exception_uses_predicate_and_needs_an_outcome():
    r = retry_if_exception(lambda e: "again" in str(e))
    assert r(make_state(exc=ValueError("try again"))) is True
    assert r(make_state(exc=ValueError("fatal"))) is False
    assert r(make_state(result="try again")) is False
    with pytest.raises(RuntimeError):
        r(make_state(with_outcome=False))


def test_retry_if_exception_cause_type_walks_the_cause_chain():
    try:
        try:
            try:
                raise ConnectionError("root")
            except ConnectionError as root:
                raise ValueError("middle") from root
        except ValueError as middle:
            raise KeyError("top") from middle
    except KeyError as top:
        exc = top

    assert retry_if_exception_cause_type(ConnectionError)(make_state(exc=exc)) is True
    assert retry_if_exception_cause_type(ValueError)(make_state(exc=exc)) is True
    assert retry_if_exception_cause_type(TypeError)(make_state(exc=exc)) is False
    # the exception itself is not considered, only its causes
    assert retry_if_exception_cause_type(KeyError)(make_state(exc=exc)) is False
    assert retry_if_exception_cause_type(ValueError)(make_state(result=1)) is False
    with pytest.raises(RuntimeError):
        retry_if_exception_cause_type(ValueError)(make_state(with_outcome=False))


def test_retry_if_exception_cause_type_terminates_on_cyclic_causes():
    a = KeyError("a")
    b = KeyError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert retry_if_exception_cause_type(TypeError)(make_state(exc=a)) is False
    self_cause = ValueError("self")
    self_cause.__cause__ = self_cause
    assert retry_if_exception_cause_type(TypeError)(make_state(exc=self_cause)) is False
    assert retry_if_exception_cause_type(ValueError)(make_state(exc=self_cause)) is True


def test_retry_if_result_and_not_result():
    positive = retry_if_result(lambda v: v > 0)
    assert positive(make_state(result=3)) is True
    assert positive(make_state(result=-1)) is False
    assert positive(make_state(exc=ValueError())) is False

    not_positive = retry_if_not_result(lambda v: v > 0)
    assert not_positive(make_state(result=3)) is False
    assert not_positive(make_state(result=-1)) is True
    assert not_positive(make_state(exc=ValueError())) is False
    with pytest.raises(RuntimeError):
        positive(make_state(with_outcome=False))
    with pytest.raises(RuntimeError):
        not_positive(make_state(with_outcome=False))


def test_retry_if_exception_message_exact_match():
    r = retry_if_exception_message(message="boom")
    assert r(make_state(exc=ValueError("boom"))) is True
    assert r(make_state(exc=ValueError("boom!"))) is False
    assert r(make_state(result="boom")) is False


def test_retry_if_exception_message_regex_match_is_anchored_at_start():
    r = retry_if_exception_message(match=r"err\d+")
    assert r(make_state(exc=ValueError("err42 happened"))) is True
    assert r(make_state(exc=ValueError("an err42"))) is False
    compiled = retry_if_exception_message(match=re.compile("abc", re.IGNORECASE))
    assert compiled(make_state(exc=ValueError("ABCdef"))) is True


def test_retry_if_exception_message_argument_validation():
    with pytest.raises(TypeError):
        retry_if_exception_message()
    with pytest.raises(TypeError):
        retry_if_exception_message(message="a", match="b")
    with pytest.raises(TypeError):
        retry_if_not_exception_message()


def test_retry_if_not_exception_message_retries_until_message_seen():
    r = retry_if_not_exception_message(message="stop now")
    assert r(make_state(result="fine")) is True
    assert r(make_state(exc=ValueError("other"))) is True
    assert r(make_state(exc=ValueError("stop now"))) is False
    by_regex = retry_if_not_exception_message(match="fatal.*")
    assert by_regex(make_state(exc=ValueError("fatal error"))) is False
    assert by_regex(make_state(exc=ValueError("transient"))) is True
    with pytest.raises(RuntimeError):
        r(make_state(with_outcome=False))


def test_retry_operators_build_and_flatten_combinators():
    a = retry_if_exception_type(ValueError)
    b = retry_if_exception_type(KeyError)
    c = retry_if_result(lambda v: v is None)

    either = a | b
    assert isinstance(either, retry_any)
    assert either.retries == (a, b)
    assert (a | b | c).retries == (a, b, c)

    both = a & b
    assert isinstance(both, retry_all)
    assert both.retries == (a, b)
    assert (a & b & c).retries == (a, b, c)


def test_retry_any_and_all_evaluate_members():
    always_or_never = retry_any(retry_never, retry_always)
    assert always_or_never(make_state(result=1)) is True
    assert retry_any(retry_never, retry_never)(make_state(result=1)) is False
    assert retry_any()(make_state(result=1)) is False

    assert retry_all(retry_always, retry_always)(make_state(result=1)) is True
    assert retry_all(retry_always, retry_never)(make_state(result=1)) is False
    assert retry_all()(make_state(result=1)) is True

    # plain callables can be mixed with strategy objects
    def plain(rs):
        return True

    mixed_any = retry_never | plain
    assert isinstance(mixed_any, retry_any)
    assert mixed_any(make_state(result=1)) is True
    mixed_all = retry_always & plain
    assert isinstance(mixed_all, retry_all)
    assert mixed_all(make_state(result=1)) is True
    assert (retry_always & (lambda rs: False))(make_state(result=1)) is False
