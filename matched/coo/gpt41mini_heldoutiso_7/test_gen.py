import pytest

from coo import Coo
from coo._exceptions import TweetTypeError, ScheduleError


def test_coo_init_empty():
    c = Coo()
    assert isinstance(c, Coo)
    # Default attributes
    assert hasattr(c, "msg_list") or hasattr(c, "schedule")


def test_coo_msg_accepts_list_of_strings():
    msgs = ["hello", "world"]
    c = Coo(msg=msgs)
    # Assuming Coo stores the list or similar
    assert hasattr(c, "msg_list") or hasattr(c, "msg")
    # If attributes are not defined, just instantiate without error


def test_coo_msg_rejects_non_list_raises_TweetTypeError():
    with pytest.raises(TweetTypeError):
        Coo(msg="not a list")


def test_coo_msg_rejects_list_with_non_strings_raises_TweetTypeError():
    with pytest.raises(TweetTypeError):
        Coo(msg=["valid", 123])


def test_coo_schedule_accepts_valid_schedule():
    from datetime import datetime, timedelta

    future = datetime.now() + timedelta(days=1)
    schedule = [(future, 10, "tweet")]
    c = Coo(schedule=schedule)
    assert hasattr(c, "schedule")
    # Again only checking no error


def test_coo_schedule_rejects_non_list_raises_ScheduleError():
    with pytest.raises(ScheduleError):
        Coo(schedule="not a list")


def test_coo_schedule_rejects_list_with_invalid_tuple_length():
    from datetime import datetime, timedelta

    future = datetime.now() + timedelta(days=1)
    schedule = [(future, 10)]  # tuple length 2, invalid
    with pytest.raises(ScheduleError):
        Coo(schedule=schedule)


def test_coo_schedule_rejects_past_date():
    from datetime import datetime, timedelta

    past = datetime.now() - timedelta(days=1)
    schedule = [(past, 10, "tweet")]
    with pytest.raises(ScheduleError):
        Coo(schedule=schedule)


def test_tweet_type_error_message():
    err = TweetTypeError()
    assert err.wrongListMsg == "A List of Strings is required."
    assert isinstance(err, TypeError)


def test_schedule_error_messages():
    err = ScheduleError()
    assert err.wrongListMsg == "A list of tuples is required."
    assert err.tupleLenError == "Every tuple need a length of 3 or 4."
    assert err.pastDateError == "A future date is needed."
    assert isinstance(err, TypeError)


def test_schedule_error_is_subclass_of_type_error():
    assert issubclass(ScheduleError, TypeError)


def test_tweet_type_error_is_subclass_of_type_error():
    assert issubclass(TweetTypeError, TypeError)


def test_coo_author_metadata():
    import coo
    assert coo.__author__ == "Carlos Montecinos Geisse"
    assert coo.__license__ == "Apache 2.0"
    assert coo.__version__ == "0.1.3"
    assert coo.__contact__ == "carlos.w.montecinos@gmail.com"
    assert coo.__url__ == "https://github.com/wilfredinni/coo"


# Below tests are conditional on what Coo implementation actually supports.
# If Coo has no constructor arguments or attributes these can fail.
@pytest.mark.parametrize("invalid_msg", ["string", 123, None, {1: 2}, [1, 2, 3]])
def test_coo_rejects_invalid_msg_types(invalid_msg):
    # List of strings expected, these are invalid
    from coo._exceptions import TweetTypeError

    # Only cases that are list of non-strings are handled in other test
    if isinstance(invalid_msg, list) and all(isinstance(i, str) for i in invalid_msg):
        # skip valid cases
        pytest.skip("valid input")
    else:
        with pytest.raises(TweetTypeError):
            Coo(msg=invalid_msg)


@pytest.mark.parametrize("invalid_schedule", [123, "string", None, [(1, 2)], [(1, 2, 3, 4, 5)]])
def test_coo_rejects_invalid_schedule_types(invalid_schedule):
    with pytest.raises(ScheduleError):
        Coo(schedule=invalid_schedule)
