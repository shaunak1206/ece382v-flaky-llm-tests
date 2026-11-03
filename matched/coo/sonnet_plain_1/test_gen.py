import asyncio
import importlib
from pathlib import Path

import pytest

from coo import Coo
from coo._exceptions import ScheduleError, TemplateError, TweetTypeError

coo_module = importlib.import_module("coo.coo")


class FakeApi:
    def __init__(self):
        self.calls = []

    def PostUpdate(self, update, media=None):
        self.calls.append((update, media))
        return update


@pytest.fixture(autouse=True)
def reset_class_state(monkeypatch):
    monkeypatch.setattr(Coo, "time_zone", "local")
    monkeypatch.setattr(Coo, "media", None)
    monkeypatch.setattr(Coo, "global_media", None)
    monkeypatch.setattr(Coo, "global_template", None)


@pytest.fixture
def make_coo():
    loops = []

    def _make(preview=True):
        c = Coo("ck", "cs", "tok", "ts", preview=preview)
        c.loop = asyncio.new_event_loop()
        loops.append(c.loop)
        return c

    yield _make
    for loop in loops:
        if not loop.is_closed():
            loop.close()


def test_init_stores_credentials(make_coo):
    c = make_coo(preview=False)
    assert c.consumer == "ck"
    assert c.consumer_secret == "cs"
    assert c.token == "tok"
    assert c.token_secret == "ts"
    assert c.preview is False


def test_init_preview_flag_and_switches(make_coo):
    c = make_coo(preview=True)
    assert c.preview is True
    assert c._delay_time is True
    assert c._interval_time is False


@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_init_rejects_non_string_credentials(position):
    args = ["a", "b", "c", "d"]
    args[position] = 123
    with pytest.raises(TypeError):
        Coo(*args)


def test_exception_classes_hierarchy_and_messages():
    assert issubclass(TweetTypeError, TypeError)
    assert issubclass(ScheduleError, TypeError)
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."


def test_tweet_non_list_raises_tweet_type_error(make_coo):
    c = make_coo()
    with pytest.raises(TweetTypeError) as exc:
        c.tweet("just a string")
    assert str(exc.value) == TweetTypeError.wrongListMsg


def test_tweet_list_of_non_strings_raises(make_coo):
    c = make_coo()
    with pytest.raises(TweetTypeError):
        c.tweet([1, 2, 3])


def test_tweet_empty_list_raises_index_error(make_coo):
    c = make_coo()
    with pytest.raises(IndexError):
        c.tweet([])


def test_tweet_preview_prints_and_returns_updates(make_coo, capsys):
    c = make_coo()
    result = c.tweet(["one", "two"])
    assert result == ["one", "two"]
    assert capsys.readouterr().out.splitlines() == ["one", "two"]


def test_tweet_applies_template(make_coo, capsys):
    c = make_coo()
    c.tweet(["hello"], template="Say: $message!")
    assert capsys.readouterr().out.strip() == "Say: hello!"


def test_tweet_template_without_message_raises(make_coo):
    c = make_coo()
    with pytest.raises(TemplateError):
        c.tweet(["hello"], template="no placeholder")


def test_tweet_aleatory_shuffles_updates(make_coo, capsys, monkeypatch):
    c = make_coo()
    monkeypatch.setattr(coo_module.random, "shuffle", lambda seq: seq.reverse())
    result = c.tweet(["a", "b", "c"], aleatory=True)
    assert result == ["c", "b", "a"]
    assert capsys.readouterr().out.splitlines() == ["c", "b", "a"]


def test_tweet_sets_time_zone_and_media(make_coo):
    c = make_coo()
    c.tweet(["x"], time_zone="UTC", media="pic.png")
    assert Coo.time_zone == "UTC"
    assert Coo.media == Path("pic.png")


def test_tweet_delay_and_interval_calls_zzz(make_coo, monkeypatch):
    c = make_coo()
    calls = []
    monkeypatch.setattr(coo_module, "zzz", lambda *a, **k: calls.append(a))
    c.tweet(["a", "b", "c"], delay=5, interval=2)
    # one delay call, then an interval before the 2nd and 3rd updates only
    assert calls[0][0] == 5
    assert [call[0] for call in calls[1:]] == [2, 2]
    assert c._delay_time is False
    assert c._interval_time is True


def test_class_setters():
    Coo.set_time_zone("Europe/Paris")
    Coo.set_media_file("m.png")
    Coo.set_global_media_file("g.png")
    Coo.set_global_template("T $message")
    assert Coo.time_zone == "Europe/Paris"
    assert Coo.media == "m.png"
    assert Coo.global_media == "g.png"
    assert Coo.global_template == "T $message"


def test_schedule_requires_list_of_tuples(make_coo):
    c = make_coo()
    with pytest.raises(ScheduleError) as exc:
        c.schedule(["not a tuple"])
    assert str(exc.value) == ScheduleError.wrongListMsg


@pytest.mark.parametrize("bad", [("now", "msg"), ("now", None, "m", None, "extra")])
def test_schedule_bad_tuple_length_raises(make_coo, bad):
    c = make_coo()
    with pytest.raises(ScheduleError) as exc:
        c.schedule([bad])
    assert str(exc.value) == ScheduleError.tupleLenError


def test_schedule_posts_updates_with_global_template(make_coo, capsys):
    c = make_coo()
    c.schedule([("now", None, "hi")], template="Hi $message")
    assert capsys.readouterr().out.strip() == "Hi hi"
    assert Coo.global_template == "Hi $message"
    assert c.loop.is_closed()


def test_str_update_preview_prints_and_returns_none(make_coo, capsys):
    c = make_coo(preview=True)
    assert c._str_update("hello", None) is None
    assert capsys.readouterr().out.strip() == "hello"


def test_str_update_posts_without_media(make_coo, monkeypatch):
    c = make_coo(preview=False)
    fake = FakeApi()
    monkeypatch.setattr(Coo, "api", property(lambda self: fake))
    result = c._str_update("hello", "[$message]")
    assert result == "[hello]"
    assert fake.calls == [("[hello]", None)]


def test_str_update_posts_with_media_file(make_coo, monkeypatch, tmp_path):
    c = make_coo(preview=False)
    fake = FakeApi()
    monkeypatch.setattr(Coo, "api", property(lambda self: fake))
    media = tmp_path / "pic.bin"
    media.write_bytes(b"data")
    Coo.set_media_file(media)
    c._str_update("with pic", None)
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == "with pic"
    assert fake.calls[0][1] is not None


def test_str_update_uses_global_template(make_coo, capsys):
    c = make_coo(preview=True)
    Coo.set_global_template("G: $message")
    c._str_update("x", None)
    assert capsys.readouterr().out.strip() == "G: x"


def test_str_returns_user_name(make_coo, monkeypatch):
    c = make_coo()

    class User:
        name = "alice"

    class Api:
        def VerifyCredentials(self):
            return User()

    monkeypatch.setattr(Coo, "api", property(lambda self: Api()))
    assert str(c) == "Twitter User: alice."
