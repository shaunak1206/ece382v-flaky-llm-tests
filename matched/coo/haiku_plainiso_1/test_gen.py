import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import twitter

import coo.coo as coo_mod
from coo import Coo
from coo._exceptions import ScheduleError, TemplateError, TweetTypeError


def _make(preview=True):
    return Coo("consumer", "consumer_secret", "token", "token_secret", preview=preview)


class _FakeApi:
    def __init__(self):
        self.posts = []

    def PostUpdate(self, update, media=None):
        self.posts.append((update, None if media is None else str(media.name)))
        return update

    def VerifyCredentials(self):
        return SimpleNamespace(name="alice")


def test_exception_classes_subclass_type_error():
    assert issubclass(TweetTypeError, TypeError)
    assert issubclass(ScheduleError, TypeError)
    assert issubclass(TemplateError, TypeError)


def test_exception_message_constants_and_str():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."
    assert str(TweetTypeError(TweetTypeError.wrongListMsg)) == (
        "A List of Strings is required."
    )
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."
    assert TemplateError.templateMsgErr == "The template must contain a '$message'."


def test_constructor_stores_credentials_and_defaults():
    c = Coo("ck", "cs", "tk", "ts", preview=True)
    assert c.consumer == "ck"
    assert c.consumer_secret == "cs"
    assert c.token == "tk"
    assert c.token_secret == "ts"
    assert c.preview is True
    assert c._delay_time is True
    assert c._interval_time is False


def test_constructor_rejects_non_string_credentials():
    bad_values = [None, 1, b"bytes"]
    for bad in bad_values:
        with pytest.raises(TypeError, match="must be strings"):
            Coo(bad, "cs", "tk", "ts")
        with pytest.raises(TypeError, match="must be strings"):
            Coo("ck", "cs", bad, "ts")


def test_class_level_defaults():
    assert Coo.time_zone == "local"
    assert Coo.media is None
    assert Coo.global_media is None
    assert Coo.global_template is None


def test_api_property_returns_twitter_api():
    c = _make()
    assert isinstance(c.api, twitter.Api)


def test_tweet_rejects_non_list_updates():
    c = _make()
    with pytest.raises(TweetTypeError, match="A List of Strings is required."):
        c.tweet("hello")
    with pytest.raises(TweetTypeError):
        c.tweet(("a", "b"))


def test_tweet_rejects_list_of_non_strings():
    c = _make()
    with pytest.raises(TweetTypeError):
        c.tweet([1, 2])


def test_tweet_preview_prints_and_returns_updates(capsys):
    c = _make()
    result = c.tweet(["one", "two"])
    assert result == ["one", "two"]
    assert capsys.readouterr().out == "one\ntwo\n"


def test_tweet_applies_template_in_preview(capsys):
    c = _make()
    result = c.tweet(["hi"], template="Hello $message!")
    assert result == ["hi"]
    assert capsys.readouterr().out == "Hello hi!\n"


def test_tweet_template_without_message_placeholder_raises():
    c = _make()
    with pytest.raises(TemplateError, match="must contain"):
        c.tweet(["a"], template="no placeholder here")


def test_tweet_aleatory_uses_shuffle(monkeypatch, capsys):
    monkeypatch.setattr(coo_mod.random, "shuffle", lambda seq: seq.reverse())
    c = _make()
    result = c.tweet(["a", "b", "c"], aleatory=True)
    assert result == ["c", "b", "a"]
    assert capsys.readouterr().out == "c\nb\na\n"


def test_tweet_media_path_is_stored_as_path(monkeypatch):
    monkeypatch.setattr(Coo, "media", None)
    c = _make()
    c.tweet(["a"], media="pic.png")
    assert Coo.media == Path("pic.png")


def test_tweet_time_zone_argument_updates_class_setting(monkeypatch):
    monkeypatch.setattr(Coo, "time_zone", "local")
    c = _make()
    c.tweet(["a"], time_zone="UTC")
    assert Coo.time_zone == "UTC"


def test_tweet_posts_text_only_through_api(monkeypatch):
    monkeypatch.setattr(Coo, "media", None)
    fake = _FakeApi()
    monkeypatch.setattr(Coo, "api", property(lambda self: fake))
    c = _make(preview=False)
    c.tweet(["hello", "world"])
    assert fake.posts == [("hello", None), ("world", None)]


def test_tweet_posts_with_media_file(monkeypatch, tmp_path):
    monkeypatch.setattr(Coo, "media", None)
    media_file = tmp_path / "pic.txt"
    media_file.write_text("x")
    fake = _FakeApi()
    monkeypatch.setattr(Coo, "api", property(lambda self: fake))
    c = _make(preview=False)
    c.tweet(["with media"], media=str(media_file))
    assert len(fake.posts) == 1
    assert fake.posts[0][0] == "with media"
    assert fake.posts[0][1] == str(media_file)


def test_delay_sleeps_once_then_is_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(coo_mod, "zzz", lambda *args: calls.append(args))
    c = _make()
    c._delay(3)
    c._delay(3)
    assert calls == [(3, "local")]
    assert c._delay_time is False


def test_interval_skips_first_update_then_sleeps(monkeypatch):
    calls = []
    monkeypatch.setattr(coo_mod, "zzz", lambda *args: calls.append(args))
    c = _make()
    c._interval(5)
    assert calls == []
    c._interval(5)
    assert calls == [(5,)]
    c._interval(None)
    assert calls == [(5,)]


def test_schedule_rejects_non_tuple_updates():
    c = _make()
    with pytest.raises(ScheduleError, match="A list of tuples is required."):
        c.schedule(["2040-10-30 00:05"])


def test_async_tasks_rejects_wrong_tuple_length():
    c = _make()
    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(ScheduleError, match="Every tuple need a length of 3 or 4."):
            loop.run_until_complete(c._async_tasks([("now", None, "x", None, "extra")]))
        with pytest.raises(ScheduleError):
            loop.run_until_complete(c._async_tasks([("now",)]))
    finally:
        loop.close()


def test_verify_and_str_use_api_name(monkeypatch):
    fake = _FakeApi()
    monkeypatch.setattr(Coo, "api", property(lambda self: fake))
    c = _make()
    assert c.verify.name == "alice"
    assert str(c) == "Twitter User: alice."


def test_global_template_and_media_classmethods(monkeypatch, capsys):
    monkeypatch.setattr(Coo, "global_template", None)
    monkeypatch.setattr(Coo, "global_media", None)
    Coo.set_global_template("<$message>")
    Coo.set_global_media_file("banner.png")
    assert Coo.global_template == "<$message>"
    assert Coo.global_media == "banner.png"
    c = _make()
    c._str_update("hi", None)
    assert capsys.readouterr().out == "<hi>\n"


def test_set_time_zone_classmethod_changes_default(monkeypatch):
    monkeypatch.setattr(Coo, "time_zone", "local")
    Coo.set_time_zone("America/Chicago")
    assert Coo.time_zone == "America/Chicago"
    assert _make().time_zone == "America/Chicago"
