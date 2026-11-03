import importlib
from pathlib import Path

import pytest

from coo import Coo
from coo._exceptions import ScheduleError, TemplateError, TweetTypeError

coo_module = importlib.import_module("coo.coo")


@pytest.fixture(autouse=True)
def restore_class_state():
    saved = (Coo.time_zone, Coo.media, Coo.global_media, Coo.global_template)
    yield
    Coo.time_zone, Coo.media, Coo.global_media, Coo.global_template = saved


@pytest.fixture
def bot():
    return Coo("ck", "cs", "tok", "toksec", preview=True)


class FakeApi:
    def __init__(self):
        self.posted = []

    def PostUpdate(self, update, media=None):
        self.posted.append((update, media))
        return update


def test_exception_classes_are_type_errors():
    assert issubclass(TweetTypeError, TypeError)
    assert issubclass(ScheduleError, TypeError)


def test_tweet_type_error_message():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."
    with pytest.raises(TweetTypeError, match="List of Strings"):
        raise TweetTypeError(TweetTypeError.wrongListMsg)


def test_schedule_error_messages():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."


def test_init_stores_credentials(bot):
    assert bot.consumer == "ck"
    assert bot.consumer_secret == "cs"
    assert bot.token == "tok"
    assert bot.token_secret == "toksec"
    assert bot.preview is True


def test_init_preview_defaults_to_false():
    assert Coo("a", "b", "c", "d").preview is False


@pytest.mark.parametrize(
    "args",
    [
        (1, "b", "c", "d"),
        ("a", None, "c", "d"),
        ("a", "b", b"c", "d"),
        ("a", "b", "c", 4.5),
    ],
)
def test_init_rejects_non_string_credentials(args):
    with pytest.raises(TypeError, match="credentials must be strings"):
        Coo(*args)


def test_init_switch_defaults(bot):
    assert bot._delay_time is True
    assert bot._interval_time is False
    assert bot.loop is not None


def test_class_setters_update_class_attributes():
    Coo.set_time_zone("UTC")
    Coo.set_media_file("m.png")
    Coo.set_global_media_file("g.png")
    Coo.set_global_template("$message!")
    assert Coo.time_zone == "UTC"
    assert Coo.media == "m.png"
    assert Coo.global_media == "g.png"
    assert Coo.global_template == "$message!"


def test_tweet_non_list_raises(bot):
    with pytest.raises(TweetTypeError):
        bot.tweet("just a string")


def test_tweet_non_string_elements_raise(bot):
    with pytest.raises(TweetTypeError):
        bot.tweet([1, 2, 3])


def test_tweet_empty_list_raises(bot):
    with pytest.raises((TweetTypeError, IndexError)):
        bot.tweet([])


def test_tweet_preview_prints_updates_and_returns_them(bot, capsys):
    updates = ["first", "second"]
    result = bot.tweet(updates)
    assert result == ["first", "second"]
    out = capsys.readouterr().out.splitlines()
    assert out == ["first", "second"]


def test_tweet_with_template_formats_messages(bot, capsys):
    bot.tweet(["hello", "world"], template="Say: $message")
    out = capsys.readouterr().out.splitlines()
    assert out == ["Say: hello", "Say: world"]


def test_tweet_template_without_message_raises(bot):
    with pytest.raises(TemplateError):
        bot.tweet(["hello"], template="no placeholder here")


def test_tweet_aleatory_keeps_all_updates(bot, capsys):
    updates = ["a", "b", "c", "d", "e"]
    result = bot.tweet(updates, aleatory=True)
    assert sorted(result) == ["a", "b", "c", "d", "e"]
    out = capsys.readouterr().out.splitlines()
    assert sorted(out) == ["a", "b", "c", "d", "e"]


def test_tweet_sets_time_zone_and_media(bot):
    bot.tweet(["x"], time_zone="UTC", media="pic.png")
    assert Coo.time_zone == "UTC"
    assert Coo.media == Path("pic.png")


def test_schedule_rejects_non_tuple_updates(bot):
    with pytest.raises(ScheduleError, match="list of tuples"):
        bot.schedule(["not a tuple"])


def test_async_tasks_rejects_bad_tuple_length(bot):
    with pytest.raises(ScheduleError, match="length of 3 or 4"):
        bot.loop.run_until_complete(bot._async_tasks([("now", "only two")]))
    with pytest.raises(ScheduleError):
        bot.loop.run_until_complete(bot._async_tasks([("now", None, "a", None, 5)]))


def test_str_update_uses_global_template(bot, capsys):
    Coo.set_global_template("[G] $message")
    bot._str_update("hi", None)
    assert capsys.readouterr().out.strip() == "[G] hi"


def test_str_update_explicit_template_overrides_global(bot, capsys):
    Coo.set_global_template("[G] $message")
    bot._str_update("hi", "[L] $message")
    assert capsys.readouterr().out.strip() == "[L] hi"


def test_str_update_posts_without_media_when_not_previewing(monkeypatch):
    fake = FakeApi()
    monkeypatch.setattr(Coo, "api", property(lambda self: fake))
    Coo.set_media_file(None)
    bot = Coo("a", "b", "c", "d", preview=False)
    result = bot._str_update("posted", None)
    assert result == "posted"
    assert fake.posted == [("posted", None)]


def test_str_update_posts_with_media_file(monkeypatch, tmp_path):
    fake = FakeApi()
    monkeypatch.setattr(Coo, "api", property(lambda self: fake))
    media = tmp_path / "img.bin"
    media.write_bytes(b"data")
    Coo.set_media_file(media)
    bot = Coo("a", "b", "c", "d", preview=False)
    bot._str_update("with media", None)
    assert len(fake.posted) == 1
    assert fake.posted[0][0] == "with media"
    assert fake.posted[0][1] is not None


def test_api_property_builds_twitter_api_with_credentials(monkeypatch, bot):
    captured = {}

    def fake_api(*args):
        captured["args"] = args
        return "api-object"

    monkeypatch.setattr(coo_module.twitter, "Api", fake_api)
    assert bot.api == "api-object"
    assert captured["args"] == ("ck", "cs", "tok", "toksec")


def test_delay_sleeps_only_once(monkeypatch, bot):
    calls = []
    monkeypatch.setattr(coo_module, "zzz", lambda *a: calls.append(a))
    bot._delay(5)
    bot._delay(5)
    assert len(calls) == 1
    assert calls[0][0] == 5
    assert bot._delay_time is False


def test_interval_skips_first_call_then_sleeps(monkeypatch, bot):
    calls = []
    monkeypatch.setattr(coo_module, "zzz", lambda *a: calls.append(a))
    bot._interval(3)
    assert calls == []
    assert bot._interval_time is True
    bot._interval(3)
    bot._interval(3)
    assert len(calls) == 2
