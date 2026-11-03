import asyncio
import importlib
import types
from pathlib import Path

import pytest

from coo import Coo
from coo._exceptions import ScheduleError, TemplateError, TweetTypeError


coo_module = importlib.import_module("coo.coo")


class FakeTwitterApi:
    """Stands in for twitter.Api so no network traffic happens."""

    created = []
    posts = []

    def __init__(self, consumer, consumer_secret, token, token_secret):
        self.credentials = (consumer, consumer_secret, token, token_secret)
        FakeTwitterApi.created.append(self)

    def PostUpdate(self, text, **kwargs):
        media = kwargs.get("media")
        media_bytes = media.read() if media is not None else None
        self.posts.append({"text": text, "kwargs": set(kwargs), "media": media_bytes})
        return {"text": text}

    def VerifyCredentials(self):
        return types.SimpleNamespace(name="alice")


@pytest.fixture(autouse=True)
def _reset_coo_class_state(monkeypatch):
    monkeypatch.setattr(Coo, "time_zone", "local")
    monkeypatch.setattr(Coo, "media", None)
    monkeypatch.setattr(Coo, "global_media", None)
    monkeypatch.setattr(Coo, "global_template", None)


@pytest.fixture(autouse=True)
def fresh_event_loop():
    # Coo.schedule closes its loop, so every test gets a new one.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    if not loop.is_closed():
        loop.close()


@pytest.fixture
def fake_twitter(monkeypatch):
    FakeTwitterApi.created = []
    FakeTwitterApi.posts = []
    monkeypatch.setattr(coo_module.twitter, "Api", FakeTwitterApi)
    return FakeTwitterApi


@pytest.fixture
def bot(fake_twitter, fresh_event_loop):
    return Coo("ck", "cs", "tk", "ts")


@pytest.fixture
def preview_bot(fake_twitter, fresh_event_loop):
    return Coo("ck", "cs", "tk", "ts", preview=True)


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        coo_module,
        "zzz",
        lambda seconds, time_zone=None: recorded.append((seconds, time_zone)),
    )
    return recorded


def test_constructor_stores_credentials_and_preview_flag(fake_twitter):
    bot = Coo("ck", "cs", "tk", "ts", preview=True)

    assert (bot.consumer, bot.consumer_secret, bot.token, bot.token_secret) == (
        "ck",
        "cs",
        "tk",
        "ts",
    )
    assert bot.preview is True
    assert fake_twitter.created == []


def test_constructor_rejects_non_string_credentials(fake_twitter):
    with pytest.raises(TypeError, match="must be strings"):
        Coo("ck", 2, "tk", "ts")


def test_api_property_builds_twitter_api_from_credentials(bot, fake_twitter):
    api = bot.api

    assert isinstance(api, FakeTwitterApi)
    assert api.credentials == ("ck", "cs", "tk", "ts")


def test_str_uses_verified_user_name(bot, fake_twitter):
    assert str(bot) == "Twitter User: alice."


@pytest.mark.parametrize("bad", ["hello", ("a", "b"), None, 42])
def test_tweet_rejects_non_list_updates(preview_bot, bad):
    with pytest.raises(TweetTypeError) as info:
        preview_bot.tweet(bad)

    assert str(info.value) == TweetTypeError.wrongListMsg


@pytest.mark.parametrize("bad", [[1, 2], [None], [("a",)]])
def test_tweet_rejects_lists_whose_first_item_is_not_str(preview_bot, bad):
    with pytest.raises(TweetTypeError):
        preview_bot.tweet(bad)


def test_custom_errors_are_type_errors_with_distinct_classes():
    assert issubclass(TweetTypeError, TypeError)
    assert issubclass(ScheduleError, TypeError)
    assert issubclass(TemplateError, TypeError)
    assert not issubclass(ScheduleError, TweetTypeError)
    assert not issubclass(TweetTypeError, ScheduleError)


def test_error_message_constants_match_documented_text():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."


def test_tweet_preview_prints_each_update_and_returns_same_list(preview_bot, fake_twitter, capsys):
    updates = ["one", "two"]

    result = preview_bot.tweet(updates)

    assert result is updates
    assert capsys.readouterr().out == "one\ntwo\n"
    assert fake_twitter.created == []


def test_tweet_preview_applies_template(preview_bot, capsys):
    preview_bot.tweet(["hi"], template="New: $message")

    assert capsys.readouterr().out == "New: hi\n"


def test_tweet_template_without_placeholder_raises_template_error(bot, fake_twitter):
    with pytest.raises(TemplateError):
        bot.tweet(["hi"], template="no placeholder here")

    assert fake_twitter.posts == []


def test_global_template_applies_unless_overridden(preview_bot, capsys):
    preview_bot.set_global_template("[$message]")

    preview_bot.tweet(["x"])
    preview_bot.tweet(["y"], template="<$message>")

    assert capsys.readouterr().out == "[x]\n<y>\n"


def test_delay_sleeps_only_before_first_batch(preview_bot, sleeps, capsys):
    preview_bot.tweet(["a", "b"], delay=5)
    preview_bot.tweet(["c"], delay=5)

    assert sleeps == [(5, "local")]


def test_interval_sleeps_between_updates_only(preview_bot, sleeps, capsys):
    preview_bot.tweet(["a", "b", "c"], interval=2)

    assert [seconds for seconds, _ in sleeps] == [2, 2]


def test_tweet_time_zone_argument_sets_class_time_zone(preview_bot, capsys):
    preview_bot.tweet(["a"], time_zone="America/Chicago")

    assert Coo.time_zone == "America/Chicago"


def test_tweet_media_path_is_stored_as_path(preview_bot, capsys, tmp_path):
    picture = tmp_path / "pic.png"
    picture.write_bytes(b"image-bytes")

    preview_bot.tweet(["a"], media=str(picture))

    assert Coo.media == Path(str(picture))


def test_tweet_with_media_posts_file_contents(bot, fake_twitter, tmp_path):
    picture = tmp_path / "pic.png"
    picture.write_bytes(b"image-bytes")

    bot.tweet(["hello"], media=str(picture))

    assert len(fake_twitter.posts) == 1
    assert fake_twitter.posts[0]["text"] == "hello"
    assert fake_twitter.posts[0]["kwargs"] == {"media"}
    assert fake_twitter.posts[0]["media"] == b"image-bytes"


def test_tweet_without_media_omits_media_argument(bot, fake_twitter):
    bot.tweet(["hello"])

    assert fake_twitter.posts == [{"text": "hello", "kwargs": set(), "media": None}]


def test_tweet_template_is_applied_to_posted_text(bot, fake_twitter):
    bot.tweet(["hi"], template="Hello $message!")

    assert [post["text"] for post in fake_twitter.posts] == ["Hello hi!"]


def test_aleatory_shuffles_updates_before_posting(preview_bot, monkeypatch, capsys):
    shuffled = []

    def reverse_in_place(seq):
        shuffled.append(list(seq))
        seq.reverse()

    monkeypatch.setattr(coo_module.random, "shuffle", reverse_in_place)
    updates = ["a", "b", "c"]

    result = preview_bot.tweet(updates, aleatory=True)

    assert shuffled == [["a", "b", "c"]]
    assert result is updates
    assert capsys.readouterr().out == "c\nb\na\n"


def test_schedule_rejects_entries_that_are_not_tuples(bot):
    with pytest.raises(ScheduleError) as info:
        bot.schedule(["2040-01-01 10:00"])

    assert str(info.value) == ScheduleError.wrongListMsg
    assert not isinstance(info.value, TweetTypeError)


@pytest.mark.parametrize("entry", [(0, None), (0, None, "a", None, "extra")])
def test_schedule_rejects_tuples_of_wrong_length(bot, entry):
    with pytest.raises(ScheduleError) as info:
        bot.schedule([entry])

    assert str(info.value) == ScheduleError.tupleLenError


def test_schedule_posts_every_tuple_in_preview_mode(preview_bot, capsys):
    preview_bot.schedule([(0, None, "first"), (0, "<$message>", "second")])

    lines = capsys.readouterr().out.splitlines()

    assert sorted(lines) == sorted(["first", "<second>"])
    assert preview_bot.loop.is_closed()


def test_schedule_template_argument_becomes_global_template(preview_bot, capsys):
    preview_bot.schedule([(0, None, "hi")], template="[$message]")

    assert Coo.global_template == "[$message]"
    assert capsys.readouterr().out == "[hi]\n"
