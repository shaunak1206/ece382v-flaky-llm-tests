import collections

import pytest

import langsplit.splitter as splitter


def as_items(result):
    return list(result.items())


@pytest.fixture
def fake_detect(monkeypatch):
    """Replace language detection with a deterministic fake that records its calls."""
    calls = []

    def install(func):
        def recorder(text, max_length=2):
            calls.append(text)
            return func(text)

        monkeypatch.setattr(splitter, 'detect_language', recorder)
        return calls

    return install


def test_empty_text_is_returned_unchanged():
    assert splitter.split('') == ''
    assert splitter.split(None) is None


def test_short_text_without_newline_returns_single_key():
    result = splitter.split('.:en')

    assert isinstance(result, collections.OrderedDict)
    assert as_items(result) == [('en', '')]


def test_splits_on_default_separator_and_colon_end():
    result = splitter.split('.:en:Hello.:fr:Bonjour')

    assert isinstance(result, collections.OrderedDict)
    assert as_items(result) == [('en', 'Hello'), ('fr', 'Bonjour')]


def test_same_language_chunks_are_combined_in_first_appearance_order():
    result = splitter.split('.:fr:A.:en:B.:fr:C')

    assert as_items(result) == [('fr', 'AC'), ('en', 'B')]


def test_newline_is_accepted_as_key_end():
    result = splitter.split('.:en\nHello.:fr\nBonjour')

    assert as_items(result) == [('en', 'Hello'), ('fr', 'Bonjour')]


def test_custom_separator_is_used():
    result = splitter.split('|en:Hello|fr:Bonjour', sep='|')

    assert as_items(result) == [('en', 'Hello'), ('fr', 'Bonjour')]


def test_empty_tokens_are_ignored():
    result = splitter.split('.:.:en:Hi')

    assert as_items(result) == [('en', 'Hi')]


def test_key_length_follows_max_key_length():
    result = splitter.split('.:eng:Hi', max_key_length=3)

    assert as_items(result) == [('eng', 'Hi')]


def test_key_shorter_than_min_key_length_is_not_a_key(fake_detect):
    calls = fake_detect(lambda text: 'en')

    result = splitter.split('.:e:Hi')

    assert as_items(result) == [('en', 'e:Hi')]
    assert calls == ['e:Hi']


def test_keyed_chunks_do_not_trigger_detection(fake_detect):
    calls = fake_detect(lambda text: 'de')

    result = splitter.split('.:en:Hi.:fr:Salut')

    assert as_items(result) == [('en', 'Hi'), ('fr', 'Salut')]
    assert calls == []


def test_autodetect_assigns_detected_language_to_unkeyed_text(fake_detect):
    calls = fake_detect(lambda text: 'en')

    result = splitter.split('.:Hello world')

    assert as_items(result) == [('en', 'Hello world')]
    assert calls == ['Hello world']


def test_autodetect_treats_paragraphs_separately(fake_detect):
    fake_detect(lambda text: 'en' if text.startswith('Eng') else 'de')

    result = splitter.split('.:Eng one\n\nDeutsch two')

    assert as_items(result) == [('en', 'Eng one\n\n'), ('de', 'Deutsch two')]


def test_empty_paragraphs_are_skipped(fake_detect):
    calls = fake_detect(lambda text: 'en')

    result = splitter.split('.:A\n\n\n\nB')

    assert as_items(result) == [('en', 'A\n\nB')]
    assert calls == ['A', 'B']


def test_detection_failure_falls_back_to_unknown_language(fake_detect):
    def failing(text):
        raise RuntimeError('no language could be detected')

    fake_detect(failing)

    result = splitter.split('.:Something')

    assert as_items(result) == [(splitter.settings.UNKNOWN_LANGUAGE, 'Something')]


def test_markdown_output_uses_newline_end():
    result = splitter.split('.:en:Hello.:fr:Bonjour', markdown=True)

    assert result == '.:en\nHello.:fr\nBonjour'


def test_markdown_title_output_uses_colon_end():
    result = splitter.split('.:en:Hello.:fr:Bonjour', markdown=True, title=True)

    assert result == '.:en:Hello.:fr:Bonjour'


def test_markdown_round_trip_preserves_languages():
    original = '.:en\nHello\n.:fr\nBonjour'

    parsed = splitter.split(original)
    markdown = splitter.split(
        '.:en:Hello.:fr:Bonjour', markdown=True
    )

    assert as_items(parsed) == [('en', 'Hello\n'), ('fr', 'Bonjour')]
    assert markdown.startswith('.:en\nHello')
