"""Unit tests for langsplit.splitter.split, convert and detect_language."""

import collections

from langsplit import splitter


class _RecordingLogger(object):
    def __init__(self):
        self.messages = []

    def info(self, msg, *args, **kwargs):
        self.messages.append(msg)


def _mock_detector(mapping):
    def _detect(text):
        return mapping[text]
    return _detect


def _never_called(text):
    raise AssertionError('detect_language should not be called')


def _failing_detector(text):
    raise ValueError('no features in text')


def test_split_two_keyed_languages():
    result = splitter.split('en:Hello.:fr:Bonjour')
    assert isinstance(result, collections.OrderedDict)
    assert dict(result) == {'en': 'Hello', 'fr': 'Bonjour'}


def test_split_merges_repeated_language_keys():
    result = splitter.split('en:Hello.:fr:Bonjour.:en:World')
    assert dict(result) == {'en': 'HelloWorld', 'fr': 'Bonjour'}


def test_split_orders_keys_by_first_appearance():
    result = splitter.split('fr:A.:en:B.:fr:C')
    assert list(result.keys()) == ['fr', 'en']
    assert result['fr'] == 'AC'
    assert result['en'] == 'B'


def test_split_returns_falsy_input_unchanged():
    assert splitter.split('') == ''
    assert splitter.split(None) is None


def test_split_short_leading_separator_special_case():
    result = splitter.split('.:en')
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [('en', '')]


def test_split_leading_separator_with_content():
    assert dict(splitter.split('.:en:Hi')) == {'en': 'Hi'}


def test_split_accepts_newline_and_carriage_return_after_key():
    result = splitter.split('en\nHello.:fr\rBonjour')
    assert dict(result) == {'en': 'Hello', 'fr': 'Bonjour'}


def test_split_keeps_later_colons_in_text():
    assert dict(splitter.split('en:Time: 10:00')) == {'en': 'Time: 10:00'}


def test_split_custom_separator():
    result = splitter.split('en:Hello|fr:Bonjour', sep='|')
    assert dict(result) == {'en': 'Hello', 'fr': 'Bonjour'}


def test_split_allows_three_letter_keys_when_max_key_length_is_three():
    result = splitter.split('eng:Hello.:fr:Bonjour', max_key_length=3)
    assert dict(result) == {'eng': 'Hello', 'fr': 'Bonjour'}


def test_split_rejects_key_shorter_than_min_key_length(monkeypatch):
    monkeypatch.setattr(splitter, 'detect_language',
                        _mock_detector({'en:Hi': 'xy'}))
    result = splitter.split('en:Hi', min_key_length=3, max_key_length=3)
    assert list(result.items()) == [('xy', 'en:Hi')]


def test_split_groups_unkeyed_segments_by_detected_language(monkeypatch):
    monkeypatch.setattr(splitter, 'detect_language',
                        _mock_detector({'Bonjour': 'fr', 'Hola': 'es',
                                        'Salut': 'fr'}))
    result = splitter.split('Bonjour.:Hola.:Salut')
    assert list(result.items()) == [('fr', 'BonjourSalut'), ('es', 'Hola')]


def test_split_detects_each_paragraph_separately(monkeypatch):
    monkeypatch.setattr(splitter, 'detect_language',
                        _mock_detector({'Bonjour': 'fr', 'Hola': 'es'}))
    result = splitter.split('Bonjour\n\nHola')
    assert list(result.items()) == [('fr', 'Bonjour\n\n'), ('es', 'Hola')]


def test_split_detection_failure_falls_back_to_unknown_and_logs(monkeypatch):
    recorder = _RecordingLogger()
    monkeypatch.setattr(splitter, 'logger', recorder)
    monkeypatch.setattr(splitter, 'detect_language', _failing_detector)
    result = splitter.split('Hello world')
    assert dict(result) == {splitter.settings.UNKNOWN_LANGUAGE: 'Hello world'}
    assert recorder.messages == ['Language not detected: Hello world']


def test_split_without_autodetect_skips_detection(monkeypatch):
    monkeypatch.setattr(splitter, 'detect_language', _never_called)
    result = splitter.split('en:Hi.:fr:Bonjour', autodetect=False)
    assert dict(result) == {'en': 'Hi', 'fr': 'Bonjour'}


def test_split_markdown_uses_newline_after_key():
    result = splitter.split('en:Hello.:fr:Bonjour', markdown=True)
    assert result == '.:en\nHello.:fr\nBonjour'


def test_split_markdown_title_uses_colon_after_key():
    result = splitter.split('en:Hello.:fr:Bonjour', markdown=True, title=True)
    assert result == '.:en:Hello.:fr:Bonjour'


def test_split_markdown_output_round_trips():
    original = {'en': 'Hello', 'fr': 'Bonjour'}
    for title in (False, True):
        markdown = splitter.split('en:Hello.:fr:Bonjour', markdown=True,
                                  title=title)
        assert dict(splitter.split(markdown)) == original


def test_convert_returns_non_dict_input_unchanged():
    assert splitter.convert('plain text') == 'plain text'
    assert splitter.convert(None) is None


def test_convert_strips_leading_whitespace_of_values():
    result = splitter.convert({'en': '   Hi', 'fr': 'Salut  '})
    assert result == '.:en\nHi.:fr\nSalut'


def test_detect_language_maps_chinese_variants_to_cn(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, 'detect', lambda text: 'zh-tw')
    assert splitter.detect_language('some text') == 'cn'
    monkeypatch.setattr(splitter.langdetect, 'detect', lambda text: 'zh-cn')
    assert splitter.detect_language('some text') == 'cn'


def test_detect_language_returns_two_letter_code(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, 'detect', lambda text: 'fr')
    assert splitter.detect_language('Bonjour le monde') == 'fr'
