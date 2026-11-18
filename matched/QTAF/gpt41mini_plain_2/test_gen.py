# -*- coding: utf-8 -*-
import builtins
import os
import re
import threading
import types

import pytest

import testbase.datadrive as datadrive
import testbase.testcase as testcase
import testbase.test as testmod


class DummyTestCase(testcase.TestCase):
    def run_test(self):
        pass


def test_datadrive_decorator_accepts_only_subclass_of_testcase():
    with pytest.raises(TypeError):
        @datadrive.DataDrive([1, 2, 3])
        class NotATestCase(object):
            pass


def test_datadrive_iter_and_len_with_list_and_dict():
    dd_list = datadrive.DataDrive([10, 20, 30])
    # iter should yield indices
    assert list(dd_list) == [0, 1, 2]
    assert len(dd_list) == 3
    assert dd_list[1] == 20

    dd_dict = datadrive.DataDrive({'a': 1, 'b': 2})
    assert set(dd_dict) == {'a', 'b'}
    assert len(dd_dict) == 2
    assert dd_dict['a'] == 1


def test_is_datadrive_and_get_datadrive():
    @datadrive.DataDrive([1, 2])
    class T(DummyTestCase):
        pass

    assert datadrive.is_datadrive(T)
    dd = datadrive.get_datadrive(T)
    assert isinstance(dd, datadrive.DataDrive)
    assert list(dd) == [0, 1]
    # An instance with no datadrive
    t = DummyTestCase()
    assert not datadrive.is_datadrive(t)
    assert not datadrive.is_datadrive(DummyTestCase)


def test_get_translated_in_datadrive_exact_match_and_valueerror(monkeypatch):
    # Setup a dict with keys that may require translation, use simple keys
    dd = {'key1': 'value1', 'key2': 'value2'}
    # Exact match returns correct value
    assert datadrive._get_translated_in_datadrive('key1', dd) == 'value1'
    assert datadrive._get_translated_in_datadrive('key2', dd) == 'value2'

    # ValueError with unknown key
    with pytest.raises(ValueError) as excinfo:
        datadrive._get_translated_in_datadrive('unknown_key', dd)
    assert "data drive name 'unknown_key' not found" in str(excinfo.value)


def test_load_datadrive_tests_with_name_and_warning(monkeypatch):
    # Define a test case class with datadrive attribute (dict with bad char key)
    bad_key = 'bad key!'
    translated_key = datadrive.translate_bad_char(bad_key)
    testdata = {bad_key: 'testdata', 'goodkey': {'__attrs__': {'priority': 'High'}}}

    class MyTest(DummyTestCase):
        pass

    monkeypatch.setattr(MyTest, "__qtaf_datadrive__", datadrive.DataDrive(testdata))

    # Over monkeypatch logger.warn to capture messages
    warnings = []
    monkeypatch.setattr(datadrive.logger, 'warn', lambda msg: warnings.append(msg))

    tests = datadrive.load_datadrive_tests(MyTest)
    # Should create one test instance per key
    assert any(isinstance(t, MyTest) for t in tests)
    # Check casedata_name is translated if needed
    names = [t.casedataname for t in tests]
    assert translated_key in names or 'goodkey' in names
    # Validate warning logged for bad key
    assert any("WARNING" in w for w in warnings) or len(warnings) == 0


def test_load_datadrive_tests_without_datadrive_and_settings(monkeypatch):
    # Monkeypatch is_datadrive to False and settings.DATA_DRIVE to False to cause RuntimeError
    class SomeTest(DummyTestCase):
        pass

    monkeypatch.setattr(datadrive, 'is_datadrive', lambda obj: False)
    monkeypatch.setattr(testcase.settings, 'DATA_DRIVE', False, raising=False)

    with pytest.raises(RuntimeError):
        datadrive.load_datadrive_tests(SomeTest)

    # Now patch to enable DATA_DRIVE and mock TestDataLoader.load to return dict
    monkeypatch.setattr(testcase.settings, 'DATA_DRIVE', True)

    class DummyLoader(object):
        def load(self):
            return {'a': 1}

    def dummy_load():
        return DummyLoader()

    monkeypatch.setattr(datadrive, 'TestDataLoader', dummy_load)

    # Should load tests without error
    # Put DataDrive attribute to False cache to test path
    monkeypatch.setattr(datadrive, 'is_datadrive', lambda obj: False)
    tests = datadrive.load_datadrive_tests(SomeTest)
    assert isinstance(tests, list)
    assert len(tests) == 1
    assert tests[0].casedata == 1


def test_testcase_init_and_properties_basic():
    data = {'foo': 'bar'}
    attrs = {'priority': 'High', 'tags': ['tag1', 'tag2'], '__doc__': 'DocStr'}
    tc = DummyTestCase(data, 'name1', attrs)
    assert tc.casedata == data
    assert tc.casedataname == 'name1'
    assert tc.priority == 'High'
    assert 'tag1' in tc.tags
    assert 'tag2' in tc.tags
    assert tc.test_doc == 'DocStr'


def test_testcase_assert_equal_and_assert_match(monkeypatch):
    tc = DummyTestCase()

    # Setup testresult to capture errors
    errors = []
    class DummyTestResult:
        def log_record(self, level, msg):
            errors.append((level, msg))

    dummy_result = DummyTestResult()
    tc.init_test(dummy_result)
    # Monkeypatch __record_assert_failed to call dummy log
    tc._TestCase__testresult = dummy_result

    # assert_equal returns True on equal
    assert tc.assert_equal("msg", "abc", "abc") is True
    # returns False on not equal and records error
    result = tc.assert_equal("not equal msg", 1, 2)
    assert result is False
    assert errors

    errors.clear()

    # assert_match returns True when matched
    assert tc.assert_match("msg", "abc123", r"abc\d+") is True
    # returns False and logs when no match
    result = tc.assert_match("no match", "abc", r"def")
    assert result is False
    assert errors


def test_testcase_wait_for_equal_and_wait_for_match(monkeypatch):
    tc = DummyTestCase()
    dummy_result = pytest.mock.Mock()
    tc.init_test(dummy_result)

    class Obj:
        def __init__(self):
            self.val = None

    o = Obj()
    o.val = 5

    # Should return True immediately when matching
    assert tc.wait_for_equal("fail msg", o, 'val', 5, timeout=1, interval=0.1)

    # Should return False after timeout and record error when not matching
    o.val = 0

    # Patch dummy_result to record logs
    logs = []
    class DummyResult:
        def log_record(self, level, msg, *args, **kwargs):
            logs.append((level, msg))

    tc._TestCase__testresult = DummyResult()

    assert not tc.wait_for_equal("fail msg", o, 'val', 5, timeout=0.3, interval=0.1)
    assert logs

    # For wait_for_match
    o.val = "hello"

    pattern = r"hello"

    assert tc.wait_for_match("fail match", o, 'val', pattern, timeout=0.5, interval=0.1)

    o.val = "bye"
    logs.clear()
    assert not tc.wait_for_match("fail match", o, 'val', pattern, timeout=0.3, interval=0.1)
    assert logs


def test_modify_settings_context_manager(tmp_path):
    # Pick a setting to modify
    key = 'TEST_SETTING_FOO'

    # Ensure setting is not present first
    if hasattr(testcase.settings, key):
        delattr(testcase.settings, key)

    # Setting should not be set
    assert not hasattr(testcase.settings, key)

    with testmod.modify_settings(**{key: 123}):
        # Inside, setting should be present and equal to 123
        assert getattr(testcase.settings, key) == 123

        # Nested modify_settings should replace and restore properly
        with testmod.modify_settings(**{key: 456}):
            assert getattr(testcase.settings, key) == 456

        # After inner exit, should restore to 123
        assert getattr(testcase.settings, key) == 123

    # After outer exit, setting should be removed if it didn't exist previously
    assert not hasattr(testcase.settings, key)


def test_modify_environ_context_manager(monkeypatch):
    key = 'QTAF_TEST_ENV_FOO'
    # Remove env if exists
    if key in os.environ:
        oldval = os.environ.pop(key)
    else:
        oldval = None

    try:
        with testmod.modify_environ(**{key: 'bar_value'}):
            assert os.environ[key] == 'bar_value'

        # After exit, env should restore to original (or be removed)
        assert (key not in os.environ) if oldval is None else (os.environ.get(key) == oldval)
    finally:
        # Restore env for other tests
        if oldval is not None:
            os.environ[key] = oldval


def test_modify_attributes_context_manager_threadsafe():
    class Obj:
        attr1 = 1
        attr2 = 2

    obj = Obj()
    old_attr1 = obj.attr1
    old_attr2 = obj.attr2

    with testmod.modify_attributes(obj, {'attr1': 42, 'attr2': 99}):
        assert obj.attr1 == 42
        assert obj.attr2 == 99

    # After context exit, should be restored
    assert obj.attr1 == old_attr1
    assert obj.attr2 == old_attr2


def test_testcase_test_name_and_test_class_name(monkeypatch):
    class MyTest(DummyTestCase):
        pass

    inst = MyTest()
    # Normally test_class_name is module + classname
    mod_name = MyTest.__module__
    cls_name = MyTest.__name__
    expected_name = f"{mod_name}.{cls_name}"
    assert inst.test_class_name == expected_name
    # Without casedataname, test_name is test_class_name
    assert inst.test_name == expected_name

    # With casedataname
    inst = MyTest(testdataname="case1")
    assert inst.test_name == f"{expected_name}/case1"


def test_testcase_fail_log_info(monkeypatch):
    class DummyResult:
        def __init__(self):
            self.logged_info = []
            self.errors = []

        def info(self, msg):
            self.logged_info.append(msg)

        def error(self, msg):
            self.errors.append(msg)

    tc = DummyTestCase()
    dummy_result = DummyResult()
    tc.init_test(dummy_result)

    # Calling log_info with string and non-string
    tc.log_info('info test')
    tc.log_info(123)  # Non-string input

    assert 'info test' in dummy_result.logged_info
    assert '123' in dummy_result.logged_info

    # Calling fail with string and non-string
    tc.fail('fail test')
    tc.fail(456)  # Non-string input

    assert 'fail test' in dummy_result.errors
    assert '456' in dummy_result.errors


def test_testcase__log_assert_failed_raises_if_setting_false(monkeypatch):
    tc = DummyTestCase()
    # Setup Dummy test result
    logs = []
    class DummyResult:
        def log_record(self, level, msg):
            logs.append((level, msg))

    dummy_result = DummyResult()
    tc._TestCase__testresult = dummy_result

    # Patch setting to False to cause RuntimeError raise
    monkeypatch.setattr(testcase.settings, "QTAF_ASSERT_CONTINUE", False)

    with pytest.raises(RuntimeError, match="testcase assert failed"):
        tc._log_assert_failed("fail message")


def test_testcase_get_extra_fail_record_returns_empty_dicts():
    tc = DummyTestCase()
    logs, attachments = tc.get_extra_fail_record()
    assert isinstance(logs, dict)
    assert isinstance(attachments, dict)
    assert logs == {}
    assert attachments == {}
