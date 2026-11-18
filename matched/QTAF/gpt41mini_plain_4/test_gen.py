# test_unit_qtaf.py
import pytest
import types
import re
import sys
import os
import threading
from collections import deque

import six

from testbase import datadrive
from testbase.datadrive import DataDrive, is_datadrive, get_datadrive, load_datadrive_tests
from testbase.testcase import TestCase, TestCaseType, TestCaseRunner
from testbase.test import modify_settings


##############################
# Helpers and Mocks
##############################

class DummyTestResult:
    def __init__(self):
        self.records = []
        self.error_msgs = []
        self.logs = []
        self.steps = []
        self.warnings = []
        self.exceptions = []
        self._started = False
        self._ended = False
        self.passed = True

    def begin_test(self, testcase):
        self._started = True
        self.testcase = testcase

    def end_test(self):
        self._ended = True

    def error(self, msg):
        self.error_msgs.append(msg)
        self.passed = False

    def info(self, info):
        self.logs.append(info)

    def begin_step(self, stepinfo):
        self.steps.append(stepinfo)

    def log_record(self, level, msg, extra=None):
        self.records.append((level, msg, extra))

    def exception(self, msg):
        self.exceptions.append(msg)

    def warning(self, msg):
        self.warnings.append(msg)


class DummyTestResultFactory:
    def create(self, testcase):
        return DummyTestResult()


class DummyTestCase(TestCase):
    '''Simple dummy TestCase child for testing'''

    def __init__(self, testdata=None, testdataname=None, attrs=None):
        super().__init__(testdata, testdataname, attrs)
        self.run_test_called = False

    def run_test(self):
        self.run_test_called = True

##############################
# Tests for DataDrive
##############################

def test_datadrive_decorator_sets_attribute():
    data = [1, 2, 3]
    @datadrive.DataDrive(data)
    class T(DummyTestCase):
        pass
    assert hasattr(T, '__qtaf_datadrive__')
    dd = T.__qtaf_datadrive__
    assert isinstance(dd, DataDrive)
    assert len(dd) == 3
    for i in range(len(data)):
        assert dd[i] == data[i]

def test_datadrive_decorator_rejects_non_testcase():
    with pytest.raises(TypeError):
        @datadrive.DataDrive([1, 2])
        class NotATest(object):
            pass

def test_datadrive_iteration_and_getitem_list_and_dict():
    # list/tuple iteration yields indices
    dd_list = DataDrive([10, 20, 30])
    indices = list(iter(dd_list))
    assert indices == [0, 1, 2]
    assert dd_list[1] == 20

    dd_tuple = DataDrive((100, 200))
    indices = list(iter(dd_tuple))
    assert indices == [0, 1]
    assert dd_tuple[0] == 100

    # dict iteration yields keys
    dd_dict = DataDrive({'a': 1, 'b': 2})
    keys = list(iter(dd_dict))
    assert set(keys) == {'a', 'b'}
    assert dd_dict['a'] == 1

def test_is_datadrive_and_get_datadrive():
    @datadrive.DataDrive([1,2])
    class T(DummyTestCase):
        pass

    assert is_datadrive(T)
    assert is_datadrive(T())
    dd = get_datadrive(T)
    assert isinstance(dd, DataDrive)

def test__get_translated_in_datadrive_basic(monkeypatch):
    dd = {'a': 1, 'b': 2}
    # This is basic and should return the value for matching key
    val = datadrive._get_translated_in_datadrive('a', dd)
    assert val == 1

    # Non-string key, convert to str
    dd2 = {1: 'val1', 2: 'val2'}

    val2 = datadrive._get_translated_in_datadrive(1, dd2)
    assert val2 == 'val1'

    # When key not found, raises
    with pytest.raises(ValueError):
        datadrive._get_translated_in_datadrive('unknown', dd)

def test_load_datadrive_tests_with_name_key(monkeypatch):
    # Define a datadrive test class with data dictionary having __attrs__
    dd = {
        'case1': {'foo': 'bar', '__attrs__': {'owner': 'someone'}},
        'case2': 'simpledata'
    }
    @datadrive.DataDrive(dd)
    class TestCls(DummyTestCase):
        pass

    # Load tests without name returns all
    loaded_tests = load_datadrive_tests(TestCls)
    assert len(loaded_tests) == 2
    names = {t.casedataname for t in loaded_tests}
    assert names == set(dd.keys())
    for test in loaded_tests:
        if test.casedataname == 'case1':
            assert test.owner == 'someone'
            assert test.casedata == dd['case1']
        else:
            assert test.casedata == dd['case2']

    # Load tests filtered by name key
    tests_name1 = load_datadrive_tests(TestCls, name='case1')
    assert len(tests_name1) == 1
    assert tests_name1[0].casedataname == 'case1'
    assert tests_name1[0].owner == 'someone'

def test_load_datadrive_tests_name_not_in_data(monkeypatch):
    data = {'GOOD': 'ok'}
    @datadrive.DataDrive(data)
    class T(DummyTestCase):
        pass

    # Monkeypatch translate_bad_char to pass through
    monkeypatch.setattr(datadrive, "translate_bad_char", lambda x: x)
    monkeypatch.setattr(datadrive, "has_bad_char", lambda x: False)

    with pytest.raises(ValueError):
        load_datadrive_tests(T, name="not_exist")

##############################
# Tests for TestCase basic properties & asserts
##############################

def test_testcase_properties_and_casedata_attrs():
    attrs = {'owner': 'me', 'priority': TestCase.EnumPriority.High,
             'status': TestCase.EnumStatus.Ready, 'timeout': 10,
             'tags': ['tag1', 'tag2'], '__doc__': 'custom doc'}

    tc = DummyTestCase(testdata={'key': 'value'}, testdataname='name123', attrs=attrs)
    # Check properties
    assert tc.casedata == {'key': 'value'}
    assert tc.casedataname == 'name123'
    assert tc.owner == 'me'
    assert tc.priority == TestCase.EnumPriority.High
    assert tc.status == TestCase.EnumStatus.Ready
    assert tc.timeout == 10
    assert 'tag1' in tc.tags and 'tag2' in tc.tags
    assert tc.test_doc == 'custom doc'

def test_testcase_test_name_and_class_name():
    class FooTest(DummyTestCase):
        pass
    inst = FooTest()
    # When no casedataname, test_name equals full class name
    assert inst.test_name == inst.test_class_name
    # When casedataname set
    inst2 = FooTest(testdataname='foo')
    assert inst2.test_name.endswith('foo')

def test_testcase_assert_equal_and_match(monkeypatch):
    # Setup dummy testcase and catch records
    tc = DummyTestCase()
    res = DummyTestResult()
    tc._TestCase__testresult = res  # monkeypatch protected attribute

    # When values are equal, returns True
    assert tc.assert_equal("msg", "abc", "abc") is True
    # When not equal, returns False and logs record
    result = tc.assert_equal("msg fail", "xxx", "yyy")
    assert result is False
    # Check log record contains the message substring
    assert any("msg fail" in item[1] for item in res.records)

    # For assert_match success (regex matches string)
    assert tc.assert_match("msg match", "abcdef", "^abc") is True
    # For assert_match failure logs and returns False
    res.records.clear()
    result2 = tc.assert_match("msg mismatch", "def", "^abc")
    assert result2 is False
    assert any("msg mismatch" in item[1] for item in res.records)

def test_testcase_assert_raises_on_false_value(monkeypatch):
    tc = DummyTestCase()
    res = DummyTestResult()
    tc._TestCase__testresult = res
    # Hook _log_assert_failed to raise RuntimeError if settings disables continue
    # Patch settings to QTAF_ASSERT_CONTINUE=False temporarily
    with modify_settings(QTAF_ASSERT_CONTINUE=False):
        with pytest.raises(RuntimeError):
            tc.assert_(message="fail msg", value=False)
    # When value is True, no exception
    tc.assert_("should pass", True)

def test_testcase_wait_for_equal_and_match(monkeypatch):
    class Obj:
        def __init__(self):
            self.foo = "bar"
    tc = DummyTestCase()
    res = DummyTestResult()
    tc._TestCase__testresult = res

    obj = Obj()

    # wait_for_equal returns True if property equals expected immediately
    assert tc.wait_for_equal("fail equal", obj, "foo", "bar", timeout=0.5, interval=0.1)
    # wait_for_equal returns False if property never changes
    result = tc.wait_for_equal("fail eq", obj, "foo", "baz", timeout=0.5, interval=0.1)
    assert result is False
    assert any("fail eq" in r[1] for r in res.records)

    # wait_for_match returns True if property matches regexp immediately
    assert tc.wait_for_match("fail match", obj, "foo", r"^ba.r$", timeout=0.5, interval=0.1)
    # wait_for_match returns False if property never matches
    res.records.clear()
    result2 = tc.wait_for_match("fail match2", obj, "foo", r"^nope$", timeout=0.5, interval=0.1)
    assert result2 is False
    assert any("fail match2" in r[1] for r in res.records)

def test_testcase_log_info_and_fail(monkeypatch):
    tc = DummyTestCase()
    res = DummyTestResult()
    tc._TestCase__testresult = res
    tc.log_info("info msg")
    assert "info msg" in res.logs

    tc.fail("fail message")
    assert "fail message" in res.error_msgs

##############################
# Tests for modify_settings context manager
##############################

def test_modify_settings_context_manager_basic():
    # Save the original value if any
    orig_value = getattr(datadrive.settings, "DATA_DRIVE", None)

    # Modify a setting temporarily
    with modify_settings(DATA_DRIVE=True):
        assert datadrive.settings.DATA_DRIVE is True

    # After context exit, settings restored
    if orig_value is not None:
        assert datadrive.settings.DATA_DRIVE == orig_value
    else:
        assert not hasattr(datadrive.settings, "DATA_DRIVE")

def test_modify_settings_add_and_remove_key():
    # Temporarily add a new key
    new_key = "TEST_KEY_12345"
    assert not hasattr(datadrive.settings, new_key)
    
    with modify_settings(**{new_key: 5}):
        assert getattr(datadrive.settings, new_key) == 5

    assert not hasattr(datadrive.settings, new_key)

##############################
# Tests for TestCaseRunner basics and error conditions
##############################

def test_testcaserunner_check_testcase_requires_attrs(monkeypatch):
    # Setup a testcase missing required attribute
    class MissingAttrTest(DummyTestCase):
        pass

    runner = TestCaseRunner()
    tc = MissingAttrTest()

    # forcibly clear required attributes to None
    for attr in ['owner', 'priority', 'status', 'timeout']:
        setattr(tc, attr, None)

    with pytest.raises(RuntimeError):
        runner._check_testcase(tc)

def test_testcaserunner_check_testcase_requires_test_doc(monkeypatch):
    class NoDocTest(DummyTestCase):
        pass
    tc = NoDocTest()
    # Set required attrs
    tc.owner = "x"
    tc.priority = TestCase.EnumPriority.Low
    tc.status = TestCase.EnumStatus.Ready
    tc.timeout = 1
    # Remove doc forcibly
    tc._TestCase__test_doc = None
    tc.__class__.__doc__ = None
    runner = TestCaseRunner()
    with pytest.raises(RuntimeError):
        runner._check_testcase(tc)

def test_testcaserunner_forbid_init_overload_raises():
    # Trying to define a class inheriting TestCase with __init__ overload should raise RuntimeError

    with pytest.raises(RuntimeError):
        class BadTest(TestCase):
            def __init__(self):
                pass

def test_testcaserunner__rewrite_assert_skips_base_TestCase(monkeypatch):
    runner = TestCaseRunner()
    # Patch settings to disable rewriting
    monkeypatch.setattr(datadrive.settings, "QTAF_REWRITE_ASSERT", False)
    class A(TestCase):
        def some_method(self):
            pass
    # Should not raise
    runner._rewrite_assert(A)
    # Should skip for TestCase class itself even if QTAF_REWRITE_ASSERT True
    monkeypatch.setattr(datadrive.settings, "QTAF_REWRITE_ASSERT", True)
    runner._rewrite_assert(TestCase)

def test_testcaserunner_thread_run_error_logged(monkeypatch):
    # Monkeypatch DummyTestCase with method that raises error on pre_test
    class ErrorTest(DummyTestCase):
        def pre_test(self):
            raise RuntimeError("fail pre_test")
        def run_test(self):
            pass

    runner = TestCaseRunner()
    tc = ErrorTest()
    tc.owner = "x"
    tc.priority = TestCase.EnumPriority.Low
    tc.status = TestCase.EnumStatus.Ready
    tc.timeout = 1
    runner._testcase = tc
    runner._testresult = DummyTestResult()
    runner._subtasks = deque(['init_test', 'pre_test', 'run_test', 'post_test', 'clean_test'])
    runner._lock = threading.Lock()

    runner._thread_run()  # should error log for pre_test, no raise

    assert any('pre_test执行失败' in s for s in runner._testresult.exceptions)

def test_testcaserunner__get_current_traceback_returns_trace(monkeypatch):
    runner = TestCaseRunner()
    # We start a dummy thread and get its stack

    def dummy_threading_func():
        while getattr(threading.currentThread(), "run_flag", True):
            pass
    th = threading.Thread(target=dummy_threading_func)
    th.daemon = True
    th.start()

    try:
        tb = runner._get_current_traceback(th)
        assert "Traceback" in tb
        assert 'dummy_threading_func' in tb
    finally:
        th.run_flag = False
        th.join(timeout=1)

def test_testcaserunner__get_current_traceback_raises(monkeypatch):
    runner = TestCaseRunner()
    dummy_thread = threading.Thread(target=lambda: None)
    # Make dummy_thread ident no match _current_frames keys to test raise path
    dummy_thread.ident = -1
    with pytest.raises(RuntimeError):
        runner._get_current_traceback(dummy_thread)

##############################
# Tests for DataDrive iterable from generator case
##############################

def test_datadrive_generator_input():
    def gen():
        yield "one"
        yield "two"
    dd = DataDrive(gen())
    assert isinstance(dd._case_data, list)  # generator converted to list internally
    assert list(dd) == [0, 1]  # yields indices for list input

##############################
# Test DataDrive with keys containing bad characters, expecting warning logged
##############################

def test_load_datadrive_tests_warn_on_bad_char(monkeypatch):
    # Setup a key with bad character
    data = {
        "bad key": {'__attrs__': {'owner': 'badowner'}}
    }
    @datadrive.DataDrive(data)
    class BadKeyTest(DummyTestCase):
        pass

    recorded_warnings = []
    monkeypatch.setattr(datadrive.logger, "warn", lambda msg: recorded_warnings.append(msg))
    monkeypatch.setattr(datadrive.has_bad_char, lambda x: True)
    monkeypatch.setattr(datadrive.translate_bad_char, lambda x: x.replace(" ", "_"))

    tests = load_datadrive_tests(BadKeyTest)
    assert len(tests) == 1
    assert recorded_warnings
    assert "badowner" == tests[0].owner

##############################
# Miscellaneous test ensuring coverage of little paths
##############################

def test_testcase_test_extra_info_dynamic_property(monkeypatch):
    class X(DummyTestCase):
        test_extra_info_def = [('foo', None), ('bar', None)]

    instance = X()
    instance.foo = 123
    # bar not set should be None
    info = instance.test_extra_info
    assert info['foo'] == 123
    assert info['bar'] is None

def test_testcase_clean_test_calls_destroy(monkeypatch):
    tc = DummyTestCase()
    called = {}

    class DummySession:
        def destroy(self):
            called['destroy'] = True

    dummy_session = DummySession()
    tc._TestCase__resmgr_session = dummy_session
    tc.clean_test()
    assert called.get('destroy') is True

def test_testcase_start_step_and_log_info(monkeypatch):
    tc = DummyTestCase()
    res = DummyTestResult()
    tc._TestCase__testresult = res
    tc.start_step("step1")
    tc.log_info("info1")
    assert "step1" in res.steps
    assert "info1" in res.logs

def test_testcase_fail_logs_error(monkeypatch):
    tc = DummyTestCase()
    res = DummyTestResult()
    tc._TestCase__testresult = res
    tc.fail("fail message")
    assert "fail message" in res.error_msgs

def test_testcase_assert_equal_string_conversion(monkeypatch):
    tc = DummyTestCase()
    res = DummyTestResult()
    tc._TestCase__testresult = res

    # Use byte string inputs to test conversion to smart_text
    b_actual = b'test'
    b_expect = b'test'
    assert tc.assert_equal("msg", b_actual, b_expect) is True
    b_bad_expect = b'fail'
    assert tc.assert_equal("msg2", b_actual, b_bad_expect) is False
    # One record logged for failure
    assert any("msg2" in r[1] for r in res.records)
