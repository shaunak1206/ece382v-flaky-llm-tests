# testbase/tests/test_datadrive_testcase_modifysettings.py
import types
import re
import sys
import threading
import pytest
from testbase import datadrive
from testbase import testcase as tc
from testbase.test import modify_settings

# Dummy classes and helpers for testing
class DummyTestCase(tc.TestCase):
    def run_test(self):
        return "run_test called"

class DummyTestCaseWithAttrs(tc.TestCase):
    def __init__(self, testdata=None, testdataname=None, attrs=None):
        super().__init__(testdata, testdataname, attrs)

class DummyTestResult:
    def __init__(self):
        self.records = []
        self.steps = []
        self.errors = []
        self.warnings = []
        self.logged = []
        self.passed = True

    def begin_step(self, info):
        self.steps.append(info)

    def info(self, info):
        self.logged.append(('info', info))

    def error(self, msg):
        self.errors.append(msg)
        self.passed = False

    def log_record(self, level, message, extra=None):
        self.records.append((level, message, extra))

    def exception(self, msg):
        self.errors.append(msg)
        self.passed = False

    def begin_test(self, testcase):
        pass

    def end_test(self):
        pass

    def warning(self, msg):
        self.warnings.append(msg)


# --- Tests for DataDrive ---

def test_datadrive_init_accepts_list_tuple_dict():
    for data in ([1, 2, 3], (4, 5, 6), {'a': 1, 'b': 2}):
        dd = datadrive.DataDrive(data)
        assert dd._case_data == data

def test_datadrive_call_decorator_sets_attribute():
    @datadrive.DataDrive([1, 2])
    class MyTest(tc.TestCase): pass
    assert hasattr(MyTest, '__qtaf_datadrive__')
    assert isinstance(MyTest.__qtaf_datadrive__, datadrive.DataDrive)

def test_datadrive_call_raises_typeerror_when_non_testcase():
    with pytest.raises(TypeError):
        @datadrive.DataDrive([1,2])
        class NotATest(object): pass

def test_datadrive_iter_and_len_with_list_and_dict():
    dd_list = datadrive.DataDrive([10, 20])
    assert list(iter(dd_list)) == [0, 1]
    assert len(dd_list) == 2

    dd_dict = datadrive.DataDrive({'x': 1, 'y': 2})
    items = list(iter(dd_dict))
    assert 'x' in items and 'y' in items
    assert len(dd_dict) == 2

def test_datadrive_getitem_returns_expected_data():
    data = {'key1': 'val1', 'key2': 'val2'}
    dd = datadrive.DataDrive(data)
    assert dd['key1'] == 'val1'
    assert dd['key2'] == 'val2'

def test_is_datadrive_and_get_datadrive_properties():
    @datadrive.DataDrive([1])
    class MyTest(tc.TestCase): pass

    assert datadrive.is_datadrive(MyTest)
    assert datadrive.is_datadrive(MyTest())
    dd = datadrive.get_datadrive(MyTest)
    assert isinstance(dd, datadrive.DataDrive)

def test__get_translated_in_datadrive_returns_correct_value(monkeypatch):
    # Prepare dictionary with keys that when translated match name
    data = {'key-1': 123, 'key_2': 456}
    dd = data.copy()

    # monkeypatch translate_bad_char to e.g. replace '-' with '_'
    def fake_translate(s):
        return s.replace('-', '_')
    monkeypatch.setattr(datadrive, "translate_bad_char", fake_translate)

    # Name matches after translation to key_1 -> should return value 123
    val = datadrive._get_translated_in_datadrive('key_1', dd)
    assert val == 123

    # Exact key found
    val = datadrive._get_translated_in_datadrive('key_2', dd)
    assert val == 456

    # Non-existing key raises error
    with pytest.raises(ValueError):
        datadrive._get_translated_in_datadrive('nonexist', dd)

# --- Tests for load_datadrive_tests ---

def test_load_datadrive_tests_with_datadrive_class(monkeypatch):
    # Provide dummy test class with datadrive attribute and constructor tracking
    call_records = []
    class MyTest(tc.TestCase):
        def __init__(self, testdata=None, testdataname=None, attrs=None):
            call_records.append((testdata, testdataname, attrs))
            super().__init__(testdata, testdataname, attrs)
    # Patch datadrive attribute with dict and monkeypatch logger.warn
    dd_data = {'A': 1, "B": 2}
    setattr(MyTest, '__qtaf_datadrive__', datadrive.DataDrive(dd_data))

    monkeypatch.setattr(datadrive.logger, "warn", lambda msg: call_records.append(msg))

    tests = datadrive.load_datadrive_tests(MyTest)
    # Should create two test instances, with correct casedataname and testdata
    assert len(tests) == 2
    for test in tests:
        assert isinstance(test, MyTest)
        assert test.casedata in dd_data.values()
        assert test.casedataname in dd_data.keys()

    # Test with name specified, direct key
    tests_named = datadrive.load_datadrive_tests(MyTest, name='A')
    assert len(tests_named) == 1
    assert tests_named[0].casedata == 1

    # Test with name that has bad char triggers warning
    dd_bad = {'k-1': {'__attrs__': {'owner': 'owner1'}}}
    setattr(MyTest, '__qtaf_datadrive__', datadrive.DataDrive(dd_bad))
    warnings = []
    monkeypatch.setattr(datadrive.logger, "warn", lambda msg: warnings.append(msg))
    tests_bad = datadrive.load_datadrive_tests(MyTest)
    assert any("WARNING" in w for w in warnings)
    assert tests_bad[0].owner == 'owner1'

@pytest.mark.parametrize("case_data", [
    [1,2,3],   # list
    (4,5,6),   # tuple
    {'X': 7, 'Y': 8}  # dict
])
def test_datadrive_str_and_bytes_key_names(case_data):
    dd = datadrive.DataDrive(case_data)
    # Should yield keys as indexes or keys depending on input type
    keys = list(dd)
    # Keys for list/tuple are ints starting from 0
    if isinstance(case_data, (list, tuple)):
        for i in range(len(case_data)):
            assert i in keys
            assert dd[i] == case_data[i]
    else:
        for k in case_data:
            assert k in keys
            assert dd[k] == case_data[k]

# --- Tests for TestCase ---

def test_testcase_init_and_properties():
    attrs = {'owner': 'me', 'priority': tc.TestCase.EnumPriority.High,
             'status': tc.TestCase.EnumStatus.Ready, 'timeout': 30, '__doc__': 'docstring',
             'tags': ['tag1']}
    t = tc.TestCase(testdata={'foo': 'bar'}, testdataname="case1", attrs=attrs)
    assert t.casedata == {'foo': 'bar'}
    assert t.casedataname == "case1"
    assert t.owner == 'me'
    assert t.priority == tc.TestCase.EnumPriority.High
    assert t.status == tc.TestCase.EnumStatus.Ready
    assert t.timeout == 30
    assert t.test_doc == 'docstring'
    assert 'tag1' in t.tags
    # Tags should merge with base_tags (which might be empty)
    assert isinstance(t.tags, set)

def test_testcase_test_class_and_test_name_properties():
    t = tc.TestCase()
    cls = type(t)
    expected_cls_name = f"{cls.__module__}.{cls.__name__}"
    assert t.test_class_name == expected_cls_name
    # Without casedataname
    t2 = tc.TestCase()
    assert t2.test_name == expected_cls_name
    # With casedataname
    t3 = tc.TestCase(testdataname="data1")
    assert t3.test_name == f"{expected_cls_name}/data1"

def test_testcase_test_doc_property():
    class DocTest(tc.TestCase):
        """   docstring with spaces   """
        pass
    t = DocTest()
    assert t.test_doc == "docstring with spaces"
    # test_doc returns __doc__ when no override

def test_testcase_assert_equal_true_and_false(monkeypatch):
    t = tc.TestCase()
    monkeypatch.setattr(t, "_TestCase__testresult", DummyTestResult())
    # Should return True when equal
    assert t.assert_equal("msg", 1, 1) is True
    # Should return False and record when not equal
    monkeypatch.setattr(t, "_TestCase__testresult", DummyTestResult())
    res = t.assert_equal("msg", 1, 2)
    assert res is False

def test_testcase_assert_match_true_and_false(monkeypatch):
    t = tc.TestCase()
    monkeypatch.setattr(t, "_TestCase__testresult", DummyTestResult())
    assert t.assert_match("msg", "abc123", r"\w+")
    monkeypatch.setattr(t, "_TestCase__testresult", DummyTestResult())
    assert t.assert_match("msg", "abc123", r"^\d+$") is False

def test_testcase_assert_(raises=RuntimeError):
    t = tc.TestCase()
    monkeypatch.setattr(t, "_TestCase__testresult", DummyTestResult())
    # value True no exception
    t.assert_("should not fail", True)
    # value False raises RuntimeError if setting disables continue
    from testbase.conf import settings
    with modify_settings(QTAF_ASSERT_CONTINUE=False):
        with pytest.raises(RuntimeError):
            t.assert_("fail message", False)

def test_wait_for_equal_and_match_behavior(monkeypatch):
    class OBJ:
        def __init__(self):
            self.value = "ready"
    obj = OBJ()
    t = tc.TestCase()
    monkeypatch.setattr(t, "_TestCase__testresult", DummyTestResult())
    # Should succeed when attribute matches expected immediately
    assert t.wait_for_equal("msg", obj, "value", "ready", timeout=0.1, interval=0.05)
    # Should fail when not matched
    obj.value = "no"
    monkeypatch.setattr(t, "_TestCase__testresult", DummyTestResult())
    assert not t.wait_for_equal("msg", obj, "value", "ready", timeout=0.1, interval=0.05)

    # For wait_for_match test
    obj.value = "abc123"
    assert t.wait_for_match("msg", obj, "value", r"abc\d+", timeout=0.1, interval=0.05)
    obj.value = "zzz"
    monkeypatch.setattr(t, "_TestCase__testresult", DummyTestResult())
    assert not t.wait_for_match("msg", obj, "value", r"abc\d+", timeout=0.1, interval=0.05)

def test_log_info_and_fail_calls(monkeypatch):
    t = tc.TestCase()
    dummy_result = DummyTestResult()
    monkeypatch.setattr(t, "_TestCase__testresult", dummy_result)

    t.log_info("info message")
    assert ('info', "info message") in dummy_result.logged

    t.fail("fail message")
    assert "fail message" in dummy_result.errors

def test_start_step(monkeypatch):
    t = tc.TestCase()
    dummy_result = DummyTestResult()
    monkeypatch.setattr(t, "_TestCase__testresult", dummy_result)
    t.start_step("step1")
    assert "step1" in dummy_result.steps

def test_clean_test_calls_resources_destroy(monkeypatch):
    t = tc.TestCase()
    class ResMgrSessionMock:
        def __init__(self):
            self.destroy_called = False
        def destroy(self):
            self.destroy_called = True
    sess = ResMgrSessionMock()
    t._TestCase__resmgr_session = sess
    t.clean_test()
    assert sess.destroy_called is True

def test_debug_run_and_debug_run_one(monkeypatch):
    # patch minimal components to test debug_run and debug_run_one
    class DummyRunner:
        def __init__(self, report):
            pass
        def run(self, tests):
            return 'run_called'

    monkeypatch.setattr(tc, "TestCase", tc.TestCase)  # ensure reference
    monkeypatch.setattr(tc, "datadrive", datadrive)
    monkeypatch.setattr(tc, "TestRunner", DummyRunner)
    monkeypatch.setattr(tc, "StreamTestReport", lambda **kwargs: "report")
    monkeypatch.setattr(tc, "EmptyTestReport", lambda factory: "empty")
    monkeypatch.setattr(tc, "StreamResult", lambda: "result")
    monkeypatch.setattr(datadrive, "is_datadrive", lambda cls: True)
    monkeypatch.setattr(datadrive, "load_datadrive_tests", lambda cls, name=None: [tc.TestCase(), tc.TestCase()])

    t = tc.TestCase()
    res = t.debug_run()
    assert res == 'run_called'

    res2 = t.debug_run_one()
    assert res2 == 'run_called'

    # debug_run_one with name parameter
    res3 = t.debug_run_one(name='non-existent')
    assert res3 == 'run_called'

def test_debug_run_one_fails_when_not_datadrive(monkeypatch):
    class NotDatadriveTest(tc.TestCase): pass
    monkeypatch.setattr(datadrive, "is_datadrive", lambda cls: False)
    monkeypatch.setattr(tc.settings, "DATA_DRIVE", False)
    # Instance with NotDatadriveTest type triggers RuntimeError
    t = NotDatadriveTest()
    with pytest.raises(RuntimeError):
        t.debug_run_one()

# --- Tests for modify_settings context manager ---

def test_modify_settings_temporarily_changes_and_restores(monkeypatch):
    from testbase.conf import settings

    orig_value = getattr(settings, "DATA_DRIVE", None)

    with modify_settings(DATA_DRIVE=not orig_value):
        assert getattr(settings, "DATA_DRIVE") == (not orig_value)

    assert getattr(settings, "DATA_DRIVE") == orig_value

def test_modify_settings_add_and_remove_key(monkeypatch):
    from testbase.conf import settings
    # Add a key not present before then restore
    key = "NEW_SETTING_FOR_TEST"
    if hasattr(settings, key):
        delattr(settings, key)
    with modify_settings(**{key: 123}):
        assert getattr(settings, key) == 123
    assert not hasattr(settings, key)

def test_modify_settings_preserves_old_value(monkeypatch):
    from testbase.conf import settings
    key = "DATA_DRIVE"
    old_val = getattr(settings, key)
    with modify_settings(**{key: not old_val}):
        assert getattr(settings, key) != old_val
    assert getattr(settings, key) == old_val

def test_modify_settings_sets_and_restores_sealed(monkeypatch):
    from testbase.conf import settings
    orig_sealed = settings._Settings__sealed
    with modify_settings(DATA_DRIVE=True):
        assert not settings._Settings__sealed
    assert settings._Settings__sealed == orig_sealed

# -- End of tests --
