# test_datadrive_testcase_test.py
import pytest
import types
import re
import sys
import threading
import builtins

import six
from testbase import datadrive, testcase, test


# Utilities for some tests
class DummyResult:
    def __init__(self):
        self.records = []
        self.passed = True

    def begin_step(self, stepinfo):
        self.records.append(('begin_step', stepinfo))

    def info(self, info):
        self.records.append(('info', info))

    def error(self, msg):
        self.records.append(('error', msg))
        self.passed = False

    def log_record(self, level, msg, kwargs=None):
        self.records.append(('log_record', level, msg, kwargs))

    def exception(self, msg):
        self.records.append(('exception', msg))

    def begin_test(self, testcase):
        self.records.append(('begin_test', testcase))

    def end_test(self):
        self.records.append(('end_test', None))


class DummyTestResultFactory:
    def create(self, testcase):
        return DummyResult()


# ------- Tests for DataDrive in datadrive.py -------

def test_datadrive_decorator_sets_attribute():
    class TC(testcase.TestCase):
        pass

    data = [1, 2, 3]
    decorated = datadrive.DataDrive(data)(TC)
    assert hasattr(decorated, '__qtaf_datadrive__')
    assert decorated.__qtaf_datadrive__._case_data == data


def test_datadrive_decorator_raises_for_non_testcase():
    with pytest.raises(TypeError):
        @datadrive.DataDrive([1,2,3])
        class NotATest(object):
            pass


def test_datadrive_iter_and_len_list_and_dict():
    dd_list = datadrive.DataDrive([10, 20])
    assert list(dd_list) == [0, 1]
    assert len(dd_list) == 2
    dd_dict = datadrive.DataDrive({'a': 1, 'b': 2})
    keys = list(dd_dict)
    assert 'a' in keys and 'b' in keys
    assert len(dd_dict) == 2


def test_datadrive_getitem_returns_data():
    dd = datadrive.DataDrive({'foo': 'bar', 5: 'baz'})
    assert dd['foo'] == 'bar'
    assert dd[5] == 'baz'


def test_is_datadrive_and_get_datadrive(monkeypatch):
    class MyTC(testcase.TestCase):
        pass
    decorated = datadrive.DataDrive([1])
    monkeypatch.setattr(MyTC, '__qtaf_datadrive__', decorated)
    assert datadrive.is_datadrive(MyTC)
    assert datadrive.get_datadrive(MyTC) is decorated
    # instance should work too
    instance = MyTC()
    instance.__qtaf_datadrive__ = decorated
    assert datadrive.is_datadrive(instance)
    assert datadrive.get_datadrive(instance) is decorated


def test_get_translated_in_datadrive_success(monkeypatch):
    dd = {'key1': 'val1', 'key-2': 'val2'}
    # The translate_bad_char replaces bad chars, so we simulate translation consistency
    # We test that requesting 'key1' returns 'val1'
    assert datadrive._get_translated_in_datadrive('key1', dd) == 'val1'
    # requesting with a string that is equal after translation returns correct value
    tkey = 'key-2'
    result = datadrive._get_translated_in_datadrive(tkey, dd)
    assert result == 'val2'


def test_get_translated_in_datadrive_raises_valueerror():
    dd = {'exist': 123}
    with pytest.raises(ValueError):
        datadrive._get_translated_in_datadrive('not_exist', dd)


def test_load_datadrive_tests_creates_testcases(monkeypatch):
    # Prepare a dummy testcase class with constructor accepting args
    calls = []

    class MyTest(testcase.TestCase):
        def __init__(self, testdata=None, testdataname=None, attrs=None):
            super().__init__(testdata, testdataname, attrs)
            calls.append((testdata, testdataname, attrs))

    # Decorate with DataDrive
    data = {'a': {'foo': 1, "__attrs__": {'priority': testcase.TestCase.EnumPriority.High}},
            'b': 'simple'}
    decorated = datadrive.DataDrive(data)(MyTest)

    # monkeypatch logger.warn so we can check warnings emitted
    warned = []
    monkeypatch.setattr(datadrive.logger, "warn", lambda msg: warned.append(msg))

    tests = datadrive.load_datadrive_tests(decorated)
    assert len(tests) == 2
    # The testcases created should reflect data passed
    found_a = any(t.casedataname == 'a' and t.casedata == data['a'] for t in tests)
    found_b = any(t.casedataname == 'b' and t.casedata == data['b'] for t in tests)
    assert found_a and found_b
    assert all(isinstance(t, MyTest) for t in tests)
    # Confirm that the attrs from '__attrs__' are assigned
    for t in tests:
        if isinstance(t.casedata, dict) and "__attrs__" in t.casedata:
            assert t.priority == testcase.TestCase.EnumPriority.High

    # keys without bad chars should not generate warnings
    assert not warned


def test_load_datadrive_tests_with_name_subset(monkeypatch):
    class MyTest(testcase.TestCase):
        pass

    data = {'key1': 'val1', 'key two': 'val2'}
    decorated = datadrive.DataDrive(data)(MyTest)

    warned = []
    monkeypatch.setattr(datadrive.logger, "warn", lambda msg: warned.append(msg))

    selected = datadrive.load_datadrive_tests(decorated, name='key1')
    # Only one test with name 'key1'
    assert len(selected) == 1
    assert selected[0].casedataname == 'key1'

    # test name that requires translation triggers warning
    selected2 = datadrive.load_datadrive_tests(decorated, name='key two')
    assert len(selected2) == 1
    assert re.search('WARNING', warned[-1], re.I)


def test_load_datadrive_tests_raises_runtimeerror_when_not_datadrive_and_settings_false(monkeypatch):
    monkeypatch.setattr(datadrive, "is_datadrive", lambda obj: False)
    monkeypatch.setattr(datadrive.settings, "DATA_DRIVE", False)
    with pytest.raises(RuntimeError):
        datadrive.load_datadrive_tests(testcase.TestCase)


def test_load_datadrive_tests_uses_loader_when_not_datadrive(monkeypatch):
    monkeypatch.setattr(datadrive, "is_datadrive", lambda obj: False)
    monkeypatch.setattr(datadrive.settings, "DATA_DRIVE", True)

    class DummyLoader:
        def load(self):
            return {'x': 'y'}

    dummy_loader = DummyLoader()
    # patching import inside function by patching datadrive.TestDataLoader to dummy_loader
    monkeypatch.setattr(datadrive, "TestDataLoader", lambda: dummy_loader)

    class MyTest(testcase.TestCase):
        def __init__(self, testdata=None, testdataname=None, attrs=None):
            super().__init__(testdata, testdataname, attrs)
            self.c = testdata

    tests = datadrive.load_datadrive_tests(MyTest)
    assert len(tests) == 1
    assert tests[0].casedata == 'y'


# ------- Tests for TestCase in testcase.py -------

def test_testcase_initialization_and_properties():
    attrs = {'priority': testcase.TestCase.EnumPriority.High,
             'status': testcase.TestCase.EnumStatus.Ready,
             'owner': 'owner1',
             'timeout': 30,
             '__doc__': 'docstring',
             'tags': ['tag1', 'tag2']}
    tc = testcase.TestCase(testdata=123, testdataname='name1', attrs=attrs)
    assert tc.casedata == 123
    assert tc.casedataname == 'name1'
    assert tc.priority == testcase.TestCase.EnumPriority.High
    assert tc.status == testcase.TestCase.EnumStatus.Ready
    assert tc.owner == 'owner1'
    assert tc.timeout == 30
    assert tc.test_doc == 'docstring'
    # tags should be a set including base_tags and provided tags
    assert isinstance(tc.tags, set)
    for tag in ['tag1', 'tag2']:
        assert tag in tc.tags


def test_testcase_test_name_and_class_name(monkeypatch):
    class DummyClass(testcase.TestCase):
        pass

    # When module != '__main__'
    dummy = DummyClass(testdata='data', testdataname='data_name')
    assert dummy.test_class_name.startswith(dummy.__class__.__module__)
    expected_name = '%s/%s' % (dummy.test_class_name, 'data_name')
    assert dummy.test_name == expected_name

    # When casedataname is None
    dummy2 = DummyClass()
    dummy2._TestCase__testdataname = None
    assert dummy2.test_name == dummy2.test_class_name


def test_testcase_test_doc_reads_class_docstring():
    class DocTest(testcase.TestCase):
        """some docstring"""

    inst = DocTest()
    assert 'some docstring' in inst.test_doc


def test_testcase_assert_equal_and_assert_match(monkeypatch):
    dummy = testcase.TestCase()
    dummy._TestCase__testresult = DummyResult()

    # Matching strings
    assert dummy.assert_equal('msg', 'abc', 'abc') is True

    # Not matching triggers False & record failure
    assert dummy.assert_equal('msg', 'abc', 'def') is False
    assert any('检查点不通过' in rec[2] for rec in dummy._TestCase__testresult.records)

    # Matching regex returns True
    assert dummy.assert_match('msg', 'abc123', r'^abc')

    # Not matching regex returns False and records failure
    dummy._TestCase__testresult.records.clear()
    assert dummy.assert_match('msg', 'xyz', r'^abc') is False
    assert any('检查点不通过' in rec[2] for rec in dummy._TestCase__testresult.records)


def test_testcase_assert_(raises=False):
    dummy = testcase.TestCase()
    dummy._TestCase__testresult = DummyResult()

    # Override setting to True to prevent RuntimeError
    from testbase import conf
    old = conf.settings.get('QTAF_ASSERT_CONTINUE', True)

    # Should raise if False
    with test.modify_settings(QTAF_ASSERT_CONTINUE=False):
        with pytest.raises(RuntimeError):
            dummy.assert_('fail message', False)

    # Should not raise if True
    with test.modify_settings(QTAF_ASSERT_CONTINUE=True):
        dummy.assert_('fail message', False)  # no exception

    # Passing assert does nothing
    dummy.assert_('fail message', True)


def test_testcase_wait_for_equal_and_wait_for_match(monkeypatch):
    class Obj:
        def __init__(self):
            self.status = 'ready'

    dummy = testcase.TestCase()
    dummy._TestCase__testresult = DummyResult()
    obj = Obj()

    # Wait_for_equal succeeds when attribute becomes expected
    obj.status = 'ok'
    result = dummy.wait_for_equal('msg', obj, 'status', 'ok', timeout=1, interval=0.1)
    assert result is True

    # Wait_for_equal fails when attribute doesn't become expected
    obj.status = 'no'
    result = dummy.wait_for_equal('msg', obj, 'status', 'expected', timeout=0.2, interval=0.1)
    assert result is False
    assert any('检查点不通过' in r[2] for r in dummy._TestCase__testresult.records)

    dummy._TestCase__testresult.records.clear()

    # wait_for_match success when matching regex
    obj.status = 'foobar'
    result = dummy.wait_for_match('msg', obj, 'status', r'foo.*', timeout=1, interval=0.1)
    assert result is True

    # wait_for_match fails when no match
    obj.status = 'barfoo'
    result = dummy.wait_for_match('msg', obj, 'status', r'^foo', timeout=0.2, interval=0.1)
    assert result is False
    assert any('检查点不通过' in r[2] for r in dummy._TestCase__testresult.records)


def test_testcase_fail_and_log_info(monkeypatch):
    dummy = testcase.TestCase()
    dummy._TestCase__testresult = DummyResult()

    dummy.log_info('info message')
    dummy.fail('fail message')

    logs = dummy._TestCase__testresult.records
    # Check that the info and error logs are present
    assert ('info', 'info message') in logs
    assert ('error', 'fail message') in logs


def test_testcase_start_step(monkeypatch):
    dummy = testcase.TestCase()
    dummy._TestCase__testresult = DummyResult()
    dummy.start_step('step 1')
    assert ('begin_step', 'step 1') in dummy._TestCase__testresult.records


def test_testcase_get_extra_fail_record_defaults():
    dummy = testcase.TestCase()
    logs, attachments = dummy.get_extra_fail_record()
    assert logs == {} and attachments == {}


def test_testcase_debug_run_and_debug_run_one(monkeypatch):
    # We test that debug_run and debug_run_one call runner.run and return its value

    class MyTest(testcase.TestCase):
        def run_test(self):
            return 'run_test called'

    # Patch imported modules in debug_run to dummy versions
    monkeypatch.setattr(testcase, 'datadrive', datadrive)
    monkeypatch.setattr(testcase, 'testbase', sys.modules.get('testbase', None))

    # Patch TestRunner and report classes to dummy ones that return dummy results
    class DummyRunner:
        def __init__(self, report=None):
            self.report = report

        def run(self, tests):
            return 'runner_run_called'

    monkeypatch.setattr(testcase, 'TestRunner', DummyRunner)
    monkeypatch.setattr(testcase, 'StreamTestReport', lambda **kwargs: 'report')
    monkeypatch.setattr(testcase, 'EmptyTestReport', lambda fn: 'empty_report')
    monkeypatch.setattr(testcase, 'StreamResult', lambda: 'stream_result')

    # Mock is_datadrive to True for class
    monkeypatch.setattr(datadrive, 'is_datadrive', lambda cls: True)

    instance = MyTest()
    result = instance.debug_run()
    assert result == 'runner_run_called'

    # debug_run_one with name
    result = instance.debug_run_one(name=None)
    assert result == 'runner_run_called'

    # debug_run_one raises when not datadrive and settings False
    monkeypatch.setattr(datadrive, 'is_datadrive', lambda cls: False)
    monkeypatch.setattr(testcase.settings, 'DATA_DRIVE', False)
    class NonDataDriveTest(testcase.TestCase):
        pass
    inst = NonDataDriveTest()
    with pytest.raises(RuntimeError):
        inst.debug_run_one()


# -------- Tests for modify_settings in test.py -----------

def test_modify_settings_changes_and_restores(monkeypatch):
    orig_value = getattr(test.settings, 'NEW_SETTING', None)
    with test.modify_settings(NEW_SETTING=123):
        assert getattr(test.settings, 'NEW_SETTING') == 123
    # After exit, original value restored (or attribute deleted)
    if orig_value is None:
        assert not hasattr(test.settings, 'NEW_SETTING')
    else:
        assert getattr(test.settings, 'NEW_SETTING') == orig_value


def test_modify_settings_adds_new_key_and_removes(monkeypatch):
    key = 'TEMP_NEW_KEY'
    if hasattr(test.settings, key):
        orig_val = getattr(test.settings, key)
        delattr(test.settings, key)
    else:
        orig_val = None

    with test.modify_settings(**{key: 'value'}):
        assert getattr(test.settings, key) == 'value'
    # Key removed after exit
    assert not hasattr(test.settings, key)

    if orig_val is not None:
        setattr(test.settings, key, orig_val)


def test_modify_settings_restores_value(monkeypatch):
    key = 'TEST_KEY'
    setattr(test.settings, key, 'original')
    with test.modify_settings(**{key: 'newval'}):
        assert getattr(test.settings, key) == 'newval'
    assert getattr(test.settings, key) == 'original'


# You can add more tests for modify_environ and modify_attributes if needed


def test_modify_attributes_basic():
    class Dummy:
        x = 1
        y = 2

    d = Dummy()
    with test.modify_attributes(d, {'x': 10, 'y': 20}):
        assert d.x == 10
        assert d.y == 20
    # Values restored
    assert d.x == 1
    assert d.y == 2


def test_modify_attributes_lock_released():
    class Dummy:
        a = 1

    d = Dummy()
    lock = threading.Lock()
    # The lock _lock attribute is class attribute, test it's locked/unlocked properly
    with test.modify_attributes(d, {'a': 100}):
        assert d.a == 100
        assert test.modify_attributes._lock.locked()
    # After context manager, lock released
    assert not test.modify_attributes._lock.locked()
    assert d.a == 1
