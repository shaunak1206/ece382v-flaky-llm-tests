import pytest
from unittest import mock

from testbase import testcase as tc
from testbase.conf import settings
from testbase.datadrive import DataDrive, get_datadrive, is_datadrive, load_datadrive_tests
from testbase.test import modify_settings
from testbase.testresult import EnumLogLevel


class FakeResult(object):
    """Minimal stand-in for a TestResult, recording calls made by TestCase."""

    def __init__(self):
        self.records = []
        self.infos = []
        self.errors = []
        self.steps = []

    def log_record(self, level, message, extra=None):
        self.records.append((level, message))

    def info(self, message):
        self.infos.append(message)

    def error(self, message):
        self.errors.append(message)

    def begin_step(self, stepinfo):
        self.steps.append(stepinfo)


# ---------------------------------------------------------------------------
# DataDrive
# ---------------------------------------------------------------------------

def test_datadrive_sequence_len_getitem_and_iter():
    for data in ([10, 20, 30], (10, 20, 30)):
        dd = DataDrive(data)
        assert len(dd) == 3
        assert dd[0] == 10
        assert dd[2] == 30
        # Sequences iterate over indices, not values.
        assert list(dd) == [0, 1, 2]


def test_datadrive_dict_iter_yields_keys():
    dd = DataDrive({'TEST1': 1, 'TEST2': 2})
    assert sorted(dd) == ['TEST1', 'TEST2']
    assert dd['TEST2'] == 2
    assert len(dd) == 2


def test_datadrive_generator_is_materialized():
    dd = DataDrive(i for i in range(3))
    assert list(dd) == [0, 1, 2]
    # After iteration the generator has been converted to a list.
    assert len(dd) == 3
    assert dd[2] == 2


def test_datadrive_decorator_marks_testcase_class():
    @DataDrive([1, 2])
    class Marked(tc.TestCase):
        def run_test(self):
            pass

    assert is_datadrive(Marked)
    assert isinstance(get_datadrive(Marked), DataDrive)
    assert len(get_datadrive(Marked)) == 2


def test_datadrive_decorator_rejects_non_testcase():
    class Plain(object):
        pass

    with pytest.raises(TypeError):
        DataDrive([1])(Plain)


def test_load_datadrive_tests_list_uses_indices_as_names():
    @DataDrive(['a', 'b'])
    class ListDriven(tc.TestCase):
        def run_test(self):
            pass

    tests = load_datadrive_tests(ListDriven)
    assert [t.casedataname for t in tests] == ['0', '1']
    assert [t.casedata for t in tests] == ['a', 'b']
    assert all(isinstance(t, ListDriven) for t in tests)


def test_load_datadrive_tests_dict_with_attrs_override():
    @DataDrive({
        'A': {'__attrs__': {'owner': 'alice', 'timeout': 3}, 'value': 1},
        'B': 2,
    })
    class DictDriven(tc.TestCase):
        def run_test(self):
            pass

    by_name = {t.casedataname: t for t in load_datadrive_tests(DictDriven)}
    assert set(by_name) == {'A', 'B'}
    assert by_name['A'].owner == 'alice'
    assert by_name['A'].timeout == 3
    # __attrs__ is only extracted, the rest of the dict stays as casedata.
    assert by_name['A'].casedata == {'__attrs__': {'owner': 'alice', 'timeout': 3}, 'value': 1}
    assert by_name['B'].owner is None
    assert by_name['B'].casedata == 2


def test_load_datadrive_tests_filters_by_name():
    @DataDrive({'A': 1, 'B': 2})
    class Filtered(tc.TestCase):
        def run_test(self):
            pass

    tests = load_datadrive_tests(Filtered, 'B')
    assert len(tests) == 1
    assert tests[0].casedataname == 'B'
    assert tests[0].casedata == 2


def test_load_datadrive_tests_unknown_name_raises():
    @DataDrive({'A': 1})
    class Unknown(tc.TestCase):
        def run_test(self):
            pass

    with pytest.raises(ValueError):
        load_datadrive_tests(Unknown, 'missing')


def test_load_datadrive_tests_plain_class_requires_setting():
    class NotDriven(tc.TestCase):
        def run_test(self):
            pass

    with modify_settings(DATA_DRIVE=False):
        with pytest.raises(RuntimeError):
            load_datadrive_tests(NotDriven)


# ---------------------------------------------------------------------------
# TestCase
# ---------------------------------------------------------------------------

def test_testcase_default_state():
    case = tc.TestCase()
    assert case.casedata is None
    assert case.casedataname is None
    assert case.test_result is None
    assert case.test_resmgr is None
    assert case.test_extra_info == {}
    assert case.test_name == case.test_class_name
    assert case.test_class_name == tc.TestCase.__module__ + '.TestCase'


def test_testcase_test_name_includes_casedataname():
    case = tc.TestCase(testdata=[1], testdataname='0')
    assert case.casedata == [1]
    assert case.casedataname == '0'
    assert case.test_name == tc.TestCase.__module__ + '.TestCase/0'


def test_testcase_doc_is_stripped():
    class Documented(tc.TestCase):
        """
        Hello world
        """

    assert Documented().test_doc == 'Hello world'


def test_testcase_attrs_whitelist_is_applied():
    case = tc.TestCase(attrs={
        'owner': 'bob',
        'priority': 'High',
        'status': 'Ready',
        'timeout': 7,
        'not_allowed': 1,
        '__doc__': 'custom doc',
    })
    assert case.owner == 'bob'
    assert case.priority == 'High'
    assert case.status == 'Ready'
    assert case.timeout == 7
    assert case.test_doc == 'custom doc'
    assert not hasattr(case, 'not_allowed')


def test_tags_are_normalized_and_inherited():
    class Parent(tc.TestCase):
        tags = ['a']

    class Child(Parent):
        tags = 'b'

    class Plain(tc.TestCase):
        pass

    class Single(tc.TestCase):
        tags = 'smoke'

    assert Child.tags == {'a', 'b'}
    assert Child.base_tags == {'a'}
    assert Plain.tags == set()
    assert Single.tags == {'smoke'}

    inst = Single(attrs={'tags': 'nightly'})
    assert inst.tags == {'nightly'}


def test_metaclass_forbids_init_override():
    with pytest.raises(RuntimeError):
        class Bad(tc.TestCase):
            def __init__(self):
                pass


def test_run_test_not_implemented_on_base_class():
    with pytest.raises(NotImplementedError):
        tc.TestCase().run_test()
    # The legacy alias points to the same behaviour.
    with pytest.raises(NotImplementedError):
        tc.TestCase().runTest()


def test_assert_equal_failure_records_log_and_returns_false():
    result = FakeResult()
    case = tc.TestCase()
    case.init_test(result)

    assert case.assert_equal('equal', 1, 1) is True
    assert result.records == []

    assert case.assert_equal('not equal', 'a', 'b') is False
    assert len(result.records) == 1
    level, message = result.records[0]
    assert level == EnumLogLevel.ASSERT
    assert 'not equal' in message


def test_assert_failure_raises_when_not_continuing():
    result = FakeResult()
    case = tc.TestCase()
    case.init_test(result)

    # A true value does not log anything.
    case.assert_('should hold', True)
    assert result.records == []

    with modify_settings(QTAF_ASSERT_CONTINUE=False):
        with pytest.raises(RuntimeError):
            case.assert_('must hold', False)
    assert result.records[-1][0] == EnumLogLevel.ASSERT


def test_fail_log_info_and_start_step_forward_to_result():
    result = FakeResult()
    case = tc.TestCase()
    case.init_test(result)

    case.fail(42)
    case.log_info(['x'])
    case.start_step('step one')

    assert result.errors == ['42']
    assert result.infos == ["['x']"]
    assert result.steps == ['step one']


# ---------------------------------------------------------------------------
# TestCaseRunner / RepeatTestCaseRunner
# ---------------------------------------------------------------------------

def test_testcase_runner_executes_lifecycle_hooks():
    calls = []

    class LifecycleCase(tc.TestCase):
        """lifecycle case"""
        owner = 'foo'
        priority = tc.TestCase.EnumPriority.Normal
        status = tc.TestCase.EnumStatus.Ready
        timeout = 1

        def pre_test(self):
            calls.append('pre_test')

        def run_test(self):
            calls.append('run_test')

        def post_test(self):
            calls.append('post_test')

    factory = mock.Mock()
    result = factory.create.return_value

    with modify_settings(QTAF_REWRITE_ASSERT=False):
        returned = tc.TestCaseRunner().run(LifecycleCase(), factory)

    assert calls == ['pre_test', 'run_test', 'post_test']
    assert returned is result
    result.begin_test.assert_called_once()
    result.end_test.assert_called_once()


def test_repeat_runner_requires_repeat_attribute():
    class NoRepeat(tc.TestCase):
        """no repeat attribute"""
        owner = 'foo'
        priority = tc.TestCase.EnumPriority.Normal
        status = tc.TestCase.EnumStatus.Ready
        timeout = 1

        def run_test(self):
            pass

    factory = mock.Mock()
    result = factory.create.return_value

    returned = tc.RepeatTestCaseRunner().run(NoRepeat(), factory)

    assert returned is result
    result.error.assert_called_once()
    assert 'repeat' in result.error.call_args[0][0]


# ---------------------------------------------------------------------------
# modify_settings
# ---------------------------------------------------------------------------

def test_modify_settings_restores_nested_existing_value():
    key = 'QTAF_UNITTEST_NESTED_KEY'
    missing = object()

    with modify_settings(**{key: 'outer'}):
        with modify_settings(**{key: 'inner'}):
            assert getattr(settings, key) == 'inner'
        assert getattr(settings, key) == 'outer'

    assert getattr(settings, key, missing) is missing


def test_modify_settings_adds_and_removes_new_key_on_exception():
    key = 'QTAF_UNITTEST_TEMP_KEY'
    missing = object()

    with pytest.raises(ValueError):
        with modify_settings(**{key: 'tmp'}):
            assert getattr(settings, key) == 'tmp'
            assert key in settings._Settings__keys
            raise ValueError('boom')

    assert getattr(settings, key, missing) is missing
    assert key not in settings._Settings__keys
