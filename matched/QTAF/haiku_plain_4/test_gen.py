# -*- coding: utf-8 -*-
import pytest

from testbase import datadrive
from testbase.conf import settings
from testbase.datadrive import DataDrive, get_datadrive, is_datadrive, load_datadrive_tests
from testbase.test import modify_settings
from testbase.testcase import Environ, TestCase
from testbase.testresult import EnumLogLevel


class _FakeResult(object):
    """Minimal stand-in for a TestResult used by TestCase assertion helpers."""

    def __init__(self):
        self.records = []
        self.errors = []
        self.infos = []
        self.steps = []

    def log_record(self, level, msg, *args, **kwargs):
        self.records.append((level, msg))

    def error(self, msg):
        self.errors.append(msg)

    def info(self, msg):
        self.infos.append(msg)

    def begin_step(self, msg):
        self.steps.append(msg)


@DataDrive([10, 20, 30])
class _ListDrivenCase(TestCase):
    """list driven case"""
    owner = 'tester'
    priority = TestCase.EnumPriority.Normal
    status = TestCase.EnumStatus.Ready
    timeout = 1

    def run_test(self):
        pass


@DataDrive({'TEST1': 1, 'TEST2': 2, 'TEST3': {'__attrs__': {'owner': 'bob'}, 'value': 3}})
class _DictDrivenCase(TestCase):
    """dict driven case"""
    owner = 'tester'
    priority = TestCase.EnumPriority.Normal
    status = TestCase.EnumStatus.Ready
    timeout = 1

    def run_test(self):
        pass


# ---------------------------------------------------------------- DataDrive

def test_datadrive_list_iter_yields_indices():
    dd = DataDrive(['a', 'b', 'c'])
    assert list(dd) == [0, 1, 2]


def test_datadrive_tuple_iter_yields_indices():
    dd = DataDrive(('x', 'y'))
    assert list(dd) == [0, 1]


def test_datadrive_dict_iter_yields_keys():
    dd = DataDrive({'k1': 1, 'k2': 2})
    assert sorted(dd) == ['k1', 'k2']


def test_datadrive_getitem_list_and_dict():
    assert DataDrive([5, 6])[1] == 6
    assert DataDrive({'a': 'alpha'})['a'] == 'alpha'


def test_datadrive_len():
    assert len(DataDrive([1, 2, 3, 4])) == 4
    assert len(DataDrive({'a': 1})) == 1


def test_datadrive_generator_is_materialized_on_iter():
    dd = DataDrive(x * 2 for x in [1, 2, 3])
    assert list(dd) == [0, 1, 2]
    # after materialization the data is a list and len works
    assert len(dd) == 3
    assert dd[2] == 6


def test_decorator_marks_testcase_subclass_and_returns_same_class():
    class Plain(TestCase):
        def run_test(self):
            pass

    dd = DataDrive([1])
    returned = dd(Plain)
    assert returned is Plain
    assert is_datadrive(Plain)
    assert get_datadrive(Plain) is dd


def test_decorator_rejects_non_testcase_class():
    class NotACase(object):
        pass

    with pytest.raises(TypeError):
        DataDrive([1])(NotACase)


def test_is_datadrive_false_for_plain_class():
    class Plain(TestCase):
        def run_test(self):
            pass

    assert not is_datadrive(Plain)


def test_load_list_driven_tests_creates_instances_with_index_names():
    tests = load_datadrive_tests(_ListDrivenCase)
    assert [t.casedata for t in tests] == [10, 20, 30]
    assert [t.casedataname for t in tests] == ['0', '1', '2']
    assert all(isinstance(t, _ListDrivenCase) for t in tests)


def test_load_dict_driven_tests_uses_keys_as_names():
    tests = load_datadrive_tests(_DictDrivenCase)
    names = sorted(t.casedataname for t in tests)
    assert names == ['TEST1', 'TEST2', 'TEST3']
    by_name = {t.casedataname: t.casedata for t in tests}
    assert by_name['TEST1'] == 1
    assert by_name['TEST2'] == 2


def test_load_datadrive_tests_by_name_selects_single_entry():
    tests = load_datadrive_tests(_DictDrivenCase, 'TEST2')
    assert len(tests) == 1
    assert tests[0].casedataname == 'TEST2'
    assert tests[0].casedata == 2


def test_load_datadrive_tests_by_index_name_for_list_data():
    tests = load_datadrive_tests(_ListDrivenCase, '1')
    assert len(tests) == 1
    assert tests[0].casedata == 20


def test_load_datadrive_tests_unknown_name_raises_value_error():
    with pytest.raises(ValueError):
        load_datadrive_tests(_DictDrivenCase, 'NO_SUCH_CASE')


def test_load_datadrive_tests_applies_attrs_from_testdata():
    tests = load_datadrive_tests(_DictDrivenCase, 'TEST3')
    assert len(tests) == 1
    assert tests[0].owner == 'bob'
    assert tests[0].casedata == {'__attrs__': {'owner': 'bob'}, 'value': 3}


def test_load_undecorated_class_without_data_drive_setting_raises():
    class Plain(TestCase):
        def run_test(self):
            pass

    with modify_settings(DATA_DRIVE=False):
        with pytest.raises(RuntimeError):
            load_datadrive_tests(Plain)


# ---------------------------------------------------------------- TestCase

def test_testcase_defaults():
    tc = TestCase()
    assert tc.casedata is None
    assert tc.casedataname is None
    assert tc.test_name == tc.test_class_name


def test_testcase_test_name_includes_datadrive_name():
    tests = load_datadrive_tests(_DictDrivenCase, 'TEST1')
    tc = tests[0]
    expected_cls = '%s.%s' % (_DictDrivenCase.__module__, _DictDrivenCase.__name__)
    assert tc.test_class_name == expected_cls
    assert tc.test_name == expected_cls + '/TEST1'


def test_testcase_forbids_overriding_init():
    with pytest.raises(RuntimeError):
        class Bad(TestCase):
            def __init__(self):
                pass


def test_testcase_tags_string_is_normalized_and_inherited():
    class Parent(TestCase):
        tags = 'alpha'

        def run_test(self):
            pass

    class Child(Parent):
        tags = 'beta'

        def run_test(self):
            pass

    assert Parent.tags == {'alpha'}
    assert Child.tags == {'alpha', 'beta'}
    assert Child.base_tags == {'alpha'}


def test_testcase_attrs_overwrite_whitelisted_attributes_only():
    tc = TestCase(attrs={'owner': 'carol', 'timeout': 7, 'not_allowed': 'x'})
    assert tc.owner == 'carol'
    assert tc.timeout == 7
    assert not hasattr(tc, 'not_allowed')


def test_testcase_attrs_doc_overrides_test_doc():
    tc = TestCase(attrs={'__doc__': 'custom doc'})
    assert tc.test_doc == 'custom doc'


def test_testcase_test_doc_strips_surrounding_whitespace():
    class Documented(TestCase):
        """   hello world   """

        def run_test(self):
            pass

    assert Documented().test_doc == 'hello world'


def test_testcase_test_extra_info_collects_defined_fields():
    class WithExtra(TestCase):
        test_extra_info_def = [('build', None), ('missing', None)]
        build = 42

        def run_test(self):
            pass

    assert WithExtra().test_extra_info == {'build': 42, 'missing': None}


def test_testcase_run_test_not_implemented_by_default():
    with pytest.raises(NotImplementedError):
        TestCase().run_test()


def test_testcase_assert_equal_true_and_false_paths():
    tc = TestCase()
    fake = _FakeResult()
    tc.init_test(fake)

    assert tc.assert_equal('same', 5, 5) is True
    assert fake.records == []

    assert tc.assert_equal('different', 5, 6) is False
    assert len(fake.records) == 1
    assert fake.records[0][0] == EnumLogLevel.ASSERT


def test_testcase_assert_true_only_logs_on_falsy_value():
    tc = TestCase()
    fake = _FakeResult()
    tc.init_test(fake)

    tc.assert_('ok', True)
    assert fake.records == []

    tc.assert_('bad', False)
    assert len(fake.records) == 1
    assert fake.records[0][0] == EnumLogLevel.ASSERT


def test_testcase_assert_raises_when_assert_continue_disabled():
    tc = TestCase()
    tc.init_test(_FakeResult())
    with modify_settings(QTAF_ASSERT_CONTINUE=False):
        with pytest.raises(RuntimeError):
            tc.assert_('boom', False)


def test_testcase_fail_and_log_info_forward_to_result_and_stringify():
    tc = TestCase()
    fake = _FakeResult()
    tc.init_test(fake)

    tc.fail(404)
    tc.log_info(3.5)
    tc.start_step(1)

    assert fake.errors == ['404']
    assert fake.infos == ['3.5']
    assert fake.steps == ['1']


def test_environ_is_singleton_dict_shared_by_testcases():
    tc1 = TestCase()
    tc2 = TestCase()
    assert isinstance(tc1.environ, Environ)
    assert tc1.environ is tc2.environ
    assert isinstance(tc1.environ, dict)


# ---------------------------------------------------------------- modify_settings

def test_modify_settings_temporarily_changes_existing_key_and_restores():
    original = getattr(settings, 'DATA_DRIVE', None)
    with modify_settings(DATA_DRIVE=not bool(original)):
        assert getattr(settings, 'DATA_DRIVE') == (not bool(original))
    assert getattr(settings, 'DATA_DRIVE', None) == original


def test_modify_settings_adds_new_key_inside_and_removes_after():
    key = 'QTAF_UNITTEST_TEMP_KEY_XYZ'
    assert not hasattr(settings, key)
    with modify_settings(**{key: 'temp'}):
        assert getattr(settings, key) == 'temp'
        assert key in settings._Settings__keys
    assert not hasattr(settings, key)
    assert key not in settings._Settings__keys


def test_modify_settings_restores_sealed_flag():
    before = settings._Settings__sealed
    with modify_settings(QTAF_UNITTEST_SEALED_CHECK=1):
        pass
    assert settings._Settings__sealed == before


def test_modify_settings_restores_value_when_body_raises():
    original = settings.get('QTAF_ASSERT_CONTINUE', True)
    with pytest.raises(ValueError):
        with modify_settings(QTAF_ASSERT_CONTINUE=not original):
            raise ValueError('inside')
    assert settings.get('QTAF_ASSERT_CONTINUE', True) == original


def test_modify_settings_nested_contexts_restore_in_order():
    original = settings.get('QTAF_ASSERT_CONTINUE', True)
    with modify_settings(QTAF_ASSERT_CONTINUE=False):
        assert settings.get('QTAF_ASSERT_CONTINUE', True) is False
        with modify_settings(QTAF_ASSERT_CONTINUE=True):
            assert settings.get('QTAF_ASSERT_CONTINUE', True) is True
        assert settings.get('QTAF_ASSERT_CONTINUE', True) is False
    assert settings.get('QTAF_ASSERT_CONTINUE', True) == original


def test_modify_settings_stores_kwargs_and_starts_with_empty_old_conf():
    ctx = modify_settings(FOO_KEY=1, BAR_KEY=2)
    assert ctx.new_conf == {'FOO_KEY': 1, 'BAR_KEY': 2}
    assert ctx.old_conf == {}
