# -*- coding: utf-8 -*-
import pytest

from testbase import datadrive
from testbase import testcase as tc
from testbase.conf import settings
from testbase.test import modify_settings
from testbase.testresult import EnumLogLevel


_TEMP_KEY = "QTAF_UNITTEST_TEMP_SETTING"


class _FakeResult(object):
    """Minimal stand-in for a TestResult that records calls."""

    def __init__(self):
        self.records = []
        self.errors = []
        self.infos = []
        self.steps = []

    def log_record(self, level, msg, extra=None):
        self.records.append((level, msg))

    def error(self, msg):
        self.errors.append(msg)

    def info(self, msg):
        self.infos.append(msg)

    def begin_step(self, step):
        self.steps.append(step)


@pytest.fixture
def fake_result():
    return _FakeResult()


@datadrive.DataDrive([10, 20, 30])
class ListDriven(tc.TestCase):
    def run_test(self):
        pass


@datadrive.DataDrive({"CASE_A": 1, "CASE_B": 2})
class DictDriven(tc.TestCase):
    def run_test(self):
        pass


@datadrive.DataDrive({"CASE": {"__attrs__": {"owner": "alice"}, "value": 1}})
class AttrDriven(tc.TestCase):
    def run_test(self):
        pass


class TagParent(tc.TestCase):
    tags = ["alpha"]


class TagChild(TagParent):
    tags = "beta"


# ---------------------------------------------------------------------------
# DataDrive
# ---------------------------------------------------------------------------

def test_datadrive_list_and_tuple_iterate_indices():
    dd = datadrive.DataDrive(["x", "y"])
    assert list(dd) == [0, 1]
    assert len(dd) == 2
    assert dd[1] == "y"

    tuple_dd = datadrive.DataDrive(("p",))
    assert list(tuple_dd) == [0]


def test_datadrive_dict_iterates_keys():
    dd = datadrive.DataDrive({"a": 1, "b": 2})
    assert sorted(dd) == ["a", "b"]
    assert dd["b"] == 2
    assert len(dd) == 2


def test_datadrive_generator_is_materialized_on_iteration():
    dd = datadrive.DataDrive(x * 10 for x in range(3))
    assert list(dd) == [0, 1, 2]
    assert len(dd) == 3
    assert dd[2] == 20


def test_datadrive_decorator_marks_testcase_subclass():
    class Local(tc.TestCase):
        def run_test(self):
            pass

    dd = datadrive.DataDrive([1])
    assert dd(Local) is Local
    assert datadrive.is_datadrive(Local)
    assert datadrive.get_datadrive(Local) is dd
    assert not datadrive.is_datadrive(tc.TestCase)


def test_datadrive_decorator_rejects_non_testcase():
    class Plain(object):
        pass

    with pytest.raises(TypeError):
        datadrive.DataDrive([1])(Plain)


# ---------------------------------------------------------------------------
# load_datadrive_tests
# ---------------------------------------------------------------------------

def test_load_list_driven_creates_one_instance_per_item():
    tests = datadrive.load_datadrive_tests(ListDriven)
    assert len(tests) == 3
    assert [t.casedata for t in tests] == [10, 20, 30]
    assert [t.casedataname for t in tests] == ["0", "1", "2"]
    assert all(isinstance(t, ListDriven) for t in tests)


def test_load_dict_driven_uses_keys_as_names():
    tests = datadrive.load_datadrive_tests(DictDriven)
    by_name = dict((t.casedataname, t.casedata) for t in tests)
    assert by_name == {"CASE_A": 1, "CASE_B": 2}


def test_load_with_integer_name_selects_single_item():
    tests = datadrive.load_datadrive_tests(ListDriven, 1)
    assert len(tests) == 1
    assert tests[0].casedata == 20
    assert tests[0].casedataname == "1"


def test_load_with_string_name_matches_translated_item():
    tests = datadrive.load_datadrive_tests(ListDriven, "2")
    assert len(tests) == 1
    assert tests[0].casedata == 30
    assert tests[0].casedataname == "2"


def test_load_unknown_name_raises_value_error():
    with pytest.raises(ValueError):
        datadrive.load_datadrive_tests(ListDriven, "does_not_exist")


def test_load_applies_dunder_attrs_from_data():
    tests = datadrive.load_datadrive_tests(AttrDriven)
    assert len(tests) == 1
    assert tests[0].owner == "alice"
    assert tests[0].casedataname == "CASE"


def test_load_non_datadrive_class_requires_data_drive_setting():
    class Plain(tc.TestCase):
        def run_test(self):
            pass

    with modify_settings(DATA_DRIVE=False):
        with pytest.raises(RuntimeError):
            datadrive.load_datadrive_tests(Plain)


# ---------------------------------------------------------------------------
# TestCase
# ---------------------------------------------------------------------------

def test_testcase_init_stores_casedata_and_name():
    case = tc.TestCase(testdata={"k": 1}, testdataname="k")
    assert case.casedata == {"k": 1}
    assert case.casedataname == "k"
    assert case.test_result is None


def test_testcase_init_applies_only_whitelisted_attrs():
    case = tc.TestCase(attrs={
        "owner": "bob",
        "priority": "High",
        "timeout": 3,
        "not_allowed": 1,
    })
    assert case.owner == "bob"
    assert case.priority == "High"
    assert case.timeout == 3
    assert not hasattr(case, "not_allowed")


def test_testcase_test_name_includes_casedataname():
    class NamedCase(tc.TestCase):
        def run_test(self):
            pass

    named = NamedCase(testdataname="CASE_A")
    assert named.test_name == named.test_class_name + "/CASE_A"

    plain = NamedCase()
    assert plain.test_name == plain.test_class_name


def test_metaclass_merges_tags_from_base_class():
    assert TagParent.tags == {"alpha"}
    assert TagParent.base_tags == set()
    assert TagChild.tags == {"alpha", "beta"}
    assert TagChild.base_tags == {"alpha"}


def test_metaclass_forbids_overriding_init():
    with pytest.raises(RuntimeError):
        class BadCase(tc.TestCase):
            def __init__(self):
                pass


def test_run_test_base_raises_not_implemented_and_alias_matches():
    with pytest.raises(NotImplementedError):
        tc.TestCase().run_test()
    assert tc.TestCase.runTest is tc.TestCase.run_test


def test_assert_equal_returns_bool_and_records_mismatch(fake_result):
    case = tc.TestCase()
    case.init_test(fake_result)

    assert case.assert_equal("equal values", 1, 1) is True
    assert fake_result.records == []

    assert case.assert_equal("different values", "a", "b") is False
    assert len(fake_result.records) == 1
    assert fake_result.records[0][0] == EnumLogLevel.ASSERT


def test_assert_raises_when_assert_continue_disabled(fake_result):
    case = tc.TestCase()
    case.init_test(fake_result)

    case.assert_("truthy", True)
    assert fake_result.records == []

    with modify_settings(QTAF_ASSERT_CONTINUE=False):
        with pytest.raises(RuntimeError):
            case.assert_("must hold", False)
    assert len(fake_result.records) == 1


# ---------------------------------------------------------------------------
# modify_settings
# ---------------------------------------------------------------------------

def test_modify_settings_overrides_and_restores_nested_values():
    with modify_settings(**{_TEMP_KEY: "outer"}):
        assert getattr(settings, _TEMP_KEY) == "outer"
        with modify_settings(**{_TEMP_KEY: "inner"}):
            assert getattr(settings, _TEMP_KEY) == "inner"
        assert getattr(settings, _TEMP_KEY) == "outer"
    assert not hasattr(settings, _TEMP_KEY)
    assert _TEMP_KEY not in settings._Settings__keys


def test_modify_settings_adds_and_removes_new_key():
    assert not hasattr(settings, _TEMP_KEY)
    with modify_settings(**{_TEMP_KEY: 1}):
        assert getattr(settings, _TEMP_KEY) == 1
        assert _TEMP_KEY in settings._Settings__keys
    assert not hasattr(settings, _TEMP_KEY)
    assert _TEMP_KEY not in settings._Settings__keys


def test_modify_settings_restores_state_on_exception():
    with pytest.raises(ValueError):
        with modify_settings(**{_TEMP_KEY: "x"}):
            raise ValueError("boom")
    assert not hasattr(settings, _TEMP_KEY)
    assert _TEMP_KEY not in settings._Settings__keys


def test_modify_settings_unseals_inside_and_restores_seal_after():
    settings._Settings__ensure_loaded()
    before = settings._Settings__sealed
    with modify_settings(**{_TEMP_KEY: 1}):
        assert settings._Settings__sealed is False
    assert settings._Settings__sealed == before
