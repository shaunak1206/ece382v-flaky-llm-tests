import sys
import types

import pytest

from testbase import datadrive as ddmod
from testbase import testcase as tcmod
from testbase.test import modify_settings


class DummyResult:
    def __init__(self):
        self.calls = []

    def begin_step(self, msg):
        self.calls.append(("begin_step", msg))

    def info(self, msg):
        self.calls.append(("info", msg))

    def error(self, msg):
        self.calls.append(("error", msg))

    def log_record(self, level, msg, extra=None):
        self.calls.append(("log_record", level, msg, extra))


class DummySession:
    def __init__(self):
        self.destroyed = False

    def destroy(self):
        self.destroyed = True


class DummyResMgr:
    def __init__(self):
        self.created = 0
        self.session = DummySession()

    def create_session(self, testcase):
        self.created += 1
        self.testcase = testcase
        return self.session


def _make_case_class(name="MyCase", module_name=__name__, **attrs):
    namespace = {
        "__module__": module_name,
        "__doc__": attrs.pop("__doc__", "case doc"),
        "owner": attrs.pop("owner", "foo"),
        "priority": attrs.pop("priority", tcmod.TestCase.EnumPriority.Normal),
        "status": attrs.pop("status", tcmod.TestCase.EnumStatus.Ready),
        "timeout": attrs.pop("timeout", 1),
        "runTest": lambda self: None,
    }
    namespace.update(attrs)
    return type(name, (tcmod.TestCase,), namespace)


def test_datadrive_decorator_marks_testcase_class():
    @ddmod.DataDrive([1, 2, 3])
    class C(tcmod.TestCase):
        owner = "foo"
        priority = tcmod.TestCase.EnumPriority.Normal
        status = tcmod.TestCase.EnumStatus.Ready
        timeout = 1

    assert ddmod.is_datadrive(C)
    assert isinstance(ddmod.get_datadrive(C), ddmod.DataDrive)


def test_datadrive_decorator_rejects_non_testcase():
    with pytest.raises(TypeError):
        ddmod.DataDrive([1])(object)


def test_datadrive_iter_from_list_and_len():
    drive = ddmod.DataDrive([10, 20, 30])
    assert list(iter(drive)) == [0, 1, 2]
    assert len(drive) == 3
    assert drive[1] == 20


def test_datadrive_iter_from_generator_consumes_to_list():
    drive = ddmod.DataDrive((x for x in [7, 8]))
    assert list(iter(drive)) == [0, 1]
    # generator should have been consumed and converted to a list
    assert len(drive) == 2
    assert drive[0] == 7
    assert drive[1] == 8


def test_load_datadrive_tests_from_list_uses_index_names():
    @ddmod.DataDrive(["a", "b"])
    class C(tcmod.TestCase):
        owner = "foo"
        priority = tcmod.TestCase.EnumPriority.Normal
        status = tcmod.TestCase.EnumStatus.Ready
        timeout = 1

    tests = ddmod.load_datadrive_tests(C)
    assert [t.casedata for t in tests] == ["a", "b"]
    assert [t.casedataname for t in tests] == ["0", "1"]
    assert all(isinstance(t, C) for t in tests)


def test_load_datadrive_tests_from_dict_preserves_keys_and_attrs():
    @ddmod.DataDrive({"k1": {"value": 1, "__attrs__": {"timeout": 9, "tags": ["x"], "__doc__": "override"}}})
    class C(tcmod.TestCase):
        owner = "foo"
        priority = tcmod.TestCase.EnumPriority.Normal
        status = tcmod.TestCase.EnumStatus.Ready
        timeout = 1
        tags = {"base"}
        __doc__ = "base doc"

    tests = ddmod.load_datadrive_tests(C)
    assert len(tests) == 1
    t = tests[0]
    assert t.casedata == {"value": 1, "__attrs__": {"timeout": 9, "tags": ["x"], "__doc__": "override"}}
    assert t.casedataname == "k1"
    assert t.timeout == 9
    assert t.tags == {"base", "x"}
    assert t.test_doc == "override"


def test_load_datadrive_tests_warns_on_bad_key(monkeypatch):
    @ddmod.DataDrive({"bad/key": 1})
    class C(tcmod.TestCase):
        owner = "foo"
        priority = tcmod.TestCase.EnumPriority.Normal
        status = tcmod.TestCase.EnumStatus.Ready
        timeout = 1

    warns = []
    monkeypatch.setattr(ddmod.logger, "warn", lambda msg: warns.append(msg))
    tests = ddmod.load_datadrive_tests(C)
    assert len(tests) == 1
    assert tests[0].casedataname == "bad_key"
    assert warns and "bad/key" in warns[0] and "bad_key" in warns[0]


def test_load_datadrive_tests_name_lookup_by_exact_key():
    @ddmod.DataDrive({"alpha": 1, "beta": 2})
    class C(tcmod.TestCase):
        owner = "foo"
        priority = tcmod.TestCase.EnumPriority.Normal
        status = tcmod.TestCase.EnumStatus.Ready
        timeout = 1

    tests = ddmod.load_datadrive_tests(C, "beta")
    assert len(tests) == 1
    assert tests[0].casedata == 2
    assert tests[0].casedataname == "beta"


def test_load_datadrive_tests_name_lookup_by_translated_key():
    @ddmod.DataDrive({"bad_key": 123})
    class C(tcmod.TestCase):
        owner = "foo"
        priority = tcmod.TestCase.EnumPriority.Normal
        status = tcmod.TestCase.EnumStatus.Ready
        timeout = 1

    tests = ddmod.load_datadrive_tests(C, "bad/key")
    assert len(tests) == 1
    assert tests[0].casedata == 123
    assert tests[0].casedataname == "bad/key"


def test_load_datadrive_tests_requires_datadrive_setting_for_plain_class():
    class C(tcmod.TestCase):
        owner = "foo"
        priority = tcmod.TestCase.EnumPriority.Normal
        status = tcmod.TestCase.EnumStatus.Ready
        timeout = 1

        def runTest(self):
            pass

    with modify_settings(DATA_DRIVE=False):
        with pytest.raises(RuntimeError, match="DATA_DRIVE is not set to True"):
            ddmod.load_datadrive_tests(C)


def test_testcase_properties_and_environ(monkeypatch):
    monkeypatch.setenv("QTAF_FOO", "bar")
    C = _make_case_class()
    t = C({"x": 1}, "d1")
    assert t.casedata == {"x": 1}
    assert t.casedataname == "d1"
    assert t.test_class_name.endswith(".MyCase")
    assert t.test_name.endswith("/d1")
    assert t.test_doc == "case doc"
    assert t.environ["FOO"] == "bar"


def test_testcase_test_doc_override_from_attrs():
    C = _make_case_class()
    t = C(attrs=None)
    assert t.test_doc == "case doc"

    t2 = C(attrs=None)
    t2 = C(testdata=None, testdataname=None, attrs={"__doc__": "override doc"})
    assert t2.test_doc == "override doc"


def test_testcase_test_resmgr_session_created_lazily():
    C = _make_case_class()
    t = C()
    resmgr = DummyResMgr()
    t.test_resmgr = resmgr
    s1 = t.test_resources
    s2 = t.test_resources
    assert s1 is s2
    assert resmgr.created == 1
    assert isinstance(s1, DummySession)


def test_testcase_clean_test_destroys_resources():
    C = _make_case_class()
    t = C()
    resmgr = DummyResMgr()
    t.test_resmgr = resmgr
    _ = t.test_resources
    t.clean_test()
    assert resmgr.session.destroyed is True


def test_testcase_logging_methods_convert_non_text():
    C = _make_case_class()
    t = C()
    tr = DummyResult()
    t.init_test(tr)
    t.start_step(123)
    t.log_info(456)
    t.fail(789)
    assert tr.calls == [
        ("begin_step", "123"),
        ("info", "456"),
        ("error", "789"),
    ]


def test_assert_equal_and_match_record_failure(monkeypatch):
    C = _make_case_class()
    t = C()
    tr = DummyResult()
    t.init_test(tr)

    assert t.assert_equal("m1", "abc", "abc") is True
    assert t.assert_equal("m2", "abc", "def") is False
    assert t.assert_match("m3", "hello", r"he.*o") is True
    assert t.assert_match("m4", "hello", r"^zz") is False

    assert any(call[0] == "log_record" and call[1] == tcmod.EnumLogLevel.ASSERT for call in tr.calls)


def test_assert_continue_setting_controls_assert_raises():
    C = _make_case_class()
    t = C()
    tr = DummyResult()
    t.init_test(tr)

    with modify_settings(QTAF_ASSERT_CONTINUE=True):
        t.assert_( "fail", False)
        assert any(call[0] == "log_record" for call in tr.calls)

    with modify_settings(QTAF_ASSERT_CONTINUE=False):
        with pytest.raises(RuntimeError, match="testcase assert failed:boom"):
            t._log_assert_failed("boom")


def test_testcase_type_merges_tags_and_module_tags():
    mod_name = "tmp_qtaf_mod"
    mod = types.ModuleType(mod_name)
    mod.__qtaf_tags__ = ["modtag", "shared"]
    sys.modules[mod_name] = mod
    try:
        class Base(tcmod.TestCase):
            __module__ = mod_name
            owner = "foo"
            priority = tcmod.TestCase.EnumPriority.Normal
            status = tcmod.TestCase.EnumStatus.Ready
            timeout = 1
            tags = ["base", "shared"]

        class Child(Base):
            __module__ = mod_name
            tags = "child"

        assert Base.tags == {"base", "shared", "modtag"}
        assert Child.tags == {"base", "shared", "modtag", "child"}
        assert Base.base_tags == {"modtag"}
        assert Child.base_tags == {"base", "shared", "modtag"}
    finally:
        sys.modules.pop(mod_name, None)
