import warnings
from typing import List

import pytest

import cynergy.container as container_module
from cynergy import attributes
from cynergy.attributes import LifeCycle
from cynergy.config import Plain
from cynergy.container import IocContainer


class Repo(object):
    pass


class SqlRepo(Repo):
    pass


class OtherRepo(Repo):
    pass


class Service(object):
    def __init__(self, repo: Repo):
        self.repo = repo


class WithDefault(object):
    def __init__(self, value=42):
        self.value = value


class NeedsStr(object):
    def __init__(self, name: str):
        self.name = name


@pytest.fixture
def ioc():
    return IocContainer(None)


@pytest.fixture
def global_container():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        container_module.initialize()
    yield
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        container_module.initialize()


def test_get_auto_creates_instance_of_unregistered_class(ioc):
    result = ioc.get(Repo)
    assert isinstance(result, Repo)


def test_get_default_lifecycle_is_singleton(ioc):
    first = ioc.get(Repo)
    second = ioc.get(Repo)
    assert first is second


def test_get_multi_instance_returns_new_instances(ioc):
    first = ioc.get(Repo, LifeCycle.MULTI_INSTANCE)
    second = ioc.get(Repo, LifeCycle.MULTI_INSTANCE)
    assert first is not second
    assert isinstance(first, Repo)
    assert isinstance(second, Repo)


def test_get_none_lifecycle_reads_class_attribute(ioc):
    class Transient(object):
        pass

    setattr(Transient, attributes.LIFECYCLE_ARGUMENT_NAME, LifeCycle.MULTI_INSTANCE)
    first = ioc.get(Transient, None)
    second = ioc.get(Transient, None)
    assert first is not second


def test_get_unsupported_lifecycle_raises(ioc):
    with pytest.raises(NotImplementedError):
        ioc.get(Repo, "not-a-lifecycle")


def test_get_resolves_constructor_dependency_by_annotation(ioc):
    ioc.register(Repo, SqlRepo)
    service = ioc.get(Service)
    assert isinstance(service.repo, SqlRepo)


def test_get_uses_default_value_for_argument(ioc):
    result = ioc.get(WithDefault)
    assert result.value == 42


def test_get_primitive_argument_without_mapping_raises_type_error(ioc):
    with pytest.raises(TypeError):
        ioc.get(NeedsStr)


def test_get_uses_plain_argument_mapping(ioc):
    class Named(object):
        def __init__(self, name):
            self.name = name

    setattr(Named, attributes.IOC_ARGUMENTS_NAME, {"name": Plain("hello")})
    result = ioc.get(Named)
    assert result.name == "hello"


def test_register_class_maps_base_to_implementation(ioc):
    ioc.register(Repo, SqlRepo)
    result = ioc.get(Repo)
    assert isinstance(result, SqlRepo)
    assert type(result) is SqlRepo


def test_register_class_later_registration_overrides_previous(ioc):
    ioc.register(Repo, SqlRepo)
    ioc.register(Repo, OtherRepo)
    result = ioc.get(Repo)
    assert type(result) is OtherRepo


def test_register_class_multi_instance_creates_distinct_mapped_instances(ioc):
    ioc.register(Repo, SqlRepo)
    first = ioc.get(Repo, LifeCycle.MULTI_INSTANCE)
    second = ioc.get(Repo, LifeCycle.MULTI_INSTANCE)
    assert first is not second
    assert type(first) is SqlRepo
    assert type(second) is SqlRepo


def test_register_instance_returns_given_object(ioc):
    obj = Repo()
    ioc.register(Repo, obj)
    assert ioc.get(Repo) is obj


def test_register_dispatches_list_to_register_many(ioc):
    ioc.register(Repo, [SqlRepo, OtherRepo])
    result = ioc.get(List[Repo])
    assert [type(item) for item in result] == [SqlRepo, OtherRepo]


def test_register_many_get_list_returns_instances_of_each_class(ioc):
    ioc.register_many(Repo, [SqlRepo, OtherRepo])
    result = ioc.get(List[Repo])
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], SqlRepo)
    assert isinstance(result[1], OtherRepo)


def test_register_many_list_is_singleton(ioc):
    ioc.register_many(Repo, [SqlRepo])
    first = ioc.get(List[Repo])
    second = ioc.get(List[Repo])
    assert first is second


def test_register_many_empty_list_returns_empty_list(ioc):
    ioc.register_many(Repo, [])
    result = ioc.get(List[Repo])
    assert result == []


def test_list_class_name_format_used_for_registration(ioc):
    assert ioc._get_class_name(List[Repo]) == "List[Repo]"


def test_clear_all_forces_new_singleton(ioc):
    first = ioc.get(Repo)
    ioc.clear_all()
    second = ioc.get(Repo)
    assert first is not second


def test_module_register_and_get(global_container):
    container_module.register(Repo, SqlRepo)
    result = container_module.get(Repo)
    assert type(result) is SqlRepo


def test_module_register_many_and_get_list(global_container):
    container_module.register_many(Repo, [SqlRepo, OtherRepo])
    result = container_module.get(List[Repo])
    assert [type(item) for item in result] == [SqlRepo, OtherRepo]


def test_module_initialize_with_class_mapping(global_container):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        container_module.initialize(class_mapping={Repo: SqlRepo})
    result = container_module.get(Repo)
    assert type(result) is SqlRepo


def test_module_initialize_again_warns(global_container):
    with pytest.warns(UserWarning):
        container_module.initialize()
