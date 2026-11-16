import pytest

from typing import List

from cynergy import container as container_module
from cynergy.container import IocContainer
from cynergy.attributes import LifeCycle


class Base(object):
    pass


class ImplA(Base):
    pass


class ImplB(Base):
    pass


class Dependency(object):
    pass


class NeedsDependency(object):
    def __init__(self, dep: Dependency):
        self.dep = dep


class WithDefault(object):
    def __init__(self, value=5):
        self.value = value


class NeedsPrimitive(object):
    def __init__(self, value: int):
        self.value = value


@pytest.fixture
def ioc():
    return IocContainer(None)


@pytest.fixture
def static_container():
    container_module.initialize(None)
    yield container_module
    container_module.initialize(None)


def test_get_creates_instance_of_plain_class(ioc):
    result = ioc.get(Base)
    assert isinstance(result, Base)


def test_get_singleton_returns_same_instance(ioc):
    assert ioc.get(Base) is ioc.get(Base)


def test_get_multi_instance_returns_new_instances(ioc):
    first = ioc.get(Base, LifeCycle.MULTI_INSTANCE)
    second = ioc.get(Base, LifeCycle.MULTI_INSTANCE)
    assert isinstance(first, Base)
    assert isinstance(second, Base)
    assert first is not second


def test_get_resolves_dependency_by_annotation(ioc):
    result = ioc.get(NeedsDependency)
    assert isinstance(result.dep, Dependency)
    assert result.dep is ioc.get(Dependency)


def test_get_uses_default_argument_value(ioc):
    assert ioc.get(WithDefault).value == 5


def test_get_primitive_argument_without_mapping_raises_type_error(ioc):
    with pytest.raises(TypeError):
        ioc.get(NeedsPrimitive)


def test_get_unknown_lifecycle_raises_not_implemented(ioc):
    with pytest.raises(NotImplementedError):
        ioc.get(Base, "not-a-lifecycle")


def test_register_class_makes_get_return_mapped_class(ioc):
    ioc.register(Base, ImplA)
    result = ioc.get(Base)
    assert isinstance(result, ImplA)


def test_register_class_singleton_is_shared(ioc):
    ioc.register(Base, ImplA)
    assert ioc.get(Base) is ioc.get(Base)


def test_register_class_multi_instance_creates_new_each_time(ioc):
    ioc.register(Base, ImplA)
    first = ioc.get(Base, LifeCycle.MULTI_INSTANCE)
    second = ioc.get(Base, LifeCycle.MULTI_INSTANCE)
    assert isinstance(first, ImplA)
    assert first is not second


def test_register_instance_returns_exact_instance(ioc):
    instance = ImplA()
    ioc.register(Base, instance)
    assert ioc.get(Base) is instance


def test_register_non_class_value_as_instance(ioc):
    ioc.register(Dependency, "some-value")
    assert ioc.get(Dependency) == "some-value"


def test_register_list_delegates_to_register_many(ioc):
    ioc.register(List[Base], [ImplA, ImplB])
    result = ioc.get(List[Base])
    assert isinstance(result, list)
    assert [type(item) for item in result] == [ImplA, ImplB]


def test_register_many_returns_instances_in_order(ioc):
    ioc.register_many(Base, [ImplB, ImplA])
    result = ioc.get(List[Base])
    assert len(result) == 2
    assert isinstance(result[0], ImplB)
    assert isinstance(result[1], ImplA)


def test_register_many_singleton_list_is_cached(ioc):
    ioc.register_many(Base, [ImplA, ImplB])
    assert ioc.get(List[Base]) is ioc.get(List[Base])


def test_register_many_elements_are_shared_singletons(ioc):
    ioc.register_many(Base, [ImplA, ImplB])
    result = ioc.get(List[Base], LifeCycle.MULTI_INSTANCE)
    assert result[0] is ioc.get(ImplA)
    assert result[1] is ioc.get(ImplB)


def test_register_many_with_empty_list(ioc):
    ioc.register_many(Base, [])
    assert ioc.get(List[Base]) == []


def test_get_by_name_returns_registered_instance(ioc):
    instance = object()
    ioc.register_instance_by_name("my-service", instance)
    assert ioc.get_by_name("my-service") is instance


def test_get_by_name_unregistered_raises_key_error(ioc):
    with pytest.raises(KeyError):
        ioc.get_by_name("missing")


def test_static_register_class_and_get(static_container):
    static_container.register(Base, ImplB)
    assert isinstance(static_container.get(Base), ImplB)


def test_static_register_instance_and_get(static_container):
    instance = ImplA()
    static_container.register(Base, instance)
    assert static_container.get(Base) is instance


def test_static_register_many_and_get(static_container):
    static_container.register_many(Base, [ImplA, ImplB])
    result = static_container.get(List[Base])
    assert [type(item) for item in result] == [ImplA, ImplB]
