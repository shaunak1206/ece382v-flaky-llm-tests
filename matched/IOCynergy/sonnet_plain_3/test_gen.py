import pytest

import cynergy.container as container
from cynergy.attributes import LifeCycle
from cynergy.container import IocContainer
from typing import List


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
    def __init__(self, name: str):
        self.name = name


@pytest.fixture
def ioc():
    return IocContainer(None)


def test_get_creates_instance_of_plain_class(ioc):
    result = ioc.get(Dependency)
    assert isinstance(result, Dependency)


def test_get_singleton_returns_same_instance(ioc):
    assert ioc.get(Dependency) is ioc.get(Dependency)


def test_get_multi_instance_returns_new_instances(ioc):
    first = ioc.get(Dependency, LifeCycle.MULTI_INSTANCE)
    second = ioc.get(Dependency, LifeCycle.MULTI_INSTANCE)
    assert isinstance(first, Dependency)
    assert isinstance(second, Dependency)
    assert first is not second


def test_get_injects_dependencies_by_annotation(ioc):
    result = ioc.get(NeedsDependency)
    assert isinstance(result.dep, Dependency)
    assert result.dep is ioc.get(Dependency)


def test_get_uses_default_argument_values(ioc):
    assert ioc.get(WithDefault).value == 5


def test_get_primitive_argument_without_mapping_raises_type_error(ioc):
    with pytest.raises(TypeError):
        ioc.get(NeedsPrimitive)


def test_get_unsupported_lifecycle_raises(ioc):
    with pytest.raises(NotImplementedError):
        ioc.get(Dependency, "not-a-lifecycle")


def test_register_class_maps_to_implementation(ioc):
    ioc.register(Base, ImplA)
    result = ioc.get(Base)
    assert type(result) is ImplA


def test_register_instance_returns_exact_instance(ioc):
    instance = ImplB()
    ioc.register(Base, instance)
    assert ioc.get(Base) is instance


def test_register_instance_is_used_as_dependency(ioc):
    dep = Dependency()
    ioc.register(Dependency, dep)
    assert ioc.get(NeedsDependency).dep is dep


def test_register_list_delegates_to_register_many(ioc):
    ioc.register(Base, [ImplA, ImplB])
    result = ioc.get(List[Base])
    assert [type(x) for x in result] == [ImplA, ImplB]


def test_register_many_returns_list_of_instances(ioc):
    ioc.register_many(Base, [ImplA, ImplB])
    result = ioc.get(List[Base])
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], ImplA)
    assert isinstance(result[1], ImplB)


def test_register_many_items_are_singletons_shared_with_get(ioc):
    ioc.register_many(Base, [ImplA, ImplB])
    result = ioc.get(List[Base])
    assert result[0] is ioc.get(ImplA)
    assert result[1] is ioc.get(ImplB)


def test_register_many_list_is_singleton_by_default(ioc):
    ioc.register_many(Base, [ImplA])
    assert ioc.get(List[Base]) is ioc.get(List[Base])


def test_register_many_replaces_previous_registration(ioc):
    ioc.register_many(Base, [ImplA, ImplB])
    ioc.register_many(Base, [ImplB])
    result = ioc.get(List[Base])
    assert len(result) == 1
    assert isinstance(result[0], ImplB)


def test_register_many_empty_list_gives_empty_list(ioc):
    ioc.register_many(Base, [])
    assert ioc.get(List[Base]) == []


def test_containers_are_isolated(ioc):
    other = IocContainer(None)
    ioc.register(Base, ImplA)
    other.register(Base, ImplB)
    assert type(ioc.get(Base)) is ImplA
    assert type(other.get(Base)) is ImplB


def test_module_level_register_and_get():
    container.initialize()
    container.register(Base, ImplA)
    assert type(container.get(Base)) is ImplA
    container.register_many(Base, [ImplA, ImplB])
    result = container.get(List[Base])
    assert [type(x) for x in result] == [ImplA, ImplB]
    container.initialize()
