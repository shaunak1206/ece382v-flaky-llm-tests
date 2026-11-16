import warnings
from typing import List

import pytest

from cynergy import container
from cynergy import attributes
from cynergy.attributes import LifeCycle
from cynergy.container import IocContainer


class Base(object):
    pass


class ImplA(Base):
    pass


class ImplB(Base):
    pass


class Simple(object):
    pass


class Dependent(object):
    def __init__(self, simple: Simple):
        self.simple = simple


class WithDefault(object):
    def __init__(self, value=5):
        self.value = value


class NeedsPrimitive(object):
    def __init__(self, number: int):
        self.number = number


class Mapped(object):
    def __init__(self, name: str):
        self.name = name


setattr(Mapped, attributes.IOC_ARGUMENTS_NAME, {'name': 'mapped-name'})


class MultiByAttribute(object):
    pass


setattr(MultiByAttribute, attributes.LIFECYCLE_ARGUMENT_NAME, LifeCycle.MULTI_INSTANCE)


@pytest.fixture
def ioc():
    return IocContainer(None)


@pytest.fixture
def static_container():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        container.initialize()
    yield container
    container._clear_all()


def test_get_creates_instance_of_simple_class(ioc):
    assert isinstance(ioc.get(Simple), Simple)


def test_get_singleton_returns_same_instance(ioc):
    assert ioc.get(Simple) is ioc.get(Simple)


def test_get_multi_instance_returns_new_instances(ioc):
    first = ioc.get(Simple, LifeCycle.MULTI_INSTANCE)
    second = ioc.get(Simple, LifeCycle.MULTI_INSTANCE)
    assert isinstance(first, Simple)
    assert isinstance(second, Simple)
    assert first is not second


def test_get_with_none_lifecycle_uses_class_attribute(ioc):
    first = ioc.get(MultiByAttribute, None)
    second = ioc.get(MultiByAttribute, None)
    assert first is not second


def test_get_with_none_lifecycle_defaults_to_singleton(ioc):
    first = ioc.get(Simple, None)
    second = ioc.get(Simple, None)
    assert first is second


def test_get_unknown_lifecycle_raises(ioc):
    with pytest.raises(NotImplementedError):
        ioc.get(Simple, "not-a-lifecycle")


def test_get_injects_dependencies_by_annotation(ioc):
    dependent = ioc.get(Dependent)
    assert isinstance(dependent.simple, Simple)
    assert dependent.simple is ioc.get(Simple)


def test_get_uses_default_argument_value(ioc):
    assert ioc.get(WithDefault).value == 5


def test_get_primitive_without_mapping_raises_type_error(ioc):
    with pytest.raises(TypeError):
        ioc.get(NeedsPrimitive)


def test_get_uses_argument_mapping_for_primitives(ioc):
    assert ioc.get(Mapped).name == 'mapped-name'


def test_register_class_maps_to_implementation(ioc):
    ioc.register(Base, ImplA)
    result = ioc.get(Base)
    assert isinstance(result, ImplA)
    assert ioc.get(Base) is result


def test_register_instance_returns_that_instance(ioc):
    instance = ImplB()
    ioc.register(Base, instance)
    assert ioc.get(Base) is instance


def test_register_list_registers_many(ioc):
    ioc.register(Base, [ImplA, ImplB])
    result = ioc.get(List[Base])
    assert len(result) == 2
    assert isinstance(result[0], ImplA)
    assert isinstance(result[1], ImplB)


def test_register_many_returns_list_of_instances(ioc):
    ioc.register_many(Base, [ImplB, ImplA])
    result = ioc.get(List[Base])
    assert [type(item) for item in result] == [ImplB, ImplA]


def test_register_many_singleton_list_is_cached(ioc):
    ioc.register_many(Base, [ImplA, ImplB])
    assert ioc.get(List[Base]) is ioc.get(List[Base])


def test_register_many_multi_instance_creates_new_list_with_shared_items(ioc):
    ioc.register_many(Base, [ImplA, ImplB])
    first = ioc.get(List[Base], LifeCycle.MULTI_INSTANCE)
    second = ioc.get(List[Base], LifeCycle.MULTI_INSTANCE)
    assert first is not second
    assert first[0] is second[0]
    assert first[1] is second[1]


def test_register_many_empty_list(ioc):
    ioc.register_many(Base, [])
    assert ioc.get(List[Base]) == []


def test_static_register_and_get(static_container):
    static_container.register(Base, ImplA)
    assert isinstance(static_container.get(Base), ImplA)


def test_static_register_many_and_get(static_container):
    static_container.register_many(Base, [ImplA, ImplB])
    result = static_container.get(List[Base])
    assert [type(item) for item in result] == [ImplA, ImplB]


def test_static_register_instance_via_register(static_container):
    instance = ImplA()
    static_container.register(Base, instance)
    assert static_container.get(Base) is instance


def test_initialize_with_class_mapping_is_used_by_get():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        container.initialize(class_mapping={Base: ImplB})
    try:
        assert isinstance(container.get(Base), ImplB)
    finally:
        container._clear_all()
