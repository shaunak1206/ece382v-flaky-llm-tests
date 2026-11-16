import typing
from typing import List

import pytest

from cynergy import container as cont
from cynergy.container import IocContainer
from cynergy.attributes import LifeCycle
from cynergy import attributes


class Simple(object):
    def __init__(self):
        self.value = 1


class Base(object):
    pass


class ImplOne(Base):
    pass


class ImplTwo(Base):
    pass


class NeedsSimple(object):
    def __init__(self, simple: Simple):
        self.simple = simple


class WithDefault(object):
    def __init__(self, number=5):
        self.number = number


class NeedsPrimitive(object):
    def __init__(self, number: int):
        self.number = number


class ClassLevelMulti(object):
    pass


@pytest.fixture
def container():
    return IocContainer(None)


@pytest.fixture
def static_container():
    cont.initialize()
    yield cont
    cont._clear_all()


def test_get_creates_instance_of_class(container):
    result = container.get(Simple)
    assert isinstance(result, Simple)
    assert result.value == 1


def test_get_singleton_returns_same_instance(container):
    assert container.get(Simple) is container.get(Simple)


def test_get_multi_instance_returns_new_instances(container):
    first = container.get(Simple, LifeCycle.MULTI_INSTANCE)
    second = container.get(Simple, LifeCycle.MULTI_INSTANCE)
    assert isinstance(first, Simple)
    assert isinstance(second, Simple)
    assert first is not second


def test_get_unsupported_life_cycle_raises(container):
    with pytest.raises(NotImplementedError):
        container.get(Simple, "not-a-life-cycle")


def test_get_with_none_life_cycle_defaults_to_singleton(container):
    assert container.get(Simple, None) is container.get(Simple, None)


def test_get_with_none_life_cycle_uses_class_attribute(container):
    setattr(ClassLevelMulti, attributes.LIFECYCLE_ARGUMENT_NAME, LifeCycle.MULTI_INSTANCE)
    try:
        first = container.get(ClassLevelMulti, None)
        second = container.get(ClassLevelMulti, None)
    finally:
        delattr(ClassLevelMulti, attributes.LIFECYCLE_ARGUMENT_NAME)
    assert first is not second


def test_get_injects_dependencies_by_annotation(container):
    result = container.get(NeedsSimple)
    assert isinstance(result.simple, Simple)
    assert result.simple is container.get(Simple)


def test_get_uses_default_argument_values(container):
    assert container.get(WithDefault).number == 5


def test_get_primitive_without_mapping_raises_type_error(container):
    with pytest.raises(TypeError):
        container.get(NeedsPrimitive)


def test_register_instance_is_returned_by_get(container):
    instance = Simple()
    instance.value = 42
    container.register(Simple, instance)
    assert container.get(Simple) is instance


def test_register_class_maps_to_implementation(container):
    container.register(Base, ImplOne)
    result = container.get(Base)
    assert type(result) is ImplOne


def test_register_class_overrides_previous_mapping(container):
    container.register(Base, ImplOne)
    container.register(Base, ImplTwo)
    assert type(container.get(Base)) is ImplTwo


def test_register_class_multi_instance_creates_new_mapped_objects(container):
    container.register(Base, ImplOne)
    first = container.get(Base, LifeCycle.MULTI_INSTANCE)
    second = container.get(Base, LifeCycle.MULTI_INSTANCE)
    assert type(first) is ImplOne
    assert type(second) is ImplOne
    assert first is not second


def test_register_list_delegates_to_register_many(container):
    container.register(Base, [ImplOne, ImplTwo])
    result = container.get(List[Base])
    assert isinstance(result, list)
    assert [type(item) for item in result] == [ImplOne, ImplTwo]


def test_register_many_returns_list_of_instances(container):
    container.register_many(Base, [ImplTwo, ImplOne])
    result = container.get(List[Base])
    assert [type(item) for item in result] == [ImplTwo, ImplOne]


def test_register_many_list_is_singleton_by_default(container):
    container.register_many(Base, [ImplOne, ImplTwo])
    assert container.get(List[Base]) is container.get(List[Base])


def test_register_many_multi_instance_new_list_same_singleton_items(container):
    container.register_many(Base, [ImplOne])
    first = container.get(List[Base], LifeCycle.MULTI_INSTANCE)
    second = container.get(List[Base], LifeCycle.MULTI_INSTANCE)
    assert first is not second
    assert first[0] is second[0]


def test_register_many_with_empty_list(container):
    container.register_many(Base, [])
    assert container.get(List[Base]) == []


def test_register_many_does_not_affect_single_registration(container):
    container.register_many(Base, [ImplOne])
    container.register(Base, ImplTwo)
    assert type(container.get(Base)) is ImplTwo
    assert [type(i) for i in container.get(List[Base])] == [ImplOne]


def test_static_register_and_get(static_container):
    instance = Simple()
    static_container.register(Simple, instance)
    assert static_container.get(Simple) is instance


def test_static_register_many_and_get(static_container):
    static_container.register_many(Base, [ImplOne, ImplTwo])
    result = static_container.get(typing.List[Base])
    assert [type(item) for item in result] == [ImplOne, ImplTwo]
