import typing
from typing import List

import pytest

from cynergy import attributes
from cynergy.attributes import LifeCycle
from cynergy.container import IocContainer


class Service(object):
    def __init__(self):
        pass


class ServiceImpl(Service):
    def __init__(self):
        pass


class OtherServiceImpl(Service):
    def __init__(self):
        pass


class Dependency(object):
    def __init__(self):
        pass


class Consumer(object):
    def __init__(self, dep: Dependency):
        self.dep = dep


class Named(object):
    def __init__(self, name='default'):
        self.name = name


class NeedsInt(object):
    def __init__(self, n: int):
        self.n = n


@pytest.fixture
def container():
    return IocContainer(None)


# --- get: lifecycles and construction -------------------------------------

def test_get_singleton_returns_same_instance_on_repeated_calls(container):
    first = container.get(Dependency)
    second = container.get(Dependency)
    assert first is second


def test_get_multi_instance_returns_new_instance_each_call(container):
    first = container.get(Dependency, LifeCycle.MULTI_INSTANCE)
    second = container.get(Dependency, LifeCycle.MULTI_INSTANCE)
    assert isinstance(first, Dependency)
    assert isinstance(second, Dependency)
    assert first is not second


def test_get_creates_instance_of_unregistered_concrete_class(container):
    result = container.get(Dependency)
    assert isinstance(result, Dependency)


def test_get_resolves_annotated_dependency_automatically(container):
    consumer = container.get(Consumer)
    assert isinstance(consumer.dep, Dependency)


def test_get_uses_default_value_when_argument_not_mapped(container):
    result = container.get(Named)
    assert result.name == 'default'


def test_get_uses_argument_mapping_for_primitive_value(container):
    class Mapped(object):
        def __init__(self, name):
            self.name = name

    setattr(Mapped, attributes.IOC_ARGUMENTS_NAME, {'name': 'custom'})

    result = container.get(Mapped)
    assert result.name == 'custom'


def test_get_raises_type_error_for_primitive_without_mapping_or_default(container):
    with pytest.raises(TypeError):
        container.get(NeedsInt)


def test_get_raises_not_implemented_for_unsupported_mapped_argument_type(container):
    class Unsupported(object):
        def __init__(self, value):
            self.value = value

    setattr(Unsupported, attributes.IOC_ARGUMENTS_NAME, {'value': object()})

    with pytest.raises(NotImplementedError):
        container.get(Unsupported)


def test_get_unknown_lifecycle_raises_not_implemented(container):
    with pytest.raises(NotImplementedError):
        container.get(Dependency, "unknown-lifecycle")


def test_get_none_lifecycle_defaults_to_singleton(container):
    first = container.get(Dependency, None)
    second = container.get(Dependency)
    assert first is second


def test_get_none_lifecycle_honours_class_level_lifecycle(container):
    class PerRequest(object):
        def __init__(self):
            pass

    setattr(PerRequest, attributes.LIFECYCLE_ARGUMENT_NAME, LifeCycle.MULTI_INSTANCE)

    first = container.get(PerRequest, None)
    second = container.get(PerRequest, None)
    assert first is not second


# --- register: class, instance and list dispatch ---------------------------

def test_register_class_maps_abstract_type_to_implementation(container):
    container.register(Service, ServiceImpl)

    result = container.get(Service)
    assert isinstance(result, ServiceImpl)


def test_register_does_not_change_resolution_of_unrelated_types(container):
    container.register(Service, ServiceImpl)

    result = container.get(Dependency)
    assert isinstance(result, Dependency)


def test_register_instance_returns_given_object_from_get(container):
    obj = Service()

    container.register(Service, obj)

    assert container.get(Service) is obj


def test_register_instance_takes_precedence_over_class_mapping(container):
    container.register(Service, ServiceImpl)
    obj = Service()

    container.register(Service, obj)

    assert container.get(Service) is obj


def test_register_list_registers_many_implementations(container):
    container.register(Service, [ServiceImpl, OtherServiceImpl])

    result = container.get(List[Service])

    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], ServiceImpl)
    assert isinstance(result[1], OtherServiceImpl)


# --- register_many -----------------------------------------------------------

def test_register_many_get_returns_list_in_registration_order(container):
    container.register_many(Service, [OtherServiceImpl, ServiceImpl])

    result = container.get(List[Service])

    assert isinstance(result[0], OtherServiceImpl)
    assert isinstance(result[1], ServiceImpl)


def test_register_many_resolves_items_as_shared_singletons(container):
    container.register_many(Service, [ServiceImpl, OtherServiceImpl])

    result = container.get(List[Service])

    assert result[0] is container.get(ServiceImpl)
    assert result[1] is container.get(OtherServiceImpl)


def test_register_many_with_empty_list_returns_empty_list(container):
    container.register_many(Service, [])

    result = container.get(List[Service])

    assert result == []


def test_register_many_second_call_replaces_previous_registration(container):
    container.register_many(Service, [ServiceImpl])
    container.register_many(Service, [OtherServiceImpl])

    result = container.get(List[Service])

    assert len(result) == 1
    assert isinstance(result[0], OtherServiceImpl)


def test_class_name_for_list_type_uses_element_type_name(container):
    assert IocContainer._get_class_name(List[Service]) == 'List[Service]'
