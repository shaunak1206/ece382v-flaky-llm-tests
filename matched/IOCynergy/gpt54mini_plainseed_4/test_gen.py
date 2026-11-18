import pytest

import cynergy.container as container
from cynergy.attributes import LifeCycle


@pytest.fixture(autouse=True)
def reset_container():
    container.__dict__["__instance"] = None
    yield
    container.__dict__["__instance"] = None


def test_get_returns_singleton_instance_and_caches_it():
    calls = []

    class Service:
        def __init__(self):
            calls.append("init")

    first = container.get(Service)
    second = container.get(Service)

    assert first is second
    assert calls == ["init"]


def test_get_with_multi_instance_creates_new_instance_each_time():
    calls = []

    class Service:
        def __init__(self):
            calls.append("init")

    first = container.get(Service, LifeCycle.MULTI_INSTANCE)
    second = container.get(Service, LifeCycle.MULTI_INSTANCE)

    assert first is not second
    assert calls == ["init", "init"]


def test_register_instance_returns_registered_object_from_get():
    class Service:
        pass

    instance = Service()
    container.register_instance(Service, instance)

    assert container.get(Service) is instance


def test_register_class_maps_source_to_target_class():
    class Interface:
        pass

    class Implementation:
        def __init__(self):
            self.value = 42

    container.register_class(Interface, Implementation)

    result = container.get(Interface)

    assert isinstance(result, Implementation)
    assert result.value == 42


def test_register_with_class_uses_register_class():
    class Interface:
        pass

    class Implementation:
        def __init__(self):
            self.name = "impl"

    container.register(Interface, Implementation)

    result = container.get(Interface)

    assert isinstance(result, Implementation)
    assert result.name == "impl"


def test_register_with_instance_uses_register_instance():
    class Service:
        pass

    instance = Service()
    container.register(Service, instance)

    assert container.get(Service) is instance


def test_register_with_list_uses_register_many_and_get_returns_list_of_instances():
    class Base:
        pass

    class A(Base):
        def __init__(self):
            self.name = "A"

    class B(Base):
        def __init__(self):
            self.name = "B"

    container.register(Base, [A, B])

    result = container.get(Base)

    assert isinstance(result, list)
    assert [type(x) for x in result] == [A, B]
    assert [x.name for x in result] == ["A", "B"]


def test_register_many_explicitly_resolves_multiple_implementations():
    class Base:
        pass

    class A(Base):
        pass

    class B(Base):
        pass

    container.register_many(Base, [A, B])

    result = container.get(Base)

    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], A)
    assert isinstance(result[1], B)


def test_register_many_caches_list_result_for_singleton_lifecycle():
    class Base:
        pass

    class A(Base):
        def __init__(self):
            self.token = object()

    class B(Base):
        def __init__(self):
            self.token = object()

    container.register_many(Base, [A, B])

    first = container.get(Base)
    second = container.get(Base)

    assert first is second
    assert [type(x) for x in first] == [A, B]


def test_get_by_name_returns_registered_instance():
    class Service:
        pass

    instance = Service()
    container.register_instance_by_name("alpha", instance)

    assert container.get_by_name("alpha") is instance


def test_clear_all_removes_cached_instances_but_keeps_registration():
    class Service:
        def __init__(self):
            self.created = object()

    first = container.get(Service)
    container._clear_all()
    second = container.get(Service)

    assert first is not second


def test_initialize_with_class_mapping_registers_mapping():
    class Interface:
        pass

    class Implementation:
        def __init__(self):
            self.flag = True

    container.initialize(class_mapping={Interface: Implementation})

    result = container.get(Interface)

    assert isinstance(result, Implementation)
    assert result.flag is True


def test_initialize_twice_emits_warning_and_reinitializes():
    class Service:
        def __init__(self):
            self.value = 1

    container.initialize()
    container.get(Service)

    with pytest.warns(UserWarning):
        container.initialize()

    result = container.get(Service)

    assert isinstance(result, Service)
    assert result.value == 1


def test_get_with_falsy_lifecycle_uses_class_level_lifecycle_attribute():
    class Service:
        __ioc_lifecycle__ = LifeCycle.MULTI_INSTANCE

        def __init__(self):
            pass

    first = container.get(Service, life_cycle=None)
    second = container.get(Service, life_cycle=None)

    assert first is not second


def test_get_unregistered_primitive_dependency_raises_type_error():
    class Service:
        def __init__(self, count: int):
            self.count = count

    with pytest.raises(TypeError):
        container.get(Service)


def test_register_then_get_nested_dependency_uses_registered_implementation():
    class Repository:
        def __init__(self):
            self.name = "repo"

    class Service:
        def __init__(self, repo: Repository):
            self.repo = repo

    container.register_class(Repository, Repository)

    result = container.get(Service)

    assert isinstance(result, Service)
    assert isinstance(result.repo, Repository)
    assert result.repo.name == "repo"
