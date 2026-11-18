import pytest

from cynergy import container as container_mod
from cynergy.attributes import LifeCycle
from cynergy.errors.ContainerException import ConfigProviderRequiredException
from cynergy.config import Config, Plain, ServiceByName


@pytest.fixture(autouse=True)
def reset_container():
    # Reset the module-level singleton before and after each test.
    container_mod.__dict__["__instance"] = None
    yield
    container_mod.__dict__["__instance"] = None


class DummyConfigProvider:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.calls = []

    def get(self, key):
        self.calls.append(key)
        return self.values[key]


class Dep:
    def __init__(self):
        self.marker = object()


class PlainDep:
    def __init__(self, value):
        self.value = value


class UsesPrimitive:
    def __init__(self, name: str):
        self.name = name


class UsesPlainAndConfig:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c


class UsesDep:
    def __init__(self, dep: Dep):
        self.dep = dep


class NeedsConfig:
    def __init__(self, token):
        self.token = token


class MultiService:
    pass


class RegisteredService:
    def __init__(self):
        self.value = 123


class ReplacementService:
    def __init__(self):
        self.value = 456


class ByNameConsumer:
    def __init__(self, service):
        self.service = service


def test_get_returns_same_singleton_instance_for_class():
    container_mod.initialize()
    first = container_mod.get(Dep)
    second = container_mod.get(Dep)

    assert first is second
    assert isinstance(first, Dep)


def test_get_multi_instance_creates_new_instance_each_time():
    container_mod.initialize()

    first = container_mod.get(Dep, life_cycle=LifeCycle.MULTI_INSTANCE)
    second = container_mod.get(Dep, life_cycle=LifeCycle.MULTI_INSTANCE)

    assert first is not second
    assert isinstance(first, Dep)
    assert isinstance(second, Dep)


def test_register_instance_and_get_returns_registered_instance():
    container_mod.initialize()
    instance = RegisteredService()

    container_mod.register(RegisteredService, instance)
    result = container_mod.get(RegisteredService)

    assert result is instance
    assert result.value == 123


def test_register_class_maps_type_to_alternate_class():
    container_mod.initialize()
    container_mod.register(RegisteredService, ReplacementService)

    result = container_mod.get(RegisteredService)

    assert isinstance(result, ReplacementService)
    assert result.value == 456
    assert container_mod.get(RegisteredService) is result


def test_register_many_maps_list_of_implementations():
    container_mod.initialize()
    container_mod.register_many(Dep, [Dep, ReplacementService])

    result = container_mod.get(Dep)

    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], Dep)
    assert isinstance(result[1], ReplacementService)


def test_register_dispatches_to_register_many_for_list():
    container_mod.initialize()
    container_mod.register(Dep, [Dep, ReplacementService])

    result = container_mod.get(Dep)

    assert isinstance(result, list)
    assert [type(x) for x in result] == [Dep, ReplacementService]


def test_register_dispatches_to_register_class_for_class():
    container_mod.initialize()
    container_mod.register(RegisteredService, ReplacementService)

    result = container_mod.get(RegisteredService)

    assert isinstance(result, ReplacementService)


def test_register_dispatches_to_register_instance_for_object():
    container_mod.initialize()
    instance = RegisteredService()

    container_mod.register(RegisteredService, instance)

    assert container_mod.get(RegisteredService) is instance


def test_get_by_name_returns_registered_instance():
    container_mod.initialize()
    instance = RegisteredService()
    container_mod.register_instance_by_name("svc", instance)

    result = container_mod.get_by_name("svc")

    assert result is instance


def test_get_by_name_raises_keyerror_when_missing():
    container_mod.initialize()

    with pytest.raises(KeyError):
        container_mod.get_by_name("missing")


def test_plain_value_is_passed_through_when_resolving_arguments():
    container_mod.initialize()
    container_mod.register_instance_by_name("svc", RegisteredService())

    class Consumer:
        def __init__(self, name, service):
            self.name = name
            self.service = service

    Consumer.__ioc_arguments__ = {
        "name": Plain("Alice"),
        "service": ServiceByName("svc"),
    }

    result = container_mod.get(Consumer)

    assert result.name == "Alice"
    assert isinstance(result.service, RegisteredService)


def test_config_value_is_resolved_from_config_provider():
    cfg = DummyConfigProvider({"api_token": "secret"})
    container_mod.initialize(cfg)

    class Consumer:
        def __init__(self, token):
            self.token = token

    Consumer.__ioc_arguments__ = {
        "token": Config("api_token"),
    }

    result = container_mod.get(Consumer)

    assert result.token == "secret"
    assert cfg.calls == ["api_token"]


def test_config_value_uses_default_when_missing():
    cfg = DummyConfigProvider({})
    container_mod.initialize(cfg)

    class Consumer:
        def __init__(self, token):
            self.token = token

    Consumer.__ioc_arguments__ = {
        "token": Config("missing_key", default="fallback"),
    }

    result = container_mod.get(Consumer)

    assert result.token == "fallback"
    assert cfg.calls == ["missing_key"]


def test_config_argument_without_provider_raises_config_provider_required_exception():
    container_mod.initialize(None)

    class Consumer:
        def __init__(self, token):
            self.token = token

    Consumer.__ioc_arguments__ = {
        "token": Config("api_token"),
    }

    with pytest.raises(ConfigProviderRequiredException):
        container_mod.get(Consumer)


def test_primitive_constructor_argument_without_mapping_raises_type_error():
    container_mod.initialize()

    with pytest.raises(TypeError, match="Could not initialize primitive argument"):
        container_mod.get(UsesPrimitive)


def test_get_uses_registered_dependency_by_type_annotation():
    container_mod.initialize()

    result = container_mod.get(UsesDep)

    assert isinstance(result.dep, Dep)
    assert isinstance(container_mod.get(Dep), Dep)


def test_initialize_with_class_mapping_uses_replacement_class():
    container_mod.initialize(class_mapping={RegisteredService: ReplacementService})

    result = container_mod.get(RegisteredService)

    assert isinstance(result, ReplacementService)
    assert result.value == 456
