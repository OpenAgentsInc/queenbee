from typing import TypeVar, Type, Any

T = TypeVar('T')

registry: dict[type, Any] = {}


def set(instance: Any):  # noqa
    assert type(instance) not in registry, "only call set once"
    registry[type(instance)] = instance


def get(typ: Type[T]) -> T:
    return registry[typ]


def lazy(typ: Type[T], *args, **kws) -> T:
    if typ not in registry:
        registry[typ] = typ(*args, **kws)
    return registry[typ]
