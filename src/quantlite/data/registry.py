"""Plugin registry for custom data sources."""

from __future__ import annotations

from typing import Any, TypeVar

from .base import DataSource

_REGISTRY: dict[str, type[DataSource]] = {}

T = TypeVar("T", bound=DataSource)


def register_source(name: str) -> Any:
    """Class decorator that registers a :class:`DataSource` subclass.

    Usage::

        @register_source("my_api")
        class MyAPISource(DataSource):
            ...

    Args:
        name: Short identifier for the source (e.g. ``"my_api"``).

    Returns:
        A class decorator.

    Raises:
        TypeError: If the decorated class is not a DataSource subclass.
    """

    def decorator(cls: type[T]) -> type[T]:
        if not (isinstance(cls, type) and issubclass(cls, DataSource)):
            msg = f"{cls!r} is not a DataSource subclass"
            raise TypeError(msg)
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_source(name: str) -> DataSource:
    """Look up a registered source by *name* and return an instance.

    Args:
        name: The registered source name.

    Returns:
        An instance of the registered :class:`DataSource`.

    Raises:
        KeyError: If no source is registered under *name*.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        msg = f"Unknown data source {name!r}. Registered sources: {available}"
        raise KeyError(msg)
    return _REGISTRY[name]()


def list_sources() -> list[str]:
    """Return the names of all registered data sources.

    Returns:
        Sorted list of registered source names.
    """
    return sorted(_REGISTRY)
