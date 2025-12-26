"""
Dependency injection container.

Manages service lifetime and dependencies.
"""
from dataclasses import dataclass, field
from typing import TypeVar, Type, Callable, Optional, Any

from .config.settings import Settings, get_settings


T = TypeVar("T")


@dataclass
class Container:
    """
    Simple dependency injection container.

    Supports:
    - Singleton and transient lifetimes
    - Factory-based registration
    - Lazy resolution
    """

    settings: Settings = field(default_factory=get_settings)
    _registry: dict[type, tuple[Callable, bool]] = field(default_factory=dict)
    _singletons: dict[type, Any] = field(default_factory=dict)

    def register(
        self,
        interface: Type[T],
        factory: Callable[[], T],
        singleton: bool = True
    ) -> None:
        """
        Register a service.

        Args:
            interface: The interface/type to register
            factory: Factory function to create instances
            singleton: If True, only one instance is created
        """
        self._registry[interface] = (factory, singleton)

    def resolve(self, interface: Type[T]) -> T:
        """
        Resolve a service.

        Args:
            interface: The interface/type to resolve

        Returns:
            Service instance

        Raises:
            KeyError: If interface is not registered
        """
        # Check for existing singleton
        if interface in self._singletons:
            return self._singletons[interface]

        # Get registration
        if interface not in self._registry:
            raise KeyError(f"No registration for {interface}")

        factory, singleton = self._registry[interface]
        instance = factory()

        # Store singleton
        if singleton:
            self._singletons[interface] = instance

        return instance

    def is_registered(self, interface: Type[T]) -> bool:
        """Check if interface is registered."""
        return interface in self._registry

    def register_defaults(self) -> None:
        """Register default implementations."""
        from ..encoding.encoder import PadicHyperbolicEncoder, EncoderConfig
        from ..core.interfaces.encoder import IEncoder

        # Register encoder
        self.register(
            IEncoder,
            lambda: PadicHyperbolicEncoder(
                config=EncoderConfig(
                    prime=self.settings.encoder.prime,
                    embedding_dim=self.settings.encoder.embedding_dim,
                    curvature=self.settings.encoder.curvature,
                    padic_precision=self.settings.encoder.padic_precision,
                )
            ),
        )

    def clear(self) -> None:
        """Clear all registrations and singletons."""
        self._registry.clear()
        self._singletons.clear()


# Global container instance
_container: Optional[Container] = None


def get_container() -> Container:
    """Get the global container instance."""
    global _container
    if _container is None:
        _container = Container()
        _container.register_defaults()
    return _container


def inject(interface: Type[T]) -> T:
    """
    Convenience function to inject a dependency.

    Args:
        interface: The interface to resolve

    Returns:
        Service instance
    """
    return get_container().resolve(interface)


def reset_container() -> None:
    """Reset container (for testing)."""
    global _container
    if _container is not None:
        _container.clear()
    _container = None
