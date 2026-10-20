"""Registry pattern for modular and decoupled component instantiation."""

from typing import Any, Callable, Dict, Optional


class Registry:
    """A registry mapping component names to their class or builder function.

    Example:
        MODELS = Registry("models")

        @MODELS.register()
        class MyNewModel(nn.Module):
            ...

        model = MODELS.build("my_new_model", **kwargs)
    """

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Any] = {}

    def register(self, name: Optional[str] = None) -> Callable:
        """Register a class or callable under an optional explicit name."""
        def decorator(cls_or_func: Any) -> Any:
            reg_name = name or cls_or_func.__name__
            reg_name_lower = reg_name.lower()
            if reg_name_lower in self._registry:
                # Allow re-registration with warning/override
                pass
            self._registry[reg_name_lower] = cls_or_func
            return cls_or_func

        return decorator

    def get(self, name: str) -> Any:
        """Retrieve registered object by name (case-insensitive)."""
        name_lower = name.lower()
        if name_lower not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"'{name}' is not registered in '{self.name}'. Available options: {available}"
            )
        return self._registry[name_lower]

    def build(self, name: str, *args, **kwargs) -> Any:
        """Construct an instance of a registered component."""
        cls_or_func = self.get(name)
        return cls_or_func(*args, **kwargs)

    def list_available(self) -> list:
        """Return list of all registered names."""
        return list(self._registry.keys())

    def __repr__(self) -> str:
        return f"Registry(name='{self.name}', items={list(self._registry.keys())})"


# Global Registries for every modular subsystem
BACKBONES = Registry("backbones")
HEADS = Registry("heads")
MODELS = Registry("models")
LOSSES = Registry("losses")
