"""
Provides general-purpose, generic registry implementations.

This module contains two main classes:
- `ClassRegistry`: A generic registry for mapping string keys to classes. It is
  type-aware and ensures that registered items are subclasses of a specified
  base class.
- `CallableRegistry`: A generic registry for mapping string keys to functions
  or other callables. It is also type-aware, allowing for static analysis of
  the retrieved callables.

Both registries support registration via decorators, making them easy to use
for building extensible, plug-in-based architectures.
"""
from abc import abstractmethod, ABC
import inspect
import logging
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Mapping, Tuple, overload
from bhtrace.globs import BHTRACE_LOG_FACTORY_PARAMS

# from bhtrace.utils.log import Log

log = logging.getLogger(__file__)


T = TypeVar("T")

class RegistryMixin(ABC):
    """
    Mixin class, implementing type-independent functional of Registry

    Note
    ----
    This implementation is not thread-safe for concurrent writes, but this
    is generally not an issue as registration typically happens once on
    the main thread at startup. A lock could be added if needed.
    """
    REGISTRY: Dict
    _alias_to_key: Dict[str, str]
    _key_to_alias: Dict[str, Tuple[str]]
    type: T

    def r(self, obj: Any, key: Any, *aliases: Any):
        if key in self.REGISTRY:
            raise ValueError(f"Unique key '{key}' is already registered.")

        all_keys = [key, *aliases]
        for k in all_keys:
            if k in self._alias_to_key:
                print(obj)
                raise ValueError(f"Key or alias '{k}' is already registered.")

        self.REGISTRY[key] = obj
        for k in all_keys:
            self._alias_to_key[k] = key
        self._key_to_alias[k] = (aliases)

        return obj

    @abstractmethod
    def _check(self, obj: Any) -> bool:
        """Controls how typecheck is done for this registry"""
        pass

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(containing:"
        )
    
    def unique(self):
        return self.REGISTRY.keys()

    def keys(self):
        return self._alias_to_key.keys()

    def items(self):
        """Returns registered items"""
        return [(k, self.REGISTRY[v]) for k, v in self._alias_to_key.items()]

    def values(self):
        return self.REGISTRY.values()

    def update(self, m: Mapping[str, Any]):
        map(self.r, m.values(), m.keys())
        # for k, v in m.items():

    def __contains__(self, item):
        return item in self._alias_to_key

    def isin(self, key: Any, raise_err: bool = True) -> bool:
        """
        Checks if a key is present in the registry.

        Parameters
        ----------
        key : str
            The key to check.
        raise_err : bool, optional
            If True, raises a KeyError if the key is not found.
            If False, returns a boolean. Defaults to True.

        Returns
        -------
        bool
            True if the key exists, False otherwise (if raise_err is False).
        """
        if key in self._alias_to_key:
            return True
        if raise_err:
            raise KeyError(
                f"Key '{key}' not found in {self.__class__.__name__}. Available keys: {list(self.keys())}"
            )
        return False
    
    def __getitem__(self, key: Any) -> Any:
        """
        Retrieves a registered item by its key.
        """
        try:
            unique_key = self._alias_to_key[key]
            return self.REGISTRY[unique_key]
        except KeyError:
            raise KeyError(
                f"Key '{key}' not found in {self.__class__.__name__}. Available keys: {list(self.keys())}"
            )

    def get(self, key: Any, default: Optional[Any] = None) -> Optional[Any]:
        """
        Retrieves a registered item by its key. Returns a default value if not found.
        """
        unique_key = self._alias_to_key.get(key)
        if unique_key is None:
            return default
        return self.REGISTRY.get(unique_key, default)
    
    def typesafe(self, o: object, default: Any):
        """
        Accepts any object, returns correct type
        """
        if isinstance(o, str):
            return self[o]
        elif self._check(o):
            return o
        elif default is not None:
            return self.typesafe(default, None)
        else:
            raise TypeError(
                f"Object `o` is not valid key or instance of {self.type} and no default is provided"
            )

    def info(self) -> str:
        """Returns information about this registry"""
        return (
            f"{self.__class__.__name__}({self.type}), containing:\n\t" +
            "\n\t".join(
                f"{k}: {self[k].__repr__()}, aliases: {self._key_to_alias.get(k)}"
                for k in self.unique()
            )
        )

class ClassRegistry(Generic[T], RegistryMixin):
    """
    A generic registry to map string keys to classes. It supports registration
    via a decorator and creating instances of registered classes. By using
    a TypeVar, it allows static type checkers to infer the correct type
    of created instances.
    """

    def __init__(self, type: Type[T]):
        """
        Initializes the registry.

        Parameters
        ----------
        type : Type[T]
            The base class that all registered items must inherit from.
        """
        self.type = type
        self.REGISTRY: Dict[str, Type[T]] = {}
        self._alias_to_key: Dict[str, str] = {}
        self._key_to_alias: Dict[str, str] = {}

    def _check(self, obj: Any) -> bool:
        return issubclass(obj, self.type)
    
    def r(self, cls: Type[T], key: str, *aliases) -> Type[T]:
        return super().r(cls, key, *aliases)
    
    def register(self, key: str, *alias: str, aliases: List[str] = None) -> Callable[[Type[T]], Type[T]]:
        """
        Returns a decorator to register a class with a given key and optional aliases.

        Parameters
        ----------
        key : str
            The primary key to associate with the class.
        aliases : Optional[List[str]]
            A list of alternative keys.

        Returns
        -------
        Callable[[Type[T]], Type[T]]
            A decorator that registers the class.

        Raises
        ------
        TypeError
            If the registered class is not a subclass of the registry's base class.
        ValueError
            If the key or any of the aliases are already registered.
        """
        def decorator(cls: Type[T]) -> Type[T]:
            return self.r(cls, key, *alias)

        return decorator

    def __getitem__(self, key: str) -> Type[T]:
        return super().__getitem__(key)
    
    def get(self, key: str, default: Optional[Type[T]] = None) -> Optional[Type[T]]:
        return super().get(key, default)
    
    def get_init_params(self, key: str) -> List[str]:
        """
        Retrieves the __init__ parameter names for a registered class.

        Parameters
        ----------
        key : str
            The key of the class to inspect.

        Returns
        -------
        List[str]
            A list of parameter names for the class's constructor.
        """
        TargetClass = self[key]
        init_signature = inspect.signature(TargetClass.__init__)
        return list(init_signature.parameters.keys())

    def create(self, key: str, *args, **kwargs) -> T:
        """
        Creates an instance of a registered class.

        Parameters
        ----------
        key : str
            The key of the class to instantiate.
        *args
            Positional arguments to pass to the class constructor.
        **kwargs
            Keyword arguments to pass to the class constructor.

        Returns
        -------
        T
            An instance of the registered class.
        """
        if BHTRACE_LOG_FACTORY_PARAMS:
            log.info(
                f"ClassRegistry {self.type} creates `{key}`:\n"
                f"args: {args} | kwargs: {kwargs}"
            )

        TargetClass = self[key]
        return TargetClass(*args, **kwargs)

Registry = ClassRegistry  # Alias for backward compatiblity

class InstanceRegistry(Generic[T], RegistryMixin):

    def __init__(self, type: Type[T]):
        """
        Initializes the registry.

        Parameters
        ----------
        type : Type[T]
            The base class that all registered items must inherit from.
        """
        self.type = type
        self.REGISTRY: Dict[str, T] = {}
        self._alias_to_key: Dict[str, str] = {}
        self._key_to_alias: Dict[str, str] = {}

    def _check(self, obj) -> bool:
        return isinstance(obj, self.type)

    def r(self, instance: T, key: str, *aliases: str) -> T:
        return super().r(instance, key, *aliases)

    def register(self, key: str, *alias, aliases: List[str] = None) -> Callable[[T], T]:
        """
        Returns a decorator to register an instance with a given key and optional aliases.

        Parameters
        ----------
        key : str
            The primary key to associate with the class.
        *aliases : 
            Key aliases

        Returns
        -------
        Callable[[T], T]
            A decorator that registers the instance.

        Raises
        ------
        TypeError
            If the object is not instance of the registry's base class.
        KeyError
            If the key or any of the aliases are already registered.
        """
        aliases = aliases or []
        def decorator(instance: T) -> T:
            return self.r(instance, key, *alias, *aliases)

        return decorator

    def __getitem__(self, key: str) -> T:
        return super().__getitem__(key)
    
    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        return super().get(key, default)
    
    def typesafe(self, o: Any, default: Optional[str | T] = None) -> T:
        """Accepts any object or key, returns instance or raises an error"""
        return super().typesafe(o, default)


# ----- Callable Registry -----
class CallableRegistry(Generic[T], RegistryMixin):
    """
    A generic registry for mapping keys to callables (e.g., functions).
    It supports registration via a decorator and uses a TypeVar to allow
    static type checkers to infer the correct callable type.

    Note
    ----
    It only differs from 
    """

    def __init__(self, type: Optional[Type[T]] = None):
        """
        Initializes the registry.

        Parameters
        ----------
        type : Optional[Type[T]], optional
            The protocol or base callable type for type hinting.
            This is not used for runtime validation.
        """
        self.type = type or Callable
        self.REGISTRY: Dict[Any, T] = {}
        self._alias_to_key: Dict[str, str] = {}
        self._key_to_alias: Dict[str, str] = {}

    # TODO: add/replace to signature check?
    def _check(self, obj) -> bool:
        return isinstance(obj, self.type)

    def r(self, instance: T, key: str, *aliases: str) -> T:
        return super().r(instance, key, *aliases)

    def register(self, key: Any, *alias, aliases: List[str] = None) -> Callable[[T], T]:
        """
        Returns a decorator to register a callable with a given key and optional aliases.

        Parameters
        ----------
        key : Any
            The primary key to associate with the callable.
        aliases : Optional[List[Any]]
            A list of alternative keys.

        Returns
        -------
        Callable[[T], T]
            A decorator that registers the callable.

        Raises
        ------
        KeyError
            If the key or any of the aliases are already registered.
        """
        # aliases = (aliases or []).extend(alias)
        def decorator(func: T) -> T:
            self.r(func, key, *alias)
            return func

        return decorator

    def signature(self, key: str) -> inspect.Signature:
        return inspect.signature(self[key])

    def __getitem__(self, key: Any) -> T:
        return super().__getitem__(key)
    
    def get(self, key: Any, default: Optional[T] = None) -> Optional[T]:
        return super().get(key, default)
