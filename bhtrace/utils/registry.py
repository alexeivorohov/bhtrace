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
import inspect
import os
import logging
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Mapping
from bhtrace.globs import BHTRACE_LOG_FACTORY_PARAMS

# from bhtrace.utils.log import Log

log = logging.getLogger(__file__)


T = TypeVar("T")

class RegistryMixin:
    """
    Mixin class, implementing type-independent functional of Registry
    """
    REGISTRY: Dict

    def keys(self):
        return self.REGISTRY.keys()
    
    def items(self):
        return self.REGISTRY.items()
    
    def values(self):
        return self.REGISTRY.values()
    
    def update(self, m: Mapping[str, Type[T]]):
        self.REGISTRY.update(m)
    
    def __contains__(self, item):
        return self.REGISTRY.__contains__(item)
    
    def isin(self, key: str, raise_err: bool = True) -> bool:
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
        if key in self.REGISTRY:
            return True
        if raise_err:
            raise KeyError(
                f"Key '{key}' not found in {self.__class__.__name__}. Available keys: {list(self.REGISTRY.keys())}"
            )
        return False
    

# TODO: Rename everywhere to ClassRegistry or Type registry cleat
class Registry(Generic[T], RegistryMixin):
    """
    A generic registry to map string keys to classes. It supports registration
    via a decorator and creating instances of registered classes. By using
    a TypeVar, it allows static type checkers to infer the correct type
    of created instances.

    Examples
    --------
    ```python
        from abc import ABC

        class Animal(ABC):
            def speak(self):
                raise NotImplementedError

        # Note the generic type hint `ClassRegistry[Animal]`
        ANIMAL_REGISTRY = ClassRegistry(type=Animal)

        @ANIMAL_REGISTRY.register(key='dog', aliases=['hound'])
        class Dog(Animal):
            def speak(self):
                return "Woof!"

        # create() returns the correct type for static analysis
        my_dog: Animal = ANIMAL_REGISTRY.create('dog')
        print(my_dog.speak())  # Output: Woof!

        my_hound: Animal = ANIMAL_REGISTRY.create('hound')
        print(my_hound.speak()) # Output: Woof!
    ```
    
    Note
    ----
    This implementation is not thread-safe for concurrent writes, but this
    is generally not an issue as registration typically happens once on
    the main thread at startup. A lock could be added if needed.
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

    
    def register(
        self, key: str, aliases: Optional[List[str]] = None
    ) -> Callable[[Type[T]], Type[T]]:
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
            if not issubclass(cls, self.type):
                raise TypeError(
                    f"Registered class '{cls.__name__}' must be a subclass of '{self.type.__name__}'"
                )

            all_keys = [key] + (aliases or [])
            for k in all_keys:
                if k in self.REGISTRY:
                    raise ValueError(f"Key '{k}' is already registered.")

            for k in all_keys:
                self.REGISTRY[k] = cls

            return cls

        return decorator

    def __getitem__(self, key: str) -> Type[T]:
        """
        Retrieves a registered class by its key.

        Parameters
        ----------
        key : str
            The key of the class to retrieve.

        Returns
        -------
        Type[T]
            The class associated with the key.

        Raises
        ------
        KeyError
            If the key is not found.
        """
        try:
            return self.REGISTRY[key]
        except KeyError:
            raise KeyError(
                f"Key '{key}' not found in {self.__class__.__name__}. Available keys: {list(self.REGISTRY.keys())}"
            )
    
    def get(self, key: str, default: Optional[Type[T]] = None) -> Optional[Type[T]]:
        """
        Retrieves a registered class by its key. Returns a default value if not found.
        """
        return self.REGISTRY.get(key, default)
    
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


# ----- Instance Registry -----

# TODO: add lazy initialization of instances?
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
        self.REGISTRY: Dict[str, Type[T]] = {}


    def register(
        self, key: str, aliases: Optional[List[str]] = None
    ) -> Callable[[T], T]:
        """
        Returns a decorator to register an instance class with a given key and optional aliases.

        Parameters
        ----------
        key : str
            The primary key to associate with the class.
        aliases : Optional[List[str]]
            A list of alternative keys.

        Returns
        -------
        Callable[[T], T]
            A decorator that registers the class.

        Raises
        ------
        TypeError
            If the object is not instance of the registry's base class.
        KeyError
            If the key or any of the aliases are already registered.
        """

        def decorator(instance: Type[T]) -> Type[T]:
            if not isinstance(instance, self.type):
                raise TypeError(
                    f"Object `{instance.__name__}` is not an instance of"
                    f"`{self.type.__name__}`"
                )

            all_keys = [key] + (aliases or [])
            for k in all_keys:
                if k in self.REGISTRY:
                    raise KeyError(f"Key '{k}' is already registered.")

            for k in all_keys:
                self.REGISTRY[k] = instance

            return instance

        return decorator

    def __getitem__(self, key: str) -> T:        
        try:
            return self.REGISTRY[key]
        except KeyError:
            raise KeyError(
                f"Key '{key}' not found in {self.__class__.__name__}. Available keys: {list(self.REGISTRY.keys())}"
            )
    
    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """
        Retrieves a registered instance by its key. Returns a default value if not found.
        """
        return self.REGISTRY.get(key, default)

# ----- Callable Registry -----

class CallableRegistry(Generic[T], RegistryMixin):
    """
    A generic registry for mapping keys to callables (e.g., functions).
    It supports registration via a decorator and uses a TypeVar to allow
    static type checkers to infer the correct callable type.

    Example usage:
    ```python
        from typing import Protocol

        class Greeter(Protocol):
            def __call__(self, name: str) -> str: ...

        # The generic type hint is the protocol
        GREETING_REGISTRY = CallableRegistry(Greeter)

        @GREETING_REGISTRY.register(key='formal')
        def formal_greeting(name: str) -> str:
            return f"Hello, {name}."

        # get() or __getitem__ returns the correct callable type
        greeter_func = GREETING_REGISTRY.get('formal') # type is inferred
        print(greeter_func("Alice"))  # Output: Hello, Alice.
    ```
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
        self.type = type
        self.REGISTRY: Dict[Any, T] = {}

    # TODO: add signature check?
    def register(
        self, key: Any, aliases: Optional[List[Any]] = None
    ) -> Callable[[T], T]:
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

        def decorator(func: T) -> T:
            all_keys = [key] + (aliases or [])
            for k in all_keys:
                if k in self.REGISTRY:
                    raise KeyError(f"Key '{k}' is already registered.")

            for k in all_keys:
                self.REGISTRY[k] = func

            return func

        return decorator

    def __getitem__(self, key: Any) -> T:
        """
        Retrieves a registered callable by its key.

        Parameters
        ----------
        key : Any
            The key of the callable to retrieve.

        Returns
        -------
        T
            The callable associated with the key.

        Raises
        ------
        KeyError
            If the key is not found.
        """
        try:
            return self.REGISTRY[key]
        except KeyError:
            available_keys = ", ".join(map(str, self.keys()))
            raise KeyError(
                f"Key '{key}' not found in registry. Available keys: [{available_keys}]"
            )
    
    def get(self, key: Any, default: Optional[T] = None) -> Optional[T]:
        """
        Retrieves a registered callable by its key. Returns a default value if not found.
        """
        return self.REGISTRY.get(key, default)
