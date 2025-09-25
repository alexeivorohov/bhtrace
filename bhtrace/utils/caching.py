import functools
from contextlib import contextmanager

class Cacher:
    """
    An instantiable class to control caching for methods.
    An instance of this class can be used as a decorator factory.
    The decorator will wrap the results of method calls. The wrap
    itself is stored in a dictionary attached to each instance of the
    decorated class.
    Usage:
        cacher = Cacher()
        class MyClass:
            @cacher.attach
            def my_method(self):
                # expensive computation
                return 42
    """

    def __init__(self, should_cache=True, should_use_cache=False):
        """
        Initializes a new Cacher instance.
        By default, cahches outputs, but not uses them on second calls

        Args:
            should_cache (bool): If True, results of decorated methods will be cached.
            should_use_cache (bool): If True, cached results will be returned when available.
        """
        self.should_cache = should_cache
        self.should_use_cache = should_use_cache

    def set_caching(self, should_cache: bool):
        """
        Enable or disable caching of results for decorated methods.
        If disabled, the method will be executed every time, and the result will not be stored.
        """
        self.should_cache = should_cache

    def set_use_cache(self, should_use_cache: bool):
        """
        Enable or disable returning results from wrap.
        If disabled, the method will be executed every time. The result will still be cached
        if caching is enabled.
        """
        self.should_use_cache = should_use_cache

    @contextmanager
    def nocache(self):
        '''Context manager to temporarily disable caching and using cache.'''
        original_should_cache = self.should_cache
        original_should_use_cache = self.should_use_cache
        self.should_cache = False
        self.should_use_cache = False
        try:
            yield
        finally:
            self.should_cache = original_should_cache
            self.should_use_cache = original_should_use_cache

    @contextmanager
    def cache(self):
        '''Context manager to temporarily enable caching, but not use cache.'''
        original_should_cache = self.should_cache
        original_should_use_cache = self.should_use_cache
        self.should_cache = True
        self.should_use_cache = False
        try:
            yield
        finally:
            self.should_cache = original_should_cache
            self.should_use_cache = original_should_use_cache

    @contextmanager
    def usecache(self):
        '''Context manager to temporarily enable caching and using cache.'''
        original_should_cache = self.should_cache
        original_should_use_cache = self.should_use_cache
        self.should_cache = True
        self.should_use_cache = True
        try:
            yield
        finally:
            self.should_cache = original_should_cache
            self.should_use_cache = original_should_use_cache

    def attach(self, func):
        """
        A decorator that caches the result of a method.
        The caching behavior is controlled by this Cacher instance.
        This decorator is intended for methods that do not take arguments (other than `self`)
        and whose result depends only on the immutable state of the instance.
        """
        method_name = func.__name__
        cacher_factory = self

        @functools.wraps(func)
        def wrapper(decorated_instance):
            if not hasattr(decorated_instance, 'cacher'):
                decorated_instance.cacher = {}
            
            instance_cache = decorated_instance.cacher

            if cacher_factory.should_use_cache and method_name in instance_cache:
                return instance_cache[method_name]

            result = func(decorated_instance)

            if cacher_factory.should_cache:
                instance_cache[method_name] = result
            
            return result
        return wrapper

cacher = Cacher()