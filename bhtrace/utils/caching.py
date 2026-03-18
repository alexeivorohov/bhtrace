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

    1. Create cacher instance and attach function:

        cacher = Cacher()
        @cacher.attach
        def my_method():
            # expensive computation
            return 42

    2. Use context managers to control attached function behaviour:

        # This will not save values to cache and not retrieve them
        with cacher.nocache():
            my_method()

        # This will save values to cache but not retrieve them
        with cacher.cache():
            my_method()

        # This will try to retrieve values from cache on every function call
        with cacher.usecache():
            my_method()
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

        Note: This decorator's caching is only safe for methods that do not
        take arguments (other than `self`). It uses the method name as the
        cache key and will return incorrect results for methods whose output
        depends on arguments.
        """
        method_name = func.__name__
        cacher_factory = self

        @functools.wraps(func)
        def wrapper(decorated_instance, *args, **kwargs):
            if not hasattr(decorated_instance, 'cacher'):
                decorated_instance.cacher = {}
            
            instance_cache = decorated_instance.cacher

            if cacher_factory.should_use_cache and method_name in instance_cache:
                return instance_cache[method_name]

            result = func(decorated_instance, *args, **kwargs)

            if cacher_factory.should_cache:
                instance_cache[method_name] = result
            
            return result
        return wrapper

CACHE = Cacher()
'''
Global cacher, for usage details, see Cacher()
'''
# TODO: move to globs