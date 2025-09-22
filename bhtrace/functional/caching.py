import functools

class Cacher:
    """
    A class to control caching for methods decorated with @Cacher.cache.
    """
    _should_cache = True
    _should_use_cache = False

    @classmethod
    def set_caching(cls, should_cache: bool):
        """
        Globally enable or disable caching of results for decorated methods.
        If disabled, the method will be executed every time, and the result will not be stored.
        """
        cls._should_cache = should_cache

    @classmethod
    def set_use_cache(cls, should_use_cache: bool):
        """
        Globally enable or disable returning results from cache.
        If disabled, the method will be executed every time. The result will still be cached
        if caching is enabled.
        """
        cls._should_use_cache = should_use_cache

    @staticmethod
    def cache(func):
        """
        A decorator that caches the result of a method `x` into a property `x_`.
        The caching behavior is controlled by Cacher.set_caching and Cacher.set_use_cache.

        This decorator is intended for methods that do not take arguments (other than `self`)
        and whose result depends only on the immutable state of the instance.
        """
        method_name = func.__name__
        property_name = f"{method_name}_"

        @functools.wraps(func)
        def wrapper(self): # 'self' is the instance of the decorated class
            if Cacher._should_use_cache and hasattr(self, property_name):
                return getattr(self, property_name)

            result = func(self)

            if Cacher._should_cache:
                setattr(self, property_name, result)
            
            return result
        
        return wrapper
