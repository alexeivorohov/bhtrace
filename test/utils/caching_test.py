import unittest
import sys
import os

# Add project root to path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_path)

from bhtrace.utils.caching import Cacher

class TestCacher(unittest.TestCase):

    def test_instantiation(self):
        cacher = Cacher()
        self.assertTrue(cacher.should_cache)
        self.assertFalse(cacher.should_use_cache)

        cacher = Cacher(should_cache=False, should_use_cache=True)
        self.assertFalse(cacher.should_cache)
        self.assertTrue(cacher.should_use_cache)

    def test_caching_works(self):
        cacher = Cacher(should_cache=True, should_use_cache=True)
        
        class MyClass:
            def __init__(self):
                self.call_count = 0

            @cacher.attach
            def my_method(self):
                self.call_count += 1
                return 42
            
            def use_cached(self):
                return self.cacher['my_method']

        instance = MyClass()
        self.assertEqual(instance.call_count, 0)

        # First call
        result1 = instance.my_method()
        self.assertEqual(result1, 42)
        self.assertEqual(instance.call_count, 1)

        # Check that instance has cacher and it has the value
        self.assertTrue(hasattr(instance, 'cacher'))
        self.assertIn('my_method', instance.cacher)
        self.assertEqual(instance.cacher['my_method'], 42)
        self.assertEqual(instance.use_cached(), 42)

        # Second call, should be cached
        result2 = instance.my_method()
        self.assertEqual(result2, 42)
        self.assertEqual(instance.call_count, 1) # Should not increment

    def test_should_use_cache_false(self):
        cacher = Cacher(should_cache=True, should_use_cache=False)
        
        class MyClass:
            def __init__(self):
                self.call_count = 0

            @cacher.attach
            def my_method(self):
                self.call_count += 1
                return self.call_count # Return something that changes

        instance = MyClass()
        
        result1 = instance.my_method()
        self.assertEqual(result1, 1)
        self.assertEqual(instance.call_count, 1)

        # should_use_cache is False, so method is called again
        result2 = instance.my_method()
        self.assertEqual(result2, 2)
        self.assertEqual(instance.call_count, 2)

        # Now, let's enable use_cache and see if it returns the last cached value
        cacher.set_use_cache(True)
        result3 = instance.my_method()
        self.assertEqual(result3, 2) # Should be the last cached value
        self.assertEqual(instance.call_count, 2) # Should not increment

    def test_should_cache_false(self):
        cacher = Cacher(should_cache=False, should_use_cache=True)
        
        class MyClass:
            def __init__(self):
                self.call_count = 0

            @cacher.attach
            def my_method(self):
                self.call_count += 1
                return self.call_count

        instance = MyClass()
        
        result1 = instance.my_method()
        self.assertEqual(result1, 1)
        self.assertEqual(instance.call_count, 1)

        # should_cache is False, so not cached
        result2 = instance.my_method()
        self.assertEqual(result2, 2)
        self.assertEqual(instance.call_count, 2)

        # Check that instance.attach is empty
        self.assertTrue(hasattr(instance, 'cacher'))
        self.assertEqual(len(instance.cacher), 0)

    def test_multiple_instances(self):
        cacher = Cacher(should_cache=True, should_use_cache=True)
        
        class MyClass:
            def __init__(self):
                self.call_count = 0

            @cacher.attach
            def my_method(self):
                self.call_count += 1
                return id(self) # Return something unique to the instance

        instance1 = MyClass()
        instance2 = MyClass()

        result1_1 = instance1.my_method()
        self.assertEqual(result1_1, id(instance1))
        self.assertEqual(instance1.call_count, 1)

        result2_1 = instance2.my_method()
        self.assertEqual(result2_1, id(instance2))
        self.assertEqual(instance2.call_count, 1)

        # Call again, should be cached
        result1_2 = instance1.my_method()
        self.assertEqual(result1_2, id(instance1))
        self.assertEqual(instance1.call_count, 1)

        result2_2 = instance2.my_method()
        self.assertEqual(result2_2, id(instance2))
        self.assertEqual(instance2.call_count, 1)

        # Check that caches are separate
        self.assertNotEqual(instance1.cacher['my_method'], instance2.cacher['my_method'])

    def test_multiple_methods(self):
        cacher = Cacher(should_cache=True, should_use_cache=True)
        
        class MyClass:
            def __init__(self):
                self.method1_calls = 0
                self.method2_calls = 0

            @cacher.attach
            def method1(self):
                self.method1_calls += 1
                return 'one'
            
            @cacher.attach
            def method2(self):
                self.method2_calls += 1
                return 'two'

        instance = MyClass()
        
        instance.method1()
        instance.method2()
        self.assertEqual(instance.method1_calls, 1)
        self.assertEqual(instance.method2_calls, 1)

        instance.method1()
        instance.method2()
        self.assertEqual(instance.method1_calls, 1)
        self.assertEqual(instance.method2_calls, 1)

        self.assertIn('method1', instance.cacher)
        self.assertIn('method2', instance.cacher)

    def test_context_managers(self):
        cacher = Cacher(should_cache=False, should_use_cache=False)
        
        class MyClass:
            def __init__(self):
                self.call_count = 0

            @cacher.attach
            def my_method(self):
                self.call_count += 1
                return self.call_count

        instance = MyClass()

        # Test nocache (default)
        instance.my_method()
        instance.my_method()
        self.assertEqual(instance.call_count, 2)
        self.assertEqual(len(instance.cacher), 0)

        # Test cache context manager
        with cacher.cache():
            instance.my_method() # call 3, should cache value 3
            instance.my_method() # call 4, should cache value 4, but not use cache
        self.assertEqual(instance.call_count, 4)
        self.assertEqual(instance.cacher['my_method'], 4)

        # Test usecache context manager
        with cacher.usecache():
            instance.my_method() # should use cache, value 4
            instance.my_method() # should use cache, value 4
        self.assertEqual(instance.call_count, 4) # no new calls

        # Check that state is restored
        instance.my_method() # call 5
        self.assertEqual(instance.call_count, 5)
        # cache is not updated because should_cache is back to False
        self.assertEqual(instance.cacher['my_method'], 4)


if __name__ == '__main__':
    unittest.main()