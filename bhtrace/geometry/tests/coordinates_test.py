import os
import sys
import inspect

import unittest
import torch

root_path = '/home/alexey/Work/bhtrace-dev'
sys.path.append(root_path)
sys.path.append(os.getcwd())

from bhtrace.geometry import Coordinates

class TestCoordinates(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        '''
        Collect all subclasses of Coordinates and instantate
        '''
        super(TestCoordinates, self).__init__(*args, **kwargs)

        self.base_class = Coordinates

        self.impls = {}

        all_impls = inspect.getmembers(
            sys.modules['bhtrace.geometry'], inspect.isclass
        )

        for name, obj in all_impls:
            if issubclass(obj, self.base_class) and obj is not self.base_class:
                self.impls[name] = obj
        
        keys_all = self.impls.keys()
        print(f'All impls: {keys_all}')

        keys_cross = (self.impls.keys())
        print(f'Supported impls: {keys_cross}')

        self.instance_to_test = {}
        for key in keys_all:
            try:
                self.instance_to_test[key] = self.impls[key]()
            except Exception as e:
                print(f'Cannot instantiate coordinate system {key}')
                    
        self.atol = 1e-4
        self.rtol = 1e-4

    
    def test_init(self):
        '''
        Test class initialization
        '''
        for key, obj in self.instance_to_test.items():
            self.assertIsInstance(obj, self.base_class, f"{key} is not a ")
            # self.assertIsNotNone(TF.inverse, f"{key} inverse is None")
            # self.assertIsInstance(TF.inverse, CoordinateTransformation, f"{key} inverse is not a CoordinateTransformation")
            # self.assertIs(TF.inverse.inverse, TF, f"{key} inverse's inverse is not itself")

if __name__ == '__main__':

    unittest.main()