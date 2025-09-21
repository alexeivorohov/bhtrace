import unittest
import torch
from bhtrace.geometry import Observer, KerrSchild

class TestObserver(unittest.TestCase):
    def setUp(self):
        self.spacetime = KerrSchild(a=0.9)
        self.position = torch.tensor([0.0, 10.0, 0.5, 0.0])
        self.camera_dir = torch.tensor([-1.0, 0.0, 0.0])
        self.u = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.observer = Observer(
            spacetime=self.spacetime,
            position=self.position,
            camera_dir=self.camera_dir,
            u=self.u
        )

    def test_save_load(self):
        state = self.observer.state_dict()
        loaded_observer = Observer.from_dict(state)
        new_state = loaded_observer.state_dict()

        # Compare spacetime separately
        self.assertEqual(state['spacetime'], new_state['spacetime'])

        # Compare tensors
        self.assertTrue(torch.allclose(torch.tensor(state['position']), torch.tensor(new_state['position'])))
        self.assertTrue(torch.allclose(torch.tensor(state['camera_dir']), torch.tensor(new_state['camera_dir'])))
        self.assertTrue(torch.allclose(torch.tensor(state['u']), torch.tensor(new_state['u'])))

if __name__ == '__main__':
    unittest.main()
