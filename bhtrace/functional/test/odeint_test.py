import unittest
import torch
import math
import sys
import os

# Add project root to path
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)

from bhtrace.functional.odeint import Euler, RK4, ODEint

_SOLVERS_ = {
    'Euler': Euler,
    'RK4': RK4,
}

class TestODEint(unittest.TestCase):

    def setUp(self):
        """Set up test parameters."""
        self.solvers = {name: constructor(dt=0.1) for name, constructor in _SOLVERS_.items()}
        self.n_steps = 10
        self.y0 = (torch.tensor([1.0], dtype=torch.float64),)
        self.t0 = 0.0

    def test_init(self):
        """Test if all solvers can be initialized."""
        for name, constructor in _SOLVERS_.items():
            try:
                solver = constructor(dt=0.1)
                self.assertIsInstance(solver, ODEint)
            except Exception as e:
                self.fail(f"Failed to initialize {name}: {e}")

    def test_simple_ode_integration(self):
        """
        Tests integration for a simple ODE (dy/dt = y) and compares accuracy.
        """
        def term(t, y):
            return (y[0],)

        analytical_solution = math.exp(self.n_steps * 0.1)
        results = {}

        for name, solver in self.solvers.items():
            solution = solver.forward(term, self.y0, self.t0, self.n_steps)
            y_final = solution['Y'][-1][0].item()
            results[name] = y_final

        # Check accuracy relative to analytical solution
        euler_error = abs(results['Euler'] - analytical_solution)
        rk4_error = abs(results['RK4'] - analytical_solution)

        self.assertLess(rk4_error, euler_error, "RK4 should be more accurate than Euler.")
        self.assertAlmostEqual(results['RK4'], analytical_solution, delta=1e-5, msg="RK4 should be very accurate.")

    def test_t_dependent_ode(self):
        """
        Tests integration for a time-dependent ODE (dy/dt = 2*t).
        """
        def term(t, y):
            return (torch.full_like(y[0], 2.0 * t),)

        y0 = (torch.tensor([0.0], dtype=torch.float64),)
        t_final = self.n_steps * 0.1

        for name, solver in self.solvers.items():
            solution = solver.forward(term, y0, self.t0, self.n_steps)
            y_final = solution['Y'][-1][0].item()
            
            self.assertTrue(isinstance(y_final, float))

    def test_solver_reusability(self):
        """
        Tests if a solver instance can be reused for different ODE problems.
        """
        solver = Euler(dt=0.01)

        # Problem 1: dy/dt = y
        def term1(t, y):
            return (y[0],)
        y0_1 = (torch.tensor([1.0]),)
        sol1 = solver.forward(term1, y0_1, 0.0, 10)
        y_final1 = sol1['Y'][-1][0].item()
        self.assertGreater(y_final1, 1.0)

        # Problem 2: dy/dt = -y
        def term2(t, y):
            return (-y[0],)
        y0_2 = (torch.tensor([1.0]),)
        sol2 = solver.forward(term2, y0_2, 0.0, 10)
        y_final2 = sol2['Y'][-1][0].item()
        self.assertLess(y_final2, 1.0)

    def test_event_function(self):
        """
        Tests the event function mechanism.
        """
        def term(t, y):
            return (torch.tensor([1.0]),) # dy/dt = 1 -> y(t) = t

        def event_fn(t, Y):
            return torch.tensor(t > 5.0)

        solver = Euler(dt=1.0, event_fn=event_fn)
        solution = solver.forward(term, (torch.tensor([0.0]),), 0.0, 10)

        self.assertEqual(len(solution['t']), 7)
        self.assertAlmostEqual(solution['t'][-1], 6.0)


if __name__ == '__main__':
    unittest.main()