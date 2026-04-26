# import unittest
# import torch
# import math
# from abc import abstractmethod
# from typing import Tuple, Dict
# import uniplot as uplt

# from bhtrace.utils.odeint import Euler, RK4, ODEint

# _SOLVERS_ = {
#     'Euler': Euler,
#     'RK4': RK4,
# }

# torch.random.manual_seed(42)

# # ToDo: fix shape misalignment between exact y and result y

# class ODETestProblem:

#     def __init__(self, y0: Tuple[torch.Tensor] = None):

#         if y0 == None:
#             self.batch = 5
#             self.shape = (self.batch, 1)
#             y0 = (torch.randn(*self.shape), )
#         else:
#             self.y0 = y0
#             self.batch = y0[0].shape[0]
#             self.shape = (self.batch, )

#         self.y0 = y0
#         self.t0 = torch.tensor([0.0])
#         self.tview = (self.batch, -1)
        
#     @abstractmethod
#     def term(self, t, Y):
#         return None

#     @abstractmethod
#     def y(self, t):
#         return None    
    
#     @abstractmethod
#     def dy(self, t):
#         return None


# class Linear(ODETestProblem):

#     def __init__(self, y0=None, v0=None):
#         '''
#         :param y0: - Tuple[torch.Tensor] - 
#         :param v0: - torch.Tensor
#         '''
#         super().__init__(y0)
#         if v0 == None:
#             v0 = torch.randn(*self.shape)

#         self.v0 = v0
        
#     def term(self, t, Y):
        
#         return tuple(self.v0 for y_ in self.y0)

#     def dy(self, t):

#         return tuple(self.v0 for y_ in self.y0)

#     def y(self, t):
        
#         return tuple(y_ + self.v0*t for y_ in self.y0)


# class Exponent(ODETestProblem):

#     def __init__(self, y0=None, k0 = None):
#         super().__init__(y0)
#         if k0 == None:
#             k0 = torch.randn(*self.shape)
#         self.k0 = k0
        
#     def term(self, t, y):

#         return (y * self.k0, )

#     def y(self, t):

#         return (self.y0[0]*torch.exp(self.k0*(t-self.t0)),)

#     def dy(self, t):

#         return (self.y0[0]*self.k0*torch.exp(self.k0*(t-self.t0)), )
    

# class Oscillator(ODETestProblem):

#     def __init__(self, x0: torch.Tensor = None, p0: torch.Tensor = None, k0: torch.Tensor = None):
#         '''
#         :param x0: torch.Tensor - initial position
#         :param p0: torch.Tensor - initial impulse
#         :param k0: torch.Tensor - oscillator stiffness
#         '''

#         self.shape = (5, 1)
#         if x0 is None:
#             x0 = torch.randn(*self.shape)
#         if p0 is None:
#             p0 = torch.randn(*self.shape)
#         if k0 is None:
#             k0 = abs(torch.randn(*self.shape)) + 1e-2
 
#         super().__init__(y0=(x0, p0))
#         self.k0 = k0

#         self.omega = torch.sqrt(k0)
#         self.A0 = torch.sqrt(x0**2 + p0**2 / k0)
#         # Add a small epsilon to A0 to avoid division by zero
#         if torch.any(self.A0 == 0):
#             self.A0[self.A0 == 0] = 1e-9
#         self.phi0 = torch.atan2(x0, p0 / self.omega) - self.omega * self.t0

#     def term(self, t, x, p):

#         return p, (- self.k0 * x)

#     def y(self, t):
        
#         phi = self.omega * t + self.phi0
#         x = self.A0 * torch.sin(phi)
#         p = self.A0 * self.omega * torch.cos(phi)
#         return (x, p)

#     def dy(self, t):
#         phi = self.omega * t + self.phi0
#         dx = self.A0 * self.omega * torch.cos(phi)
#         dp = -self.A0 * self.omega**2 * torch.sin(phi)
#         return (dx, dp)
    

# def GeodSchw(ODETestProblem):

#     def __init__(self, X0, Y0, mu0):
#         pass

#     def term(self, t, X, P):
#         pass


# class TestODEint_logic(unittest.TestCase):

#     def test_init(self):
#         '''
#         Test if all solvers can be initialized.
#         '''
#         for name, constructor in _SOLVERS_.items():
#             try:
#                 solver = constructor(dt=0.1)
#                 self.assertIsInstance(solver, ODEint)
#             except Exception as e:
#                 self.fail(f"Failed to initialize {name}: {e}")
            
        


#     # def test_set_event_fn(self):
#     #     '''
#     #     '''
#     #     pass

#     # def test_set_step_fn(self):
#     #     '''
        
#     #     '''

#     #     pass

#     # def test_track_LTE(self):
#     #     '''
        
#     #     '''

#     #     pass

#     # def test_set_adjoints(self):
#     #     '''
        
#     #     '''
#     #     pass
    
#     # def test_LTE(self):

#     #     pass

#     # def test_state_dict(self):

#     #     pass

#     # def test_load_state_dict(self):

#     #     pass

#     # def test_to(self):

#     #     pass

#     # def test_factory(self):

#     #     pass



# class TestODEint_problems(unittest.TestCase):

#     def setUp(self):
#         """Set up test parameters."""

#         self.problems: Dict[str, ODETestProblem] = {
#             'linear': Linear(),
#             'exponent': Exponent(),
#             'oscillator': Oscillator()
#         }

#         self.solvers: Dict[str, ODEint] = {name: constructor(dt=1e-2) for name, constructor in _SOLVERS_.items()}

#         self.dtype = torch.float64
#         self.tol = 2e-2

#         pass
        

#     def test_1step(self):
#         '''
#         Test if solver is capable to correctly perform step for a simple terms:
#         - Checks if the derivative and new value have proper shape and are not nan,
#         - Checks if the derivative and new value are close to analytical results
#         '''
#         for solver_name, s in self.solvers.items():
#             for problem_name, p in self.problems.items():

#                 case = f'solver:{solver_name}, task:{problem_name}'

#                 s.term = p.term
#                 s.n_vars = len(p.y0)
#                 s.to(dtype=self.dtype)
#                 try:
#                     y_new, dy = s.__step__(p.t0, p.y0)
#                     dy_true = p.dy(p.t0)
#                     y_true = p.y(p.t0 + s.dt)
#                 except:
#                     self.assertTrue(False, f'Problem arised when tried to evaluate term for {case}')
                
#                 shape_consistnecy = [y_.shape == y0_.shape for y_, y0_ in zip(y_new, p.y0)]
#                 self.assertTrue(all(shape_consistnecy), 'Y0 shape')

#                 dy_nan = [torch.any(torch.isnan(dy_)) for dy_ in dy]
#                 self.assertFalse(any(dy_nan), case + ' dy contains torch.nan ')
                
#                 y_nan = [torch.any(torch.isnan(y_)) for y_ in y_new]
#                 self.assertFalse(any(y_nan), case + ' dy contains torch.nan ')
                
#                 dy_err = sum([abs(dy_ - dyt_).mean() for dy_, dyt_ in zip(dy, dy_true)])
#                 self.assertLess(dy_err, self.tol, case + f' dy error exceeds given tolerance: {dy_err} > {self.tol}')

#                 y_err = sum([abs(y_ - yt_).mean() for y_, yt_ in zip(y_new, y_true)])
#                 self.assertLess(y_err, self.tol, case + f' y error exceeds given tolerance: {y_err} > {self.tol}')

#         pass

#     # def test_post_step(self):
#     #     '''
#     #     Test if all post-step logic is done correctly.

#     #     '''
#     #     pass


#     def test_2problems(self):
#         '''
#         '''
#         nsteps = 64
#         for solver_name, s in self.solvers.items():
#             for problem_name, p in self.problems.items():
                
#                 case = f'solver:{solver_name}, task:{problem_name}'

#                 solution = s.forward(p.term, p.y0, p.t0, nsteps)
                
#                 result = solution['Y']
#                 exact = p.y(solution['t'])
#                 print(result[0].shape, exact[0].shape)
#                 err = [abs(y_ - yt_).mean() for y_, yt_ in zip(result, exact)]
#                 criterion_Err = sum(err) > self.tol
#                 if criterion_Err:
#                     for i in range(s.n_vars):
#                         if err[i] > self.tol:
#                             for j in range(s.batch_size):
#                                 ys = [
#                                     result[i][j,...].view(-1).cpu().numpy(),
#                                     exact[i][j,...].view(-1).cpu().numpy()
#                                 ]

#                                 xs = solution['t'].view(-1).cpu().numpy()

#                                 uplt.plot(xs=xs,
#                                           ys=ys,
#                                           lines=True,
#                                           character_set = 'braille',
#                                           legend_labels=['result', 'exact'],
#                                           title=f'Result vs Exact (var_n:{i}, batch_n:{j}, {case})'
#                                           )
#                                 # print(f'Result: {ys[0]} \n Exact: {ys[1]}')
 
                            
#                 self.assertFalse(criterion_Err,
#                                 f'\n For {case} mean error is greater than given tolerance: {err} > {self.tol} \n')

#     # def test_t_dependent_ode(self):
#     #     """
#     #     Tests integration for a time-dependent ODE (dy/dt = 2*t).
#     #     """
#     #     def term(t, y):
#     #         return (torch.full_like(y, 2.0 * t),)

#     #     y0 = (torch.zeros(1, 1, dtype=torch.float64),)
#     #     t_final = self.n_steps * 0.1

#     #     for name, solver in self.solvers.items():
#     #         solution = solver.forward(term, y0, self.t0, self.n_steps)
#     #         y_final = solution['Y'][-1][0].item()
            
#     #         self.assertTrue(isinstance(y_final, float))

#     # def test_reusability(self):
#     #     """
#     #     Tests if a solver instance can be reused for different ODE problems.
#     #     """
#     #     solver = Euler(dt=0.01)

#     #     # Problem 1: dy/dt = y
#     #     def term1(t, y):
#     #         return (y[0],)
#     #     y0_1 = (torch.ones(1, 1),)
#     #     sol1 = solver.forward(term1, y0_1, 0.0, 10)
#     #     y_final1 = sol1['Y'][-1][0].item()
#     #     self.assertGreater(y_final1, 1.0)

#     #     # Problem 2: dy/dt = -y
#     #     def term2(t, y):
#     #         return (-y[0],)
#     #     y0_2 = (torch.ones(1, 1),)
#     #     sol2 = solver.forward(term2, y0_2, 0.0, 10)
#     #     y_final2 = sol2['Y'][-1][0].item()
#     #     self.assertLess(y_final2, 1.0)

#     # def test__function(self):
#     #     """
#     #     Tests the event function mechanism.
#     #     """
#     #     def term(t, y):
#     #         return (torch.tensor([1.0]),) # dy/dt = 1 -> y(t) = t

#     #     def event_fn(t, Y):
#     #         return torch.tensor(t > 5.0)

#     #     solver = Euler(dt=1.0, event_fn=event_fn)
#     #     solution = solver.forward(term, (torch.ones(1, 1), ), 0.0, 10)

#     #     self.assertEqual(len(solution['t']), 7)
#     #     self.assertAlmostEqual(solution['t'][-1], 6.0)


# if __name__ == '__main__':
#     unittest.main()
