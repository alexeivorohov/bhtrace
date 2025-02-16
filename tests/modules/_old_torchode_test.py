import matplotlib.pyplot as pp
import torch
import torchode as to

def f(t, y):
    return -0.5 * y

y0 = torch.tensor([[1.2], [5.0]])
n_steps = 10
t_eval = torch.stack((torch.linspace(0, 5, n_steps), torch.linspace(3, 4, n_steps)))

term = to.ODETerm(f)
step_method = to.Dopri5(term=term)
step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
solver = to.AutoDiffAdjoint(step_method, step_size_controller)
jit_solver = torch.compile(solver)

sol = jit_solver.solve(to.InitialValueProblem(y0=y0, t_eval=t_eval))
print(sol.stats)
# => {'n_f_evals': tensor([26, 26]), 'n_steps': tensor([4, 2]),
# =>  'n_accepted': tensor([4, 2]), 'n_initialized': tensor([10, 10])}

pp.plot(sol.ts[0], sol.ys[0])
pp.plot(sol.ts[1], sol.ys[1])
pp.show()