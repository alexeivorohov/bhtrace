import torch
import torchode as tode
import time

from ..geometry import Spacetime, Particle
from .tracer import Tracer
from ..functional import print_status_bar

class CTracer(Tracer):

  def __init__(self, ode_scheme='Euler'):

    self.name = 'CTracer'
    self.m_param = None

    pass

  
  def __term__(self, t, XP):

    X, P = XP[0, :4], XP[0, 4:]
    G_ = self.spc.conn(X)

    dX = P
    dP = - G_ @ P @ P

    return torch.cat([dX, dP]).view(1, 8)

        
  def forward(self, 
    particle: Particle,
    X0, 
    P0,
    T,
    nsteps = 128,
    r_max = 30.0,
    max_proper_t=500.0,
    dev = 'cpu',
    ):
    '''
    
    '''

    self.particle = particle
    self.spc = particle.spacetime


    self.Nt = nsteps
    self.Ni = X0.shape[0]
    self.X = torch.zeros(nsteps, self.Ni, 4)
    self.P = torch.zeros(nsteps, self.Ni, 4)
    self.X[0, :, :] = X0
    self.P[0, :, :] = P0

    N_tr = X0.shape[0]
    XP0 = torch.cat([X0, P0], axis=1)
    print(XP0.shape)

    # Задаём тензор начальных времён
    t_eval = torch.linspace(0, T, nsteps).reshape(1, -1)

    # Инициализируем ДУ из нашей функции
    term = tode.ODETerm(self.__term__)

    # Отправляем тензор начальных условий и временную сетку в память устройства,
    # на котором хотим производить вычисления
    XP0.to(dev)
    t_eval.to(dev)

    # Выбираем решатель и контроллер шага
    step_method = tode.Euler(term=term)
    step_size_controller = tode.IntegralController(atol=1e-6, rtol=1e-3, term=term)

    solver = tode.AutoDiffAdjoint(step_method, step_size_controller)

    # Выполняем jit-компиляцию кода решателя напрямую в машинный код, чтобы
    # избежать затрат на интерпретацию при каждом вызове
    jit_solver = torch.compile(solver)

    start_time = time.time()
    for i in range(self.Ni):
      Y0 = XP0[i, :].view(1, 8)
      sol = jit_solver.solve(tode.InitialValueProblem(y0=Y0, t_eval=t_eval))


      self.X[:, i, :] = sol.ys[0, :, :4]
      self.P[:, i, :] = sol.ys[0, :, 4:]
      elapsed_time = time.time() - start_time
      print_status_bar(i, self.Ni, elapsed_time)
    print('\n Done!')

    return self.X, self.P
