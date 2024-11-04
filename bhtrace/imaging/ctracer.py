import torch
import os
import pickle
import torchode as tode

from ..geometry import Spacetime

class CTracer():

  def __init__(self):

    self.solv = 'CTracer'
    self.m_param = None

    self.Ni = 0
    self.Nt = 0
    self.t = 0

    self.X = None
    self.P = None
    self.X0 = None
    self.P0 = None

    pass


  def spacetime_set(self, spacetime: Spacetime):
    '''
    Attach class of spacetime to be traced

    ## Input:
    spacetime: Spacetime        
    '''

    self.spacetime = spacetime


  def step_size(self, X, P, gX, dgX):


    return dt


  def evnt_check(self, X, P):


    return mask

    
  def __step__(self,  X: torch.Tensor, P: torch.Tensor, dt):

    conn = self.spacetime.conn(X)

    dP = - torch.einsum('bmuv,bu,bv->bm', conn, P, P)
    dX = P*dt

    P += dP
    X += dX

    # P = self.pcle.normp(X, P)

    return X, P

        
  def trace(self, X0, P0, eps, nsteps, dt):
    '''
    
    '''
    self.X0 = X0
    self.P0 = P0
    self.Nt = nsteps
    self.Ni = X0.shape[0]

    self.eps = 1e-5

    self.X = torch.zeros(nsteps, self.Ni, 4)
    self.P = torch.zeros(nsteps, self.Ni, 4)

    self.X[0, :, :] = X0
    self.P[0, :, :] = P0
    X, P = X0, P0

    for i in range(nsteps-1):

        X, P = self.__step__(X, P, dt)

        self.X[i+1, :, :] = X
        self.P[i+1, :, :] = P

    return self.X, self.P


    def save(self, filename, directory=None):
      '''
      Save last result to a file specified by filename and optional directory.

      Parameters:
      - filename: The name of the file or full path.
      - directory: Optional; directory path to save the file in.

      Returns:
      - str: The full path of the saved file.
      '''

      if directory:
          full_path = os.path.join(directory, filename)
      else:
          
          full_path = filename

      result = {
          'spacetime': None,
          'solv': self.solv, 
          'm_param': self.m_param,
          'Ni': self.Ni,
          'Nt': self.Nt,
          't': self.t,
          'X0': self.X0,
          'P0': self.P0,
          'X': self.X,
          'P': self.P}


      with open(full_path, 'wb') as file:
          pickle.dump(result, file)

      return full_path

  # old method
  def __eq__(self, t, XP):

    X_, P_ = XP[:, 0:4], XP[:, 4:]

    G = self.spacetime.dg(X_)
    ginvX = self.spacetime.ginv(X_)

    dP = - torch.einsum('bmuv,bu,bv->bm', G, P_, P_)

    # dP = torch.zeros_like(P_)
    # for i in range(P_.shape[0]):
    #   if True:
    #     dP[i] = - G[i] @ P_[i] @ P_[i]

    return torch.cat([P_, dP], axis=1)

  # old method
  def solve(self, X0, P0, t_sim, n_steps, dev='cpu'):

    N_tr = X0.shape[0]
    XP0 = torch.cat([X0, P0], axis=1)

    # Задаём тензор начальных времён
    t_eval = torch.linspace(0, t_sim, n_steps).reshape(1, -1)
    t_eval = torch.kron(t_eval, torch.ones(N_tr).view(N_tr, 1))

    # Отправляем тензор начальных условий и временную сетку в память устройства,
    # на котором хотим производить вычисления
    XP0.to(dev)
    t_eval.to(dev)

    # Инициализируем ДУ из нашей функции
    term = tode.ODETerm(self.__eq__)

    # Выбираем решатель и контроллер шага
    step_method = tode.Dopri5(term=term)
    step_size_controller = tode.IntegralController(atol=1e-6, rtol=1e-3, term=term)

    solver = tode.AutoDiffAdjoint(step_method, step_size_controller)

    # Выполняем jit-компиляцию кода решателя напрямую в машинный код, чтобы
    # избежать затрат на интерпретацию при каждом вызове
    jit_solver = torch.compile(solver)

    # Запускаем решатель
    sol = jit_solver.solve(tode.InitialValueProblem(y0=XP0, t_eval=t_eval))

    return sol

# # OK
# # Процесс трассировки
# def tracer(INITXYZ, th_obs=0.5, device='cpu', C = 0.45, h = 2e6):

#   # Общие начальные условия:
#   xx, yy, zz, vx, vy, vz = INITXYZ
#   r, th, ph, vr, vth, vph = cart2sph([xx, yy, zz, vx, vy, vz])

#   t0 = torch.zeros_like(r)
#   vt = torch.ones_like(r)

#   Y0 = torch.cat([t0, r, th, ph, vt, vr, vth, vph], axis=1)

#   t_eval = torch.linspace(0, 80, 200).reshape(1, -1)

#   disc_normv = sph2cart(torch.Tensor([1, th_obs, 0, 0, 0, 0]))

#   # Считаем траектории РН и получаем картинку

#   C2=C**2

#   fRN = lambda r: 1 - 2/r + C2/(r**2)
#   dfRN = lambda r: 2/(r**2) - 2*C2/(r**3)

#   eq_r_p = lambda r: r*dfRN(r)-2*fRN(r)

#   r_H_RN = float(fsolve(fRN, 2))
#   r_p_RN = float(fsolve(eq_r_p, 3))
#   b_cr_RN = float(r_p_RN/np.sqrt(fRN(r_p_RN)))

#   EQ_ = lambda t, X: EQ_func(t, X, fRN, dfRN, r_H_RN, 20)
#   ResRN = imcomp(EQ_, Y0, t_eval, dev=device)

#   Y = ResRN.ys
#   RN_r = Y[:, :, 1]
#   RN_x = Y[:, :, 1]*torch.sin(Y[:, :, 2])*torch.cos(Y[:, :, 3])
#   RN_y = Y[:, :, 1]*torch.sin(Y[:, :, 2])*torch.sin(Y[:, :, 3])
#   RN_z = Y[:, :, 1]*torch.cos(Y[:, :, 2])

#   i2_RN = lambda r: i2_r(r, fRN, r_p_RN)
#   RN_I2 = thin_disk(RN_r, RN_x, RN_y, RN_z, disc_normv[:3], i2_RN, r_H_RN)

#   # Считаем траектории модифицированной модели и получаем картинку
#   B_EH = float(1/225/np.pi/np.sqrt(137)*C**2*h)
#   # print(B_EH)

#   fEH = lambda r: 1 - 2/r + C/(r**2) - B_EH/(r**6)
#   dfEH = lambda r: 2/(r**2) - 2*C/(r**3) + 7*B_EH/(r**7)

#   eq_r_p = lambda r: r*dfEH(r)-2*fEH(r)
#   r_H_EH = float(fsolve(fEH, 2))
#   r_p_EH = float(fsolve(eq_r_p, 3))
#   b_cr_EH = float(r_p_EH/np.sqrt(fEH(r_p_EH)))

#   EQ_ = lambda t, X: EQ_func(t, X, fEH, dfEH, r_H_EH, 20)
#   ResEH = imcomp(EQ_, Y0, t_eval, dev=device)

#   Y = ResEH.ys
#   EH_x = Y[:, :, 1]*torch.sin(Y[:, :, 2])*torch.cos(Y[:, :, 3])
#   EH_y = Y[:, :, 1]*torch.sin(Y[:, :, 2])*torch.sin(Y[:, :, 3])
#   EH_z = Y[:, :, 1]*torch.cos(Y[:, :, 2])

#   i2_EH = lambda r: i2_r(r, fEH, r_p_EH)
#   EH_I2 = thin_disk(Y[:, :, 1], EH_x, EH_y, EH_z, disc_normv[:3], i2_EH, r_H_EH)

#   outp = {'I2_EH': EH_I2, 'I2_RN': RN_I2, 'b.c.': INITXYZ, \
#           'ResEH': ResEH, 'ResRN': ResRN, 'res': r.shape, 'C':C, 'h': h}

#   return outp


# def trace(rng):
#   prefix = 'Sim_{}'.format(rng) + '.pickle'
#   print(prefix + 'in process')
#   INITXYZ = sample('circle', db=[8, 10, 0, 20], rng = rng, D0 = 15, dth=0)
#   sim_data = Sim(INITXYZ, device='cpu')
  

#   with open(prefix, 'wb') as handle:
#     pickle.dump(sim_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

#   return sim_data
