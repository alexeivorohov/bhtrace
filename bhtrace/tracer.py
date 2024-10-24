import torch
import torchode as tode


class tracer:

  def __init__(self, dev='cpu',):
    pass

  def equation(XP, r_h, r_lim):
    '''
    Equation function

    ### Input:
    XP: torch.Tensor [:, 8] - points in phase space
    r_h 
    '''
    # Avoiding strategy may be changed to more general:
    # Event(X) and Replacement

    X_ = X[:, 0:4]
    Y_ = X[:, 4:]

    G = GMA_spq(X_)

    outp = torch.zeros_like(X)

    for i in range(X.shape[0]):
        if (X[i, 1]>r_H) & (X[i, 1] < r_lim):
            dY = - G[i] @ Y_[i] @ Y_[i]
            res[i] = torch.cat([Y_[i], dY])

    return res

  def imcomp(EQ_, Y0, t_eval, dev='cpu'):

    # Определяем размер множества НУ
    N_tr = Y0.shape[0]

    # Задаём тензор начальных времён
    t_eval = torch.kron(t_eval, torch.ones(N_tr).view(N_tr, 1))

    # Отправляем тензор начальных условий и временную сетку в память устройства,
    # на котором хотим производить вычисления
    Y0.to(dev)
    t_eval.to(dev)

    # Инициализируем ДУ из нашей функции
    term = tode.ODETerm(EQ_)

    # Выбираем решатель и контроллер шага
    step_method = tode.Dopri5(term=term)
    step_size_controller = tode.IntegralController(atol=1e-6, rtol=1e-3, term=term)

    solver = tode.AutoDiffAdjoint(step_method, step_size_controller)

    # Выполняем jit-компиляцию кода решателя напрямую в машинный код, чтобы
    # избежать затрат на интерпретацию при каждом вызове
    jit_solver = torch.compile(solver)

    # Запускаем решатель
    sol = jit_solver.solve(tode.InitialValueProblem(y0=Y0, t_eval=t_eval))
    
    return sol


# OK
# Процесс трассировки
def tracer(INITXYZ, th_obs=0.5, device='cpu', C = 0.45, h = 2e6):

  # Общие начальные условия:
  xx, yy, zz, vx, vy, vz = INITXYZ
  r, th, ph, vr, vth, vph = cart2sph([xx, yy, zz, vx, vy, vz])

  t0 = torch.zeros_like(r)
  vt = torch.ones_like(r)

  Y0 = torch.cat([t0, r, th, ph, vt, vr, vth, vph], axis=1)

  t_eval = torch.linspace(0, 80, 200).reshape(1, -1)

  disc_normv = sph2cart(torch.Tensor([1, th_obs, 0, 0, 0, 0]))

  # Считаем траектории РН и получаем картинку

  C2=C**2

  fRN = lambda r: 1 - 2/r + C2/(r**2)
  dfRN = lambda r: 2/(r**2) - 2*C2/(r**3)

  eq_r_p = lambda r: r*dfRN(r)-2*fRN(r)

  r_H_RN = float(fsolve(fRN, 2))
  r_p_RN = float(fsolve(eq_r_p, 3))
  b_cr_RN = float(r_p_RN/np.sqrt(fRN(r_p_RN)))

  EQ_ = lambda t, X: EQ_func(t, X, fRN, dfRN, r_H_RN, 20)
  ResRN = imcomp(EQ_, Y0, t_eval, dev=device)

  Y = ResRN.ys
  RN_r = Y[:, :, 1]
  RN_x = Y[:, :, 1]*torch.sin(Y[:, :, 2])*torch.cos(Y[:, :, 3])
  RN_y = Y[:, :, 1]*torch.sin(Y[:, :, 2])*torch.sin(Y[:, :, 3])
  RN_z = Y[:, :, 1]*torch.cos(Y[:, :, 2])

  i2_RN = lambda r: i2_r(r, fRN, r_p_RN)
  RN_I2 = thin_disk(RN_r, RN_x, RN_y, RN_z, disc_normv[:3], i2_RN, r_H_RN)

  # Считаем траектории модифицированной модели и получаем картинку
  B_EH = float(1/225/np.pi/np.sqrt(137)*C**2*h)
  # print(B_EH)

  fEH = lambda r: 1 - 2/r + C/(r**2) - B_EH/(r**6)
  dfEH = lambda r: 2/(r**2) - 2*C/(r**3) + 7*B_EH/(r**7)

  eq_r_p = lambda r: r*dfEH(r)-2*fEH(r)
  r_H_EH = float(fsolve(fEH, 2))
  r_p_EH = float(fsolve(eq_r_p, 3))
  b_cr_EH = float(r_p_EH/np.sqrt(fEH(r_p_EH)))

  EQ_ = lambda t, X: EQ_func(t, X, fEH, dfEH, r_H_EH, 20)
  ResEH = imcomp(EQ_, Y0, t_eval, dev=device)

  Y = ResEH.ys
  EH_x = Y[:, :, 1]*torch.sin(Y[:, :, 2])*torch.cos(Y[:, :, 3])
  EH_y = Y[:, :, 1]*torch.sin(Y[:, :, 2])*torch.sin(Y[:, :, 3])
  EH_z = Y[:, :, 1]*torch.cos(Y[:, :, 2])

  i2_EH = lambda r: i2_r(r, fEH, r_p_EH)
  EH_I2 = thin_disk(Y[:, :, 1], EH_x, EH_y, EH_z, disc_normv[:3], i2_EH, r_H_EH)

  outp = {'I2_EH': EH_I2, 'I2_RN': RN_I2, 'b.c.': INITXYZ, \
          'ResEH': ResEH, 'ResRN': ResRN, 'res': r.shape, 'C':C, 'h': h}

  return outp


def trace(rng):
  prefix = 'Sim_{}'.format(rng) + '.pickle'
  print(prefix + 'in process')
  INITXYZ = sample('circle', db=[8, 10, 0, 20], rng = rng, D0 = 15, dth=0)
  sim_data = Sim(INITXYZ, device='cpu')
  

  with open(prefix, 'wb') as handle:
    pickle.dump(sim_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return sim_data
