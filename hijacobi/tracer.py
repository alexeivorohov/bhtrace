import torch
import torchode as tode

# Функция для вычисления символов Кристоффеля, возвращает torch.tensor размерности [:, 4, 4, 4]
def GMA_spq(X, f, df):
    # X: [t, r, th, phi]
    r = X[:, 1]
    th = X[:, 2]
    phi = X[:, 3]

    outp = torch.zeros([X.shape[0], 4, 4, 4])

    f_ = f(r)
    df_ = df(r)

    # t
    outp[:, 0, 1, 0] = df_/2/f_
    outp[:, 0, 0, 1] = outp[:, 0, 1, 0]

    # r
    outp[:, 1, 0, 0] = f_*df_/2
    outp[:, 1, 1, 1] = -outp[:, 0, 1, 0]
    outp[:, 1, 2, 2] = -f_*r
    outp[:, 1, 3, 3] = outp[:, 1, 2, 2]*torch.sin(th)**2

    # th
    outp[:, 2, 1, 2] = 1/r
    outp[:, 2, 2, 1] = outp[:, 2, 1, 2]
    outp[:, 2, 3, 3] = -0.5*torch.sin(2*th)

    #phi
    outp[:, 3, 3, 1] = 1/r
    outp[:, 3, 1, 3] = outp[:, 3, 3, 1]
    outp[:, 3, 3, 2] = 1/torch.tan(th)
    outp[:, 3, 2, 3] = outp[:, 3, 3, 2]

    return outp

# Функция, задающая систему ДУ (уравнение геодезической)
def EQ_func(t, X, f, df, r_H, r_lim):
    X_ = X[:, 0:4]
    Y_ = X[:, 4:]

    G = GMA_spq(X_, f, df)

    res = torch.zeros_like(X)

    for i in range(X.shape[0]):
        if (X[i, 1]>r_H) & (X[i, 1] < r_lim):
            dY = - G[i] @ Y_[i] @ Y_[i]
            res[i] = torch.cat([Y_[i], dY])

    return res

# Функция-обёртка для процесса интегрирования
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