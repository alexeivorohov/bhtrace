import torch

def thin_disk(R, X, Y, Z, norm_v, I_r, r_H):

  N_tr, t_ = R.shape[0], R.shape[1]

  F = torch.zeros(N_tr) # результирующие интенсивности

  # список точек, в которых излучены фотоны, нужен для отладки
  emit = [[] for n in range(N_tr)]

  # Вычислим проекцию на нормаль к диску
  proj = (X*norm_v[0]+Y*norm_v[1]+Z*norm_v[2])

  for n in range(N_tr):
      for t in range(t_-1):
          if (R[n, t]>r_H):
              if (proj[n, t] > 0):
                  if (proj[n, t+1] < 0):
                      #emit[n].append(t)
                      F[n] += I_r(R[n, t])
              elif (proj[n, t+1] > 0):
                  #emit[n].append(t)
                  F[n] += I_r(R[n, t])

  return F

# Функция интенсивности излучения

def i2_r(r, f, r_p=3):
    I = torch.heaviside(r, r_p*torch.ones_like(r))*f(r)**(2)*torch.pow(r-r_p+1, -3)
    return I
