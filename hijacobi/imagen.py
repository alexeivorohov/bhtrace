import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle

import matplotlib.colors as colors
from scipy.optimize import fsolve
from IPython.display import clear_output

from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

import sys
sys.path.append('..')

# import hijacobi
from hijacobi import cart2sph, sph2cart, sample, EQ_func, imcomp, i2_r, thin_disk

## Модельные функции

# Метод для расчёта наблюдаемой интенсивности в модели аккреции тонкого диска

# Изучение величины эффекта в зависимости от разрешения сетки

def Sim(INITXYZ, th_obs=0.5, device='cpu', C = 0.45, h = 2e6):

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



def sim_task(rng):
  prefix = 'Sim_{}'.format(rng) + '.pickle'
  print(prefix + 'in process')
  INITXYZ = sample('circle', db=[8, 10, 0, 20], rng = rng, D0 = 15, dth=0)
  sim_data = Sim(INITXYZ, device='cpu')
  

  with open(prefix, 'wb') as handle:
    pickle.dump(sim_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return sim_data

# Create a list of items to process
rngs = [5, 10, 15, 20, 25, 30, 35, 40]

# Use pool.map to apply parallel_task to each item
# results = pool.map(sim_task, rngs)
results = [sim_task(rng) for rng in rngs]