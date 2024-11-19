import torch

def i2_r(r, f, r_p=3):
    I = torch.heaviside(r, r_p*torch.ones_like(r))*f(r)**(2)*torch.pow(r-r_p+1, -3)
    return I


def thin_disk(R, X, Y, Z, norm_v, I_r=i2_r, r_H=2.0):

    N_tr, t_ = R.shape[1], R.shape[0]

    F = torch.zeros(N_tr) # результирующие интенсивности

    # список точек, в которых излучены фотоны, нужен для отладки
    emit = [[] for n in range(N_tr)]

    # Вычислим проекцию на нормаль к диску
    proj = (X*norm_v[0]+Y*norm_v[1]+Z*norm_v[2])

    for n in range(N_tr):
        for t in range(t_-1):
            if (R[t, n]>r_H):
                if (proj[t, n] > 0):
                    if (proj[t+1, n] < 0):
                        #emit[n].append(t)
                        F[n] += I_r(R[t, n])
                elif (proj[t+1, n] > 0):
                    #emit[n].append(t)
                    F[n] += I_r(R[t, n])

    return F

# Функция интенсивности излучения


