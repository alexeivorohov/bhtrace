import torch

def blackbody():

    return 0


def i2_r(r, f, r_p = 3.0):

    I = torch.heaviside(r, r_p*torch.ones_like(r))*f(r)**(2)*torch.pow(r-r_p+1, -3)

    return I