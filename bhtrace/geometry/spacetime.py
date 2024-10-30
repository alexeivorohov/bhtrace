from abc import ABC, abstractmethod
# from typing_extensions import ParamSpecArgs

import torch

class Spacetime(torch.nn.Module):
    '''
    Base class for handling different spacetimes.

    Required methods:
    - g(X) - evaluating metric function
    
    Optional methods:
    - conn(X) - evaluating connections

    may be moved to particle class:
    ? uR(X), uTh(X), iTh(X) and etc
    '''

    # Метрика
    @abstractmethod
    def g(self, X):
        '''
        Metric tensor

        ## Input
        X: torch.Tensor [:, 4] - coordinates

        ## Output
        g: torch.Tensor [:, 4, 4] - metric tensor at each coordinate
        '''
        pass


    @abstractmethod
    def ginv(self, X):
        '''
        Metric inverse

        ## Input
        X: torch.Tensor [:, 4] - coordinates

        ## Output
        g: torch.Tensor [:, 4, 4] - metric inverze at each coordinate
        '''
        pass


    # Связность:
    @abstractmethod
    def conn(self, X):
        '''
        Exact calculation of connection symbols, if possible.
        Output in sta
        X: torch.Tensor() - coordinates
        '''

        return None


    def conn_(self, X):
        '''
        Evaluation of connection symbols by numerical differentiation
        X: torch.Tensor() - coordinates
        '''

        return None


    # Потенциалы
    @abstractmethod
    def uR(self, r: torch.Tensor, l_s: torch.Tensor, q_s: torch.Tensor):
        '''
        Radial potential, as appears in the equations

        ## Input:
        r - tensor: evaluation points;

        l_s, q_s  - tensors of same shape: particle impulses

        ## Output:
        uR - tensor, r.shape x l_s.shape
        '''
        pass


    def uR_(self, r, l, q):

        outp = [self.uR(r_, l, q) for r_ in r]

        return torch.stack(outp)

    @abstractmethod
    def uTh(self, th: torch.Tensor | float, l_s: torch.Tensor | float, q_s: torch.Tensor | float):
        '''
        Polar potential, as figuring in equations

        ## Input:
        th - tensor: evaluation points;

        l_s, q_s  - tensors of same shape: particle impulses

        ## Output:
        uTh - tensor, r.shape x l_s.shape
        '''
        pass


    def uTh_(self, th, l, q):

        outp = [self.uTh(th_, l, q) for th_ in th]

        return torch.stack(outp)

    # Интегралы
    def useCustomIth(self, X: bool):
        '''
        Directive to use custom expression for polar potential integrals
        ... solver integration routine
        '''
        self.CustomIth = X


    def useCustomIr(self, X: bool):
        '''
        Directive to use custom expression for radial potential integrals
        ... solver integration routine
        '''
        self.CustomIr = X

    @abstractmethod
    def Ith(self, th: torch.Tensor | list, l_s: torch.Tensor, q_s: torch.Tensor):
        '''
        Integral of polar potential, as seen in equation on motion constants

        '''
        pass

    @abstractmethod
    def Ith_t(self, th, l_s: float, q_s: float):
        '''
        Integral of polar potential, as seen in equation on motion constants,
        second lim is th_turn
        '''
        pass

    @abstractmethod
    def Ir(self, r, l_s, q_s):
        '''
        Integral of radial potential, as seen in equation on motion constants

        '''
        pass

    @abstractmethod
    def turning_Th(self, l_s, q_s):
        '''
        Turning points along th axis
        '''
        return torch.nan

    @abstractmethod
    def turning_R(self, l_s, q_s):
        '''
        Turning points along r axis
        '''
        return torch.nan


class SphericallySymmetric(Spacetime):

    def __init__(self, f, f_r):
        '''
        Class for handling spherically-symmetric spacetimes 

        ## Input:
        f - callable: metric function
        f_r - it's derivative wrt r
        '''
        self.f = f
        self.df = f_r
        self.useCustomIr(False)
        self.useCustomIth(False)

    def g(self, X):
        
        outp = torch.zeros([X.shape[0], 4, 4])

        f_ = self.f(X[:, 1])
        R2 = X[:, 1]**2

        outp[:, 0, 0] = -f_
        outp[:, 1, 1] = 1/f_
        outp[:, 2, 2] = R2
        outp[:, 3, 3] = R2 * torch.sin(X[:, 2])**2

        return outp

    def ginv(self, X):

        outp = torch.zeros([X.shape[0], 4, 4])

        f_ = self.f(X[:, 1])

        outp[:, 0, 0] = -1/f_
        outp[:, 1, 1] = f_
        outp[:, 2, 2] = torch.pow(X[:, 1], -2)
        outp[:, 3, 3] = torch.pow(X[:, 1]*torch.sin(X[:, 2]), -2)

        return outp

    def conn(self, X):
        
        # X: [t, r, th, phi]
        r = X[:, 1]
        th = X[:, 2]
        phi = X[:, 3]

        outp = torch.zeros([X.shape[0], 4, 4, 4])

        f_ = self.f(r)
        df_ = self.df(r)

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

  # Радиальный потенциал
    def uR(self, r, l_s, q_s):

        # outp = [r_**2*(r_**2 - self.f(r_)*(l_s**2 + q_s**2)) for r_ in r]
    
        return r**2*(r**2 - self.f(r)*(l_s**2 + q_s**2))
    
    
  # Полярный потенциал
    def uTh(self, th, l_s, q_s):
        #xi = l_s/q_s
        outp = [q_s-torch.pow(l_s/torch.tan(th_), 2) for th_ in th]

        return torch.stack(outp)
  

    def turning_Th(self, l_s, q_s):
        '''
        Returns turning point for given (l, q)
        '''
        
        outp = torch.abs(torch.arctan(l_s/q_s))
    
        return outp    

  # Известное точное выражение для интеграла полярного потенциала
    def Ith(self, th, l_s, q_s):
        '''
        Precise exspression for 1/\sqrt(Uth) integral over interval 
        without turning points, if exists
        '''

        xi2 = (l_s/q_s)**2

        invmodulusLQ = torch.pow(l_s**2 + q_s**2, -0.5)
 
  
        if abs(th[0]) == torch.pi/2:
            outp0 = torch.zeros_like(q_s)
        else:
            tanth = torch.tan(th[0])
            arg = torch.sqrt((1+xi2)/(tanth**2-xi2))*torch.sign(tanth)
            outp0 = -invmodulusLQ*torch.atan(arg)

        if abs(th[1]) == torch.pi/2:
            outp1 = torch.zeros_like(q_s)
        else:
            tanth = torch.tan(th[1])
            arg = torch.sqrt((1+xi2)/(tanth**2-xi2))*torch.sign(tanth)
            outp1 = -invmodulusLQ*torch.atan(arg)
        
        return (outp1 - outp0)
     
    
    def Ith_t(self, th, l_s: float, q_s: float):
        '''
        Computes I_th(th, theta_turn)
        '''
        invmodulusLQ = torch.pow(l_s**2 + q_s**2, -0.5)

        outp1 = -invmodulusLQ*torch.pi/2

        if abs(th) == torch.pi/2:
            outp0 = torch.zeros_like(q_s)
        else:
            xi2 = (l_s/q_s)**2
            tanth = torch.tan(th)
            arg = torch.sqrt((1+xi2)/(tanth**2-xi2))*torch.sign(tanth)
            outp0 = -invmodulusLQ*torch.atan(arg)

        return (outp1 - outp0)


class Kerr(Spacetime):

    def __init__(self, a: float):
        '''
        a: float - rotation parameter in units a/M
        '''
        self.a = a
        self.Dlta = lambda r: r**2 - 2*r + a**2
        self.Sgma = lambda r, th: r**2 + a**2 * torch.cos(th)**2
        self.P = lambda r, l: r**2+a**2-a*l
    
    def uR(self, r, l_s, q_s):

        outp = self.P(r, l_s) - self.Dlta(r)*((l_s-a)**2 + q_s**2)

        return outp

    def uTh(self, th, l_s, q_s):

        outp = q_s**2 - torch.cos(th)**2*(-a**2+ (l_s*torch.sin(th))**2)

        return outp
