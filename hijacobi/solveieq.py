# This file describes base class SolveIEQ, which holds the process
# of solving integral equations, and its implementations.

from hijacobi.model import *

from abc import ABC, abstractmethod
from typing_extensions import ParamSpecArgs

import matplotlib.pyplot as plt
from matplotlib import colors
import pickle

import torch
from torchquad import Simpson, MonteCarlo, set_up_backend
from xitorch.optimize import rootfinder

set_up_backend("torch", data_type="float32")


class SolveIEQ(ABC):

    # Инициализация
    def __init__(self, BH: model, N=31):
        self.BH = BH
        self.N = 31
        self.Ir = self.integrate_Ur
        self.Ith = self.integrate_Uth


    # ИНТЕГРИРОВАНИЕ
    def set_Simpson(self):
        self.int = Simpson()


    def set_MonteCarlo(self):
        self.int = MonteCarlo()


    def setCustomIth(self, func: callable):
        '''
        method to set alternative expression as integral U_th

        Input:
        func - callable(th_lims: Tensor, l: Tensor, q: Tensor)
          
        '''

        thlims = torch.Tensor([1, 1])
        q_t = torch.ones([2, 3])*2
        l_t = torch.ones([2, 3])

        tst = func(thlims, l_t, q_t)

        if torch.allclose(tst, torch.zeros_like(tst)):
            print('Test passed, setting fuction as Ith')
            self.Ith = func

        pass


    # Преобразование координат в уравнении, по умолчанию - тождественное:
    def transform_eq(self, eq: callable):
        return eq
    
    def outp_transform(self, l, q):
        return l, q


    # Интерфейсы для интегралов потенциалов
    def Ir(self, domain_r, l_s, q_s):
        pass


    def Ith(self, domain_th, l_s, q_s):
        pass


    # Сами интегралы:
    # допускается введение любого хранящего числа итерируемого объекта в качестве пределов.
    # на выходе будет len(lims)-1 интегралов по каждому из интервалов
    # этот функционал полезен, когда нужно увеличить точность
    def integrate_Ur(self, lims, l_s, q_s):

        dI = lambda r: torch.pow(self.BH.uR_(r, l_s, q_s), -1/2)

        domains = [torch.Tensor([[lims[i], lims[i+1]]]) for i in range(len(lims)-1)]

        outp = [self.int.integrate(dI, dim=1, N=self.N, integration_domain=domain) for domain in domains]

        return torch.sum(torch.stack(outp), dim=0)
    

    def integrate_Uth(self, lims, l_s, q_s):

        dI = lambda th: torch.pow(self.BH.uTh(th, l_s, q_s), -1/2)

        domains = [torch.Tensor([[lims[i], lims[i+1]]]) for i in range(len(lims)-1)]

        outp = [self.int.integrate(dI, dim=1, N=self.N, integration_domain=domain) for domain in domains]

        return torch.sum(torch.stack(outp), dim=0)

    ### РЕШЕНИЕ
    @abstractmethod
    def prepare(eq: callable):
        pass
    
    
    @abstractmethod
    def solve(eq: callable):
        ## Реализации этой функции должны содержать 
        ## только обязательные при решении операции, 
        ## всё остальное выносится в prepare
        pass
        

    ### ОТЛАДКА
    # Отладочная функция - вывод значений интегралов на сетке
    def debugImLQ(self, Lgrid, Qgrid, rlims, thlims):

        Ir_val = self.Ir(rlims, Lgrid, Qgrid)
        Ith_val = self.Ith(thlims, Lgrid, Qgrid)

        dlta = Ir_val + Ith_val
        print(dlta.shape)
        fig, ax = plt.subplots(1, 3, figsize=(18, 8))

        ax[0].imshow(Ir_val)
        ax[0].set_title('Ir')

        ax[1].imshow(Ith_val)
        ax[1].set_title('I_th')

        ax[2].imshow(dlta)
        ax[2].set_title('$\Delta = I_{\theta} - I_r$')

        plt.show()

        return Ir_val, Ith_val


    # Отладочная функция - вывод значений любой функции (l,q) на сетке параметров
    def debugImLQ_f(self, fLQ, Lgrid, Qgrid):
        
        val = fLQ(Lgrid, Qgrid)

        fig, ax = plt.subplots(1, 1, figsize=(8,8))
        ax.imshow(val)
        plt.show()

        pass


    # Отладочная функция - вывод значений любой функции (l,q) вдоль некоторой кривой
    def debugTrLQ_f(self, fLQ, param_s, L_s, Q_s):
        '''
        ## Input:
        fLQ - callable l, q: function to be investigated
        param_s - torch.Tensor: affine parameter of curve (L(s), Q(s))
        L_s - torch.Tensor: L impulse values along the curve
        Q_s - torch.Tensor: Q impulse values along the curve
        '''
        val = fLQ(L_s, Q_s)

        fig, ax = plt.subplots(1, 1, figsize=(8,8))
        ax.plot(param_s, val)
        plt.show()

        pass

  # Функция для отладки числа разбиения
    def debug_integrator_f(self, dI, N_s, r_s):

        for n, i in enumerate(N_s):
            pass

        return 0
    
    def get_rootfinder(self):

        return rootfinder
    


class reparam(SolveIEQ):

    def __init__(self, BH):
        super().__init__(BH)
        self.set_Simpson()
        self.q0 = 0
        self.l0 = 0


    def L(self, gma, p):

        l_out = self.l0 + p*torch.sin(gma)

        return l_out


    def Q(self, gma, p):

        q_out = self.q0 + p*torch.cos(gma)

        return q_out
    
    def transform_eq(self, eq: callable):
        
        return lambda p, gma: eq(self.L(gma, p), self.Q(gma, p))
    

    def setup(self, q0: float, tol: float, maxiter: int, integr_N:int):

        self.N = integr_N
        self.q0 = q0
        self.tol = tol
        self.maxiter = maxiter

        self.N=31
        self.Ir = self.integrate_Ur
        self.Ith = self.integrate_Uth


    def prepare(self, eq: callable):
    
        # добавить процедуру определения q0
        # пока не важно

        pass
    

    def gmap_net_politics(self, gma, p, tol_pass_mask):

        left_ind = []
        right_ind = []

        for i in range(len(p)-1):
            if (tol_pass_mask[i]):
                if (not tol_pass_mask[i+1]):
                    # Sol & Sol
                    right_ind.append(i)
            elif (tol_pass_mask[i+1]):
                #not sol and sol
                left_ind.append(i+1)
            
        gL = torch.Tensor([(gma[i]+gma[i-1])/2 for i in left_ind])
        gR = torch.Tensor([(gma[i]+gma[i+1])/2 for i in right_ind])

        pL = p[left_ind]
        pR = p[right_ind]

        gma_add = torch.cat((gL, gR))
        p0_add = torch.cat((pL, pR))

        return gma_add, p0_add
    

    def solve_p(self, eq_p, gma0: torch.Tensor, p0: torch.Tensor, recN: int):
        '''
        eq_p: callable(p, gma)
        '''
        p0, pmax = def_fspace(eq_p, p0, p0*1.3, gma0, alpha=0.7, maxiter=7)
        
        p0 = bisection(eq_p, p0, pmax, gma0, tol=1e-5, maxiter=self.maxiter)

        err = abs(eq_p(p0, gma0))
        

        if (recN > 1):

            err_mask = torch.less(err, self.tol)

            gma_add, p_add = self.gmap_net_politics(gma0, p0, err_mask)
        
            outp_gma, sort_ind = torch.sort(torch.cat((gma0, gma_add)))
            outp_p = torch.cat((p0, p_add))[sort_ind]
            #outp_err = torch.cat((err, err_add))[sort_ind]

            outp_gma, outp_p, outp_err = self.solve(eq_p, outp_gma, outp_p, recN-1)

            return outp_gma, outp_p, outp_err

        else:
            return gma0, p0, err
        


class repOfan(SolveIEQ):
    '''
    The only working method, that uses fan-like geometry in phase space to obtain solutions
    '''
    def __init__(self, BH):
        super().__init__(BH)
        self.set_Simpson()
    

    def L(self, p, gma):

        l_out = p*torch.sin(gma)

        return l_out
    
    def Q(self, p, gma):

        q_out = p*torch.cos(gma)

        return q_out
    

    def transform_eq(self, eq: callable):
        
        return lambda p, gma: eq(self.L(p, gma), self.Q(p, gma))
    

    def gma_space(self, gma_min, gma_max):
        xspan = torch.linspace(-1+self.gma_eps, 1-self.gma_eps, self.gma_N)
        ticks = xspan/(1+torch.pow(abs(xspan), self.distr_poly_N))
        
        return (gma_min+gma_max)/2 + (gma_max - gma_min)*ticks
    

    def setup(self, method='no_pdef', rf_tol=1e-5, rf_maxiter=100, int_iterN=31, \
               gma_eps=1e-4, gma_N=21, distr_poly_N=1.3, pdef_maxiter=10):

        self.method = method
        
        if self.method == 'no_pdef':
            self.sol = self.sol_wrapper0
        elif self.method == 'use_pdef':
            self.sol = self.sol_wrapper1

        self.int_iterN = int_iterN
        self.rf_tol = rf_tol
        self.rf_maxiter = rf_maxiter
        self.gma_eps = gma_eps
        self.gma_N = gma_N
        self.distr_poly_N = distr_poly_N

        self.pdef_maxiter = pdef_maxiter
    

    def est_plims(self, task_inps):
        p_minV = 1e-4
        p_maxV = 6 
        return p_minV, p_maxV
    

    def prepare(self, task_inps):


        if self.method == 'no_pdef':
            self.p_minV, self.p_maxV = self.est_plims(task_inps)
        
        pass


    def get_gma_lims(self, task_inps):
        '''
        task_inps: dictionary of boundary conditions in format {'r_s':value, 'r_obs':value and etc}
        '''

        gma_max = task_inps['th_obs']
        gma_min = -gma_max

        return gma_min, gma_max
    

    def sol_wrapper0(self, eq, task_inps):

        gma_min, gma_max = self.get_gma_lims(task_inps)

        eq_p = self.transform_eq(eq)

        gma0 = self.gma_space(gma_min, gma_max)
        p_min = torch.ones(self.gma_N)*self.p_minV
        p_max = torch.ones(self.gma_N)*self.p_maxV

        p_res = bisection(eq_p, p_min, p_max, gma0, self.rf_tol, self.rf_maxiter)

        return p_res, gma0


    def sol_wrapper1(self, eq, task_inps):

        gma_min, gma_max = self.get_gma_lims(task_inps)

        eq_p = self.transform_eq(eq)

        gma0 = self.gma_space(gma_min, gma_max, self.rN, self.s)

        p_min = torch.ones(self.gma_N)*0
        p_max = torch.ones(self.gma_N)*10

        p_min, p_max = def_fspace(eq_p, p_min, p_max, gma0, maxiter=self.pdef_maxiter)

        p_res = bisection(eq_p, p_min, p_max, gma0, self.rf_tol, self.rf_maxiter)

        return p_res, gma0

    
    def solve(self, eq: callable, task_inps):

        self.prepare(task_inps)

        p, gma = self.sol(eq, task_inps)

        return self.outp_transform(p, gma) 





