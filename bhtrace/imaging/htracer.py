import torch
import torchode as tode
import os
import pickle

from ..geometry import Particle

class HTracer():

    def __init__(self):

        self.pcle = None
        self.solv = 'HTracer'
        self.m_param = None

        self.Ni = 0
        self.Nt = 0
        self.t = 0

        self.X = None
        self.P = None
        self.X0 = None
        self.P0 = None

        self.DVec = None
        self.eps = None

        pass


    def particle_set(self, particle: Particle):
        '''
        Attach class of particles to be traced

        ## Input:
        particle: Particle        
        '''

        self.pcle = particle
        self.spc = particle.Spacetime
        

    def step_size(self, X, P, gX, dgX):



        return dt


    def evnt_check(self, X, P):



        return mask

    
    def __step__(self,  X: torch.Tensor, P: torch.Tensor, dt):

        dH = self.pcle.dHmlt(X, P)

        dP = - dH*dt
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

        self.DVec = torch.einsum('bi,ij->bij', torch.ones_like(X0), torch.eye(4))
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

    # old method
    def __eq__(self, t, XP):

        X_, P_ = XP[:, 0:4], XP[:, 4:]

        # G = self..conn(X_)

        dP = self.pcle.dHmlt_(X_, P_, self.DVec, self.eps) 

        # dP = torch.einsum('bmuv,bu,bv->bm', G, P_, P_)

        # dP = torch.zeros_like(P_)
        # for i in range(P_.shape[0]):
        #   if True:
        #     dP[i] = - G[i] @ P_[i] @ P_[i]

        return torch.cat([P_, dP], axis=1)

    # old method
    def solve(self, X0, P0, t_sim, n_steps, dev='cpu'):

        N_tr = X0.shape[0]
        XP0 = torch.cat([X0, P0], axis=1)

        self.DVec = torch.einsum('bi,ij->bij', torch.ones_like(X0), torch.eye(4))
        self.eps = 1e-5

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
            'particle': None,
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





