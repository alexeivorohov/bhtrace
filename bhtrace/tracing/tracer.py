import torch
import os
import pickle
import time 

from abc import abstractmethod
from ..geometry import Spacetime, Particle
from ..functional import ODE, print_status_bar


class Tracer():

    def __init__(self, ode_method='Euler'):

        self.odeint = ODE(ode_method)

        pass

    @abstractmethod
    def __term__(self, t, XP):

        return XP


    def forward(self,
        particle: Particle,
        X0,
        P0,
        T,
        nsteps = 128,
        r_max = 30.0,
        max_proper_t=500.0,
        dev = 'cpu',
        eps = 1e-3,
        ):
        '''
        Trace particle trajectories

        ### Inputs:
        - X0: torch.Tensor [n, 4] - initial positions
        - P0: torch.Tensor [n, 4] - initial impulses
        - T: float - simulation time on observer's clock
        Optional:
        - eps: float - parameter for numerical differentiation
        - nsteps: int - number of steps per trajectory
        - r_max: float - distance from center, on which computation should be stopped
        - max_proper_t: float - particle proper time, on which computation should be stopped
        - dev: str - device to perfom computation (see PyTorch documentation)
        '''

        self.particle = particle
        self.spc = particle.spacetime

        self.Nt = nsteps
        self.Ni = X0.shape[0]
        self.eps = eps

        self.r_max = torch.Tensor([r_max])
        self.max_proper_t = torch.Tensor([max_proper_t])

        self.X = torch.zeros(nsteps, self.Ni, 4)
        self.P = torch.zeros(nsteps, self.Ni, 4)
        self.X[0, :, :] = X0
        self.P[0, :, :] = P0

        T0 = torch.Tensor([0.0])

        start_time = time.time()
        for n in range(self.Ni):

            XP0 = torch.cat((X0[n], P0[n]))

            sol = self.odeint.forward(
                term=self.__term__, 
                X0=XP0, 
                T = (0.0, T),
                nsteps=nsteps,
                event_fn=self.evnt,
                reg = self.reg
                )

            self.X[:, n, :] = sol['X'][:, :4]
            self.P[:, n, :] = sol['X'][:, 4:]
            elapsed_time = time.time() - start_time
            print_status_bar(n, self.Ni, elapsed_time)

        print('\n Done!')
        return self.X, self.P


    def save(self, filename, directory=None, comment=None):
        '''
        Save last result to a file specified by filename and optional directory.

        Parameters:
        - filename: The name of the file or full path.
        - directory: Optional; directory path to save the file in.

        Returns:
        - str: The full path of the saved file.
        '''

        if directory != None:
            full_path = os.path.join(directory, filename)
        else:
            full_path = filename

        result = {
            'particle': None,
            'trcr': self.name, 
            # 'ode': self.ode.name,
            # 't_param':self.param,
            # 'ode_param': self.ode_param,
            'Ni': self.Ni,
            # 'Nt': self.Nt,
            # 't': self.t,
            'X': self.X,
            'P': self.P}
        
        if comment != None:
            result['comment'] = comment


        with open(full_path, 'wb') as file:
            pickle.dump(result, file)

        return full_path

