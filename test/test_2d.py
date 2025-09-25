import os
import shutil
import unittest

import torch
import matplotlib.pyplot as plt
import numpy as np
import uniplot as uplt

from bhtrace.tracing import PTracer
from bhtrace.geometry import Photon, SphericallySymmetric, KerrSchild, Observer
from bhtrace.utils.transform import sph2cart

def console_plot(
        case,
        xs,
        ys,
        bs,
        dphis,
        H,
        LTE,
        r_s = 2.0
    ):
    '''
    Plots an overview of test case in console, using uniplot

    :param xs: - input coordinates(required to be in cartesian coordinates)
    :param ys: - 
    :param b: - 
    :param dphi: - lensing angle
    :param H: - evaluated hamiltonian
    :param LTE: - truncation error
    '''

    uplt.plot(xs=xs, ys=ys, character_set='braille', title=f'Trajectories for case {case}', color=True)


def mpl_plot(
        plot_dir,
        case,
        xs=None,
        ys=None,
        b=None,
        dphi=None,
        H=None,
        LTE=None,
        r_s = 2.0
    ):
    '''
    Plots and saves an overview of test case, using matplotlib

    :param X: - input coordinates(required to be in cartesian coordinates)
    :param dphi: - lensing angle
    :param H: - evaluated hamiltonian
    :param LTE: - truncation error
    '''
    plots = []
    if (xs is not None) and (ys is not None):
        plots.append('X-Y')
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.plot(xs, ys)
        ax.grid(True)
    if (b is not None) and (dphi is not None):
        plots.append('Lensing')
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.plot(b, dphi, '-')
        ax.set_xlabel('Impact Parameter')
        ax.set_ylabel('Deflection Angle')
        ax.set_title('Lensing Function')
        ax.grid(True)
        save_path = os.path.join(plot_dir, f'{case}_lensing.png')
        plt.savefig(save_path)
        plt.close(fig)
        print(f"\nLensing plot saved to {save_path}")
    if H is not None:
        plots.append('Hamiltonian preservation')
    if LTE is not None:
        plots.append('Truncation Error')
    
    if len(plots) == 0:
        print('Nothing to plot!')
       
    pass

class Test2D(unittest.TestCase):

    def setUp(self):

        self.dev = 'cpu'
        self.tracers = {
            'ptracer': PTracer()
        }

        self.plot_dir = os.path.join(os.path.dirname(__file__), 'plots')
        os.makedirs(self.plot_dir, exist_ok=True)

    def test_schwarzschild(self):

        spacetime = SphericallySymmetric()
        particle = Photon(spacetime)
        tracer = PTracer(ode_method='RK4')

        # Observer setup
        observer = Observer(
            spacetime=spacetime,
            position=torch.tensor([0., 20., 0., 0.]),
            camera_dir=torch.tensor([-1., 0., 0.])
        )
        net_rng = (20, 1) # A line of rays
        net_size = (20, 0)
        observer.setup_ic(
            particle=particle,
            net_shape='square',
            net_rng=net_rng,
            net_size=net_size
        )

        X0 = observer.X_net
        P0 = observer.P_net
        
        # Tracing:
        trajectory = tracer.forward(particle, X0, P0, 40.0, 256)
        X = trajectory.X
        P = trajectory.P


        # Extracting key data and transforming to cartesian for plotting:

        # Key checks:

        # Performing plots:
        xs = [
            X[:, i, 1].detach().cpu().numpy() for i in range(X.shape[1])
        ]
        ys = [
            X[:, i, 2].detach().cpu().numpy() for i in range(X.shape[1])
        ]
        console_plot(f'',
                     xs=xs,
                     ys=ys,
                     bs=None,
                     dphis=None,
                     H=None,
                     LTE=None)
        

    # def test_lte_plot(self):
    #     spacetime = SphericallySymmetric()
    #     particle = Photon(spacetime)
    #     # To test LTE, we need to use a solver that supports it, and enable it.
    #     # This is a placeholder test, as the current Tracer does not expose LTE tracking.
    #     # I will assume for now that I can enable it via a parameter.
    #     tracer = PTracer(ode_method='RK4')
    #     X0 = torch.tensor([[0.0, 3.0, torch.pi / 2.0, 0.0]])
    #     P0 = torch.tensor([[-1.0, 0.0, 0.0, 3.0]])
        
    #     # This is a hypothetical way to enable LTE tracking
    #     # solution = tracer.forward(particle, X0, P0, 50.0, 200, track_lte=True)
    #     # lte_values = solution['LTE'] 
    #     # t = solution['t']
    #     # console_plot(t, lte_values, title="LTE", plot_type='lte')
    #     pass # Placeholder

if __name__ == '__main__':
    unittest.main()
