
from bhtrace.geometry import Spacetime, Photon
from bhtrace.tracing import Tracer
from bhtrace.functional import print_status_bar
from bhtrace.geometry.observer import Observer

from .medium import Medium
import torch

# TODO:
# [ ] Set up a factory method for creating different scenes
# [ ] hash/key generation method for naming data entries

class Scene:


    # def __new__(cls, 
    #             type, *args, **kwargs):
        
    #     if type in _SCENES_.keys():
    #         return _SCENES_[type].__new__(*args, **kwargs)
    #     else:
    #         raise NotImplementedError
 
    
    def __init__(self,
                 spacetime: Spacetime,
                 medium: Medium,
                 photon: Photon,
                 observer: Observer,
                 tracer: Tracer,
                 ):

        self.spacetime = spacetime
        self.medium = medium
        self.photon = photon
        self.observer = observer
        self.tracer = tracer

        self.ray_data = {}
        self.imgs = {}


    def key(self, *args):

        return 'abcdef'
    

    def compute(self, *args):

        return None
    

    def propagate_photons(self, *args, **kwargs):

        X0 = self.observer.X_net
        P0 = self.observer.P_net

        ray_XP = self.tracer.forward(
            self.photon, X0, P0, T=60.0, nsteps=128
        )

        key = self.key(*args)

        self.tracer.save(key, directory='')      


    def compute_radiation(self, *args, **kwargs):
        '''
        gX: callable(X) - local metric

        how to distinguish between background and effective metric?
        Pass as different arguments or use polymorphism?

        '''

        # (photon, medium, X_cart, X_sph, P_sph, X_obs, U_obs)

        X, Y, Z = X_cart[:, :, 1], X_cart[:, :, 2], X_cart[:, :, 3]
        R, Phi, Th = X_sph[:, :, 1], X_sph[:, :, 3], X_sph[:, :, 2]

        N_tr, t_ = R.shape[1], R.shape[0]

        F = torch.zeros(N_tr) # Array to store intensities

        r_isco = torch.Tensor([6.0])    

        ## Searching intersections

        # Calculate projection on disc normal
        norm_v = (-np.cos(th_obs), 0, np.sin(th_obs))
        proj = (X*norm_v[0]+Y*norm_v[1]+Z*norm_v[2])

        emit = [[] for n in range(N_tr)]

        for n in range(N_tr):

            print_status_bar(N_tr, n)
            
            for t in range(t_ - 1):
                if (R[t, n] > r_isco):
                    if (proj[t, n]*proj[t+1, n] > 0):
                        emit[n].append(t)

        
            pass

        ## Calculate radiation
        for n in range(N_tr):
            
            F[n] = 0.0

            if len(emit[n]) > 0:

                E_o = photon_energy(photon, P_sph[0, n, :], U_obs, X_obs)

                for t in emit[n]:

                    Xsph_e = X_sph[t, n, :]
                    
                    f_ = medium.Flux(X_sph[t, n, :])
                    U_e = medium.U(X_sph[t, n, :])
                    
                    E_e = photon_energy(photon, P_sph[t, n, :], U_e, X_sph[t, n, :])
                    z = E_e/E_o - 1

                    F[n] += torch.pow(z, 4)*f_

            return F   


    def save(self, *args):

        pass


class ObjectAndDisk(Scene):

    def __init__(self,
                 spacetime: Spacetime,
                 medium: Medium,
                 photon: Photon,
                 observer: Observer,
                 tracer: Tracer,
                 ):

        pass


    def compute(self):

        pass


_SCENES_ = {'ObjectAndDisk': ObjectAndDisk}