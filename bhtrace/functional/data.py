# Several data-operating routines are planned, including support of hdf5 format, but for now
# reading and storing results is possible with pickle:

# Trajectories (ray-tracing output) stored in dictionaries in next format:

# 'particle' : Particle() - a copy of particle class, which was traced. Remeber, that this class also 
# contains Spacetime()
# 'X0': torch.Tensor of shape [Ni, 4] - initial coords
# 'P0' : torch.Tensor of shape [Ni, 4] - initial impulses
# 'solv': string - type of solver used
# 'm_param': list - parameters of the solver
# 'X' : torch.Tensor of shape [Nt, Ni, 4] - coords
# 'P' : torch.Tensir of shape [Nt, Ni, 4] - impulses
# 'Ni' : number of initial conditions (number of particles)
# 'Nt' : number of steps
# 't' : float - simulation 'time' (time of a remote observer)


# Images

# class TrajData():

#     def __init__(self, X):

#         pass


#     def load(self, X)

#         pass


# class ImageData