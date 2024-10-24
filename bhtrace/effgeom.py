from .spacetime import Spacetime


def EffGeomSPH(Spacetime):

    def __init__(self, SPc: Spacetime):

        self.SPc = SPc

        pass

    def g(self, X):

        g = self.SPc.g(X)

        pass

    def ginv(self, X):

        ginv = self.SPc.ginv(X)

        pass

