from .spacetime import Spacetime


def EffGeomSPH(Spacetime):

    def __init__(self, Spacetime: Spacetime):

        self.Spacetime = Spacetime

        pass

    def g(self, X):

        g = self.Spacetime.g(X)

        pass

    def ginv(self, X):

        ginv = self.Spacetime.ginv(X)

        pass

