"""
Concrete implementations of spacetimes in spherical-like coordinates.

This module provides concrete implementations of the `Spacetime` abstract base
class for several common spacetimes, including Minkowski, Schwarzschild, Kerr,
and Kerr-Newman, all in spherical or Boyer-Lindquist coordinates.

"""

import torch
import math
from abc import abstractmethod

from bhtrace.geometry.spacetime._base import Spacetime
from bhtrace.utils import bisection


class MinkowskiSph(Spacetime):
    """Minkowski spacetime in spherical coordinates.

    Attributes
    ----------
    _has_analytic_conn : bool
        True, as analytic connection coefficients are provided.
    _diff_ord : int
        Order of numerical differentiation, set to 1.

    """

    _has_analytic_conn = True
    _has_analytic_tetrad = True
    _diff_ord = 1

    def __init__(self):
        super().__init__()
        pass

    def g(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate the metric tensor for Minkowski spacetime in spherical coordinates.

        Parameters
        ----------
        X : torch.Tensor
            Coordinates of shape [..., 4] (t, r, theta, phi).

        Returns
        -------
        torch.Tensor
            Metric tensor of shape [..., 4, 4].
        """
        outp = torch.zeros(*X.shape, 4, dtype=X.dtype, device=X.device)

        outp[..., 0, 0] = -1
        outp[..., 1, 1] = 1
        outp[..., 2, 2] = torch.pow(X[..., 1], 2)
        outp[..., 3, 3] = torch.pow(X[..., 1] * torch.sin(X[..., 2]), 2)

        return outp

    def ginv(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate the inverse metric tensor for Minkowski spacetime.

        Parameters
        ----------
        X : torch.Tensor
            Coordinates of shape [..., 4] (t, r, theta, phi).

        Returns
        -------
        torch.Tensor
            Inverse metric tensor of shape [..., 4, 4].
        """
        outp = torch.zeros(*X.shape, 4, dtype=X.dtype, device=X.device)

        outp[..., 0, 0] = -1
        outp[..., 1, 1] = 1
        outp[..., 2, 2] = torch.pow(X[..., 1], -2)
        outp[..., 3, 3] = torch.pow(X[..., 1] * torch.sin(X[..., 2]), -2)

        return outp

    def conn(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate analytic connection coefficients for Minkowski spacetime.

        Parameters
        ----------
        X : torch.Tensor
            Coordinates of shape [..., 4] (t, r, theta, phi).

        Returns
        -------
        torch.Tensor
            Connection coefficients of shape [..., 4, 4, 4].
        """
        r = X[..., 1]
        th = X[..., 2]

        outp = torch.zeros(*X.shape, 4, 4, dtype=X.dtype, device=X.device)

        f_ = 1
        df_ = 0

        # t
        outp[..., 0, 1, 0] = df_ / 2 / f_
        outp[..., 0, 0, 1] = outp[..., 0, 1, 0]

        # r
        # outp[:, 1, 0, 0] = f_*df_/2
        outp[..., 1, 1, 1] = -outp[..., 0, 1, 0]
        outp[..., 1, 2, 2] = -r
        outp[..., 1, 3, 3] = outp[..., 1, 2, 2] * torch.sin(th) ** 2

        # th
        outp[..., 2, 1, 2] = 1 / r
        outp[..., 2, 2, 1] = outp[..., 2, 1, 2]
        outp[..., 2, 3, 3] = -0.5 * torch.sin(2 * th)

        # phi
        outp[..., 3, 3, 1] = 1 / r
        outp[..., 3, 1, 3] = outp[..., 3, 3, 1]
        outp[..., 3, 3, 2] = 1 / torch.tan(th)
        outp[..., 3, 2, 3] = outp[..., 3, 3, 2]

        return outp

    def crit(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate a criticality function.

        .. note:: The physical meaning of this function is unclear. It returns
                  the absolute value of the radial coordinate.

        Parameters
        ----------
        X : torch.Tensor
            Coordinates of shape [..., 4].

        Returns
        -------
        torch.Tensor
            The absolute value of the radial coordinate.
        """
        return abs(X[..., 1])

    def tetrad(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Calculate the local tetrad for Minkowski spacetime.

        Parameters
        ----------
        x : torch.Tensor
            Coordinates of shape [..., 4].

        Returns
        -------
        torch.Tensor
            The tetrad vectors of shape [..., 4, 4].
        """
        outp = torch.zeros(*x.shape, 4, dtype=x.dtype, device=x.device)

        r = x[..., 1]
        th = x[..., 2]

        outp[..., 0, 0] = 1.0
        outp[..., 1, 1] = 1.0
        outp[..., 2, 2] = r
        outp[..., 3, 3] = r * th.sin()

        return outp


class SphericallySymmetric(Spacetime):
    """
    Generic class for spherically-symmetric spacetimes.

    The metric is of the form:
    ds^2 = -A(r) dt^2 + B(r) dr^2 + r^2 dth^2 + r^2 sin^2(th) dphi^2

    If no functions are provided, it defaults to Schwarzschild spacetime.

    Parameters
    ----------
    A : callable, optional
        Function `A(r)` for the tt-component of the metric.
    A_r : callable, optional
        The r-derivative of `A(r)`.
    B : callable, optional
        Function `B(r)` for the rr-component of the metric.
    B_r : callable, optional
        The r-derivative of `B(r)`.

    Attributes
    ----------
    A, A_r, B, B_r : callable
        The functions defining the metric components.
    r_h : float
        The radius of the event horizon, calculated via bisection if `A` is provided.
    """

    _has_analytic_conn = True
    _has_analytic_tetrad = True
    _coords = "Spherical"
    _diff_ord = 2

    def __init__(self, A=None, A_r=None, B=None, B_r=None):
        if A == None:
            self.A = lambda r: -(1.0 - 2.0 / r)
            self.A_r = lambda r: -2.0 * torch.pow(r, -2)
            self.B = lambda r: 1.0 / (1.0 - 2.0 / r)
            self.B_r = lambda r: 2.0 * torch.pow(r, -2) * torch.pow(1.0 - 2.0 / r, -2)
            self.r_h = 2.0
        elif B == None:
            self.A = lambda r: -A(r)
            self.A_r = lambda r: -A_r(r)
            self.B = lambda r: 1 / A(r)
            self.B_r = lambda r: -A_r(r) / (A(r)) ** 2
        else:
            self.A = A
            self.A_r = A_r
            self.B = B
            self.B_r = B_r

        if A is not None:
            self.r_h = float(bisection(A, 0.0, 4.0))

        super().__init__()
        pass

    def g(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate the metric tensor.

        Parameters
        ----------
        X : torch.Tensor
            Coordinates of shape [..., 4].

        Returns
        -------
        torch.Tensor
            Metric tensor of shape [..., 4, 4].
        """
        outp = torch.zeros(*X.shape, 4, dtype=X.dtype, device=X.device)

        A_ = self.A(X[..., 1])
        B_ = self.B(X[..., 1])

        R2 = torch.pow(X[..., 1], 2)

        outp[..., 0, 0] = A_
        outp[..., 1, 1] = B_
        outp[..., 2, 2] = R2
        outp[..., 3, 3] = R2 * torch.sin(X[..., 2]) ** 2

        return outp

    def ginv(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate the inverse metric tensor.

        Parameters
        ----------
        X : torch.Tensor
            Coordinates of shape [..., 4].

        Returns
        -------
        torch.Tensor
            Inverse metric tensor of shape [..., 4, 4].
        """
        outp = torch.zeros(*X.shape, 4, dtype=X.dtype, device=X.device)

        A_ = self.A(X[..., 1])
        B_ = self.B(X[..., 1])

        outp[..., 0, 0] = 1 / A_
        outp[..., 1, 1] = 1 / B_
        outp[..., 2, 2] = torch.pow(X[..., 1], -2)
        outp[..., 3, 3] = torch.pow(X[..., 1] * torch.sin(X[..., 2]), -2)

        return outp

    def conn(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate analytic connection coefficients.

        Parameters
        ----------
        X : torch.Tensor
            Coordinates of shape [..., 4] (t, r, th, phi).

        Returns
        -------
        torch.Tensor
            Connection coefficients of shape [..., 4, 4, 4].
        """
        # X: [t, r, th, phi]
        r = X[..., 1]
        th = X[..., 2]

        outp = torch.zeros(*X.shape, 4, 4, dtype=X.dtype, device=X.device)

        A_ = self.A(r)
        dA_ = self.A_r(r)

        B_ = self.B(r)
        dB_ = self.B_r(r)

        # t
        outp[..., 0, 1, 0] = dA_ / 2 / A_
        outp[..., 0, 0, 1] = outp[..., 0, 1, 0]

        # r
        outp[..., 1, 0, 0] = dA_ / 2 / B_
        outp[..., 1, 1, 1] = dB_ / 2 / B_
        outp[..., 1, 2, 2] = -r / B_
        outp[..., 1, 3, 3] = outp[..., 1, 2, 2] * torch.sin(th) ** 2

        # th
        outp[..., 2, 1, 2] = 1 / r
        outp[..., 2, 2, 1] = outp[..., 2, 1, 2]
        outp[..., 2, 3, 3] = -0.5 * torch.sin(2 * th)

        # phi
        outp[..., 3, 3, 1] = 1 / r
        outp[..., 3, 1, 3] = outp[..., 3, 3, 1]
        outp[..., 3, 3, 2] = 1 / torch.tan(th)
        outp[..., 3, 2, 3] = outp[..., 3, 3, 2]

        return outp

    def tetrad(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Calculate the local tetrad.

        Parameters
        ----------
        x : torch.Tensor
            Coordinates of shape [..., 4].

        Returns
        -------
        torch.Tensor
            The tetrad vectors of shape [..., 4, 4].
        """
        r = x[..., 1]
        th = x[..., 2]

        outp = torch.zeros(*x.shape, 4, dtype=x.dtype, device=x.device)

        outp[..., 0, 0] = torch.sqrt(-self.A(r))
        outp[..., 1, 1] = torch.sqrt(self.B(r))
        outp[..., 2, 2] = r
        outp[..., 3, 3] = r * th.sin()

        return outp

    def crit(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate a criticality function.

        .. note:: The physical meaning of this function is unclear. It returns
                  the absolute value of the g_tt metric component.

        Parameters
        ----------
        X : torch.Tensor
            Coordinates of shape [..., 4].

        Returns
        -------
        torch.Tensor
            The absolute value of `A(r)`.
        """
        return abs(self.A(X[..., 1]))


class KerrBL(Spacetime):
    """
    Kerr spacetime in Boyer-Lindquist coordinates.

    Parameters
    ----------
    a : float, optional
        Dimensionless spin parameter, by default 0.6.

    Attributes
    ----------
    a : float
        Dimensionless spin parameter.
    a2 : float
        Spin parameter squared.
    r_h : float
        Event horizon radius.

    References
    ----------
    [1] Catalogue of spacetimes, p.51
    """

    _coords = "Spherical"
    _has_analytic_conn = False
    _has_analytic_tetrad = True
    # Coords = "BoyerLindquist"

    def __init__(self, a=0.6):
        self.a = a
        self.a2 = a**2
        self.r_h = 1 + math.sqrt(1 - self.a2)
        self.__labels__ = ["t", "r", "\\theta", "\\phi"]
        super().__init__()

    def g(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Kerr metric tensor in Boyer-Lindquist coordinates.

        Parameters
        ----------
        x : torch.Tensor
            Coordinates of shape [..., 4].

        Returns
        -------
        torch.Tensor
            Metric tensor of shape [..., 4, 4].
        """
        outp = torch.zeros(*x.shape, 4, dtype=x.dtype, device=x.device)

        r = x[..., 1]
        r2 = torch.pow(r, 2)

        costh = torch.cos(x[..., 2])
        sinth = torch.sin(x[..., 2])

        costh2 = torch.pow(costh, 2)
        sinth2 = torch.pow(sinth, 2)

        sgma = r2 + self.a2 * costh2
        z = 2.0 * r / sgma
        dlta = r2 + self.a2 - 2 * r
        # sgma = (r2 + self.a2) ** 2 - self.a2 * dlta * sinth2

        outp[..., 0, 0] = z - 1
        outp[..., 0, 3] = - z * self.a * sinth2
        outp[..., 3, 0] = outp[..., 0, 3]

        outp[..., 1, 1] = sgma / dlta
        outp[..., 2, 2] = sgma
        outp[..., 3, 3] = (r2 + self.a2 + self.a2 * z) * sinth2

        return outp

    def ginv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the inverse Kerr metric tensor.

        Parameters
        ----------
        x : torch.Tensor
            Coordinates of shape [..., 4].

        Returns
        -------
        torch.Tensor
            Inverse metric tensor of shape [..., 4, 4].
        """
        outp = torch.zeros(*x.shape, 4, dtype=x.dtype, device=x.device)

        r = x[..., 1]
        r2 = torch.pow(r, 2)

        costh = torch.cos(x[..., 2])
        sinth = torch.sin(x[..., 2])

        costh2 = torch.pow(costh, 2)
        sinth2 = torch.pow(sinth, 2)

        sgma = r2 + self.a2 * costh2
        dlta = r2 + self.a2 - 2.0 * r
        inv_sgma_dlta = (sgma * dlta).pow(-1)
        
        big_a = (r2 + self.a2).pow(2) - self.a2 * dlta * sinth2

        # g^tt
        outp[..., 0, 0] = - big_a * inv_sgma_dlta
        # g^tphi
        outp[..., 0, 3] = - 2.0 * self.a * r * inv_sgma_dlta
        outp[..., 3, 0] = outp[..., 0, 3]
        # g^rr
        outp[..., 1, 1] = dlta / sgma
        # g^thth
        outp[..., 2, 2] = 1.0 / sgma
        # g^phiphi
        outp[..., 3, 3] = (dlta - self.a2 * sinth2) * inv_sgma_dlta / sinth2

        return outp

    def tetrad(
        self, x: torch.Tensor, zeta: float | torch.Tensor = 0.0, sgn: float = -1
    ) -> torch.Tensor:
        """
        Compute a general local tetrad for Kerr spacetime.

        Parameters
        ----------
        x : torch.Tensor
            Point(s) of computation, shape [..., 4].
        zeta : float or torch.Tensor, optional
            Frame angular velocity, by default 0.0.
        sgn : float, optional
            A sign parameter, by default +1.
           The physical interpretation of this sign is unclear.

        Returns
        -------
        torch.Tensor
            The tetrad vectors of shape [..., 4, 4].
        """
        if isinstance(zeta, (float, int)):
            zeta = zeta * torch.ones([*x.shape[:-1]], dtype=x.dtype, device=x.device)
        # print(zeta.shape)
        outp = torch.zeros(*x.shape, 4, dtype=x.dtype, device=x.device)
        # print(outp.shape)
        r = x[..., 1]
        r2 = r.pow(2)

        costh = torch.cos(x[..., 2])
        sinth = torch.sin(x[..., 2])

        costh2 = torch.pow(costh, 2)
        sinth2 = torch.pow(sinth, 2)

        dlta = r2 + self.a2 - 2 * r
        sgma = r2 + self.a2 * costh2

        dlta_sqrt = dlta.sqrt()
        sgma_sqrt = sgma.sqrt()

        g_tt = -(1 - 2.0 * r / sgma)
        g_tphi = - 2.0 * self.a * r * sinth2 / sgma
        g_phiphi = (r2 + self.a2 - g_tphi*self.a) * sinth2

        gma_inv_sqare = - g_tt - 2 * zeta * g_tphi - zeta.pow(2) * g_phiphi
        # print(gma_inv_sqare.shape)
        gma = gma_inv_sqare.pow(-2)

        # e_(0)^{\mu}
        outp[..., 0, 0] = gma_inv_sqare
        outp[..., 0, 3] = zeta * gma_inv_sqare
        # e_(1)^{\mu}
        outp[..., 1, 1] = dlta_sqrt / sgma_sqrt
        # e_(2)^{\mu}
        outp[..., 2, 2] = 1 / sgma_sqrt
        # e_(3)^{\mu}
        e3_coef = sgn * gma / dlta_sqrt * sinth
        outp[..., 3, 0] = -e3_coef * (g_tphi + zeta * g_phiphi)
        outp[..., 3, 3] = e3_coef * (g_tt + zeta * g_tphi)

        return outp

    def lnrf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the tetrad for the locally non-rotating frame (LNRF).

        Notes
        -----
        This implementation appears complex and may contain hardcoded logic
        or assumptions (e.g., about `omega != 0`). It might benefit from
        simplification or clarification.

        Parameters
        ----------
        x : torch.Tensor
            Coordinates of shape [..., 4].

        Returns
        -------
        torch.Tensor
            The LNRF tetrad vectors of shape [..., 4, 4].
        """
        if self.a == 0:
            return self.tetrad(x)

        outp = torch.zeros(*x.shape, 4, dtype=x.dtype, device=x.device)
        r = x[..., 1]
        r2 = r.pow(2)


        costh = torch.cos(x[..., 2])
        sinth = torch.sin(x[..., 2])

        costh2 = torch.pow(costh, 2)
        sinth2 = torch.pow(sinth, 2)

        dlta = r2 + self.a2 - 2 * r
        sgma = r2 + self.a2 * costh2

        dlta_sqrt = dlta.sqrt()
        sgma_sqrt = sgma.sqrt()

        g_tphi = - 2.0 * self.a * r * sinth2 / sgma
        g_phiphi = (r2 + self.a2 - g_tphi*self.a) * sinth2

        omega = - g_tphi / g_phiphi  # tends to infinity near poles
        _a_sqrt = (2 * self.a * r / omega).sqrt()

        # e_(0)^{\mu}
        e0_coef = _a_sqrt / sgma_sqrt / dlta_sqrt
        outp[..., 0, 0] = e0_coef
        outp[..., 0, 3] = e0_coef * omega
        # e_(1)^{\mu}
        outp[..., 1, 1] = dlta_sqrt / sgma_sqrt
        # e_(2)^{\mu}
        outp[..., 2, 2] = 1.0 / sgma_sqrt
        # e_(3)^{\mu}
        outp[..., 3, 3] = sgma_sqrt / _a_sqrt / sinth

        return outp

    def horizon(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the horizon function `Delta`.

        Parameters
        ----------
        x : torch.Tensor
            Coordinates of shape [..., 4].

        Returns
        -------
        torch.Tensor
            The value of `Delta = r^2 - 2r + a^2`.
        """
        r = x[..., 1]
        r2 = torch.pow(r, 2)
        dlta = r2 + self.a2 - 2 * r

        return dlta


class KerrNewmanBL(Spacetime):
    """
    Kerr-Newman spacetime in Boyer-Lindquist coordinates.

    Parameters
    ----------
    a : float, optional
        Dimensionless spin parameter, by default 0.6.
    q : float, optional
        Dimensionless charge, by default 0.4.

    Attributes
    ----------
    a : float
        Dimensionless spin parameter.
    q : float
        Dimensionless charge.
    a2 : float
        Spin squared.
    q2 : float
        Charge squared.
    r_h : float
        Event horizon radius.
    """

    _coords = "Spherical"
    _has_analytic_conn = False
    _has_analytic_tetrad = False
    # _coords = "BoyerLindquist"

    def __init__(self, a=0.6, q=0.4):

        self.a = a
        self.q = q
        self.a2 = a**2
        self.q2 = q**2
        self.r_h = 1 + math.sqrt(1 - self.a2 - self.q2)
        self.__labels__ = ["t", "r", "\\theta", "\\phi"]
        super().__init__()

    def g(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Kerr-Newman metric tensor.

        Parameters
        ----------
        X : torch.Tensor
            Coordinates of shape [..., 4].

        Returns
        -------
        torch.Tensor
            Metric tensor of shape [..., 4, 4].
        """
        outp = torch.zeros(*X.shape, 4, dtype=X.dtype, device=X.device)

        r = X[..., 1]
        r2 = torch.pow(r, 2)

        costh = torch.cos(X[..., 2])
        sinth = torch.sin(X[..., 2])

        costh2 = torch.pow(costh, 2)
        sinth2 = torch.pow(sinth, 2)

        l2 = r2 + self.a2
        rho2 = r2 + self.a2 * costh2
        dlta = 1 - 2 * r / l2 + self.q2 / r2
        f = l2 * dlta / rho2
        xi2 = l2**2 * sinth2 / rho2

        outp[..., 0, 0] = -f + self.a2 * sinth2 / rho2
        outp[..., 0, 3] = self.a * sinth2 * (f - l2 / rho2)
        outp[..., 3, 0] = outp[..., 0, 3]

        outp[..., 1, 1] = 1 / f
        outp[..., 2, 2] = rho2
        outp[..., 3, 3] = xi2 - self.a2 * sinth2**2

        return outp

    def ginv(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate the inverse Kerr-Newman metric tensor.

        Parameters
        ----------
        X : torch.Tensor
            Coordinates of shape [..., 4].

        Returns
        -------
        torch.Tensor
            Inverse metric tensor of shape [..., 4, 4].
        """
        outp = torch.zeros(*X.shape, 4, dtype=X.dtype, device=X.device)

        r = X[..., 1]
        r2 = torch.pow(r, 2)

        costh = torch.cos(X[..., 2])
        sinth = torch.sin(X[..., 2])

        costh2 = torch.pow(costh, 2)
        sinth2 = torch.pow(sinth, 2)

        l2 = r2 + self.a2
        rho2 = r2 + self.a2 * costh2
        dlta = 1 - 2 * r / l2 + self.q2 / r2
        f = l2 * dlta / rho2
        xi2 = l2**2 * sinth2 / rho2

        outp[..., 0, 0] = -f + self.a2 * sinth2 / rho2
        outp[..., 0, 3] = self.a * sinth2 * (f - l2 / rho2)

        outp[..., 1, 1] = f
        outp[..., 2, 2] = 1 / rho2
        outp[..., 3, 3] = xi2 - self.a2 * sinth2**2

        subdet = outp[..., 0, 0] * outp[..., 3, 3] - outp[..., 0, 3] ** 2

        outp[..., 0, 0] = outp[..., 0, 0] / subdet
        outp[..., 3, 3] = outp[..., 3, 3] / subdet
        outp[..., 0, 3] = outp[..., 0, 3] / subdet
        outp[..., 3, 0] = outp[..., 0, 3]

        return outp

    def horizon(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the horizon function `Delta`.

        Parameters
        ----------
        x : torch.Tensor
            Coordinates of shape [..., 4].

        Returns
        -------
        torch.Tensor
            The value of `Delta = r^2 - 2r + a^2 + q^2`.
        """
        r = x[..., 1]
        r2 = torch.pow(r, 2)
        dlta = r2 - 2 * r + self.a2 + self.q2

        return dlta


if __name__ == "__main__":

    st = KerrBL()

    x = torch.randn(3, 10, 4)
    x[..., 1] += 6.0

    print(st.tetrad_(x).shape)
