import torch
from .odeint import RK4


class WeierstrassElliptic:
    """
    Weierstrass elliptic function defined by invariants g2 and g3. 
    
    It satisfies the differential equation:
    (d/dz p(z))^2 = 4 p(z)^3 - g2 p(z) - g3

    Attributes
    ----------
    g2 : float | torch.Tensor
        Second invariant of Weierstrass elliptic function.
    g3 : float | torch.Tensor
        Third invariant of Weierstrass elliptic function.

    Methods
    -------
    __call__(z: torch.Tensor) -> torch.Tensor
        Evaluate Weierstrass elliptic function at given z.
    d(z: torch.Tensor) -> torch.Tensor
        Derivative of Weierstrass elliptic function with respect to z.
    
    """

    def __init__(self, g2: float | torch.Tensor, g3: float | torch.Tensor):
        """
        Initialize Weierstrass elliptic function with given invariants.

        Parameters
        ----------
        g2 : float | torch.Tensor
            Second invariant of Weierstrass elliptic function.
        g3 : float | torch.Tensor
            Third invariant of Weierstrass elliptic function.

        Notes
        -----
        If g2 and g3 are specified as tensors, they should have the same shape.
        The function then can be evaluated only for z with shapes, broadcastable to the shape of g2 and g3.
        """


        self.g2 = torch.as_tensor(g2)
        self.g3 = torch.as_tensor(g3)

    def _get_initial_conditions(self, z_start, dtype, device):
        g2 = self.g2
        g3 = self.g3
        z0 = torch.tensor(z_start, dtype=dtype, device=device)

        # Series expansion for p(z) and p'(z) near z=0
        p_z0 = 1/z0**2 + g2*z0**2/20 + g3*z0**4/28
        dp_z0 = -2/z0**3 + g2*z0/10 + g3*z0**3/7
        return p_z0, dp_z0

    def _solve_ode(self, z: torch.Tensor):
        """
        Solves the ODE for p(z) and p'(z).
        Assumes z is real-valued.
        """
        z_start = 1e-2
        dt = 1e-4

        p_results = torch.zeros_like(z)
        dp_results = torch.zeros_like(z)
        
        z_batch = z.flatten()

        p0, dp0 = self._get_initial_conditions(z_start, z.dtype, z.device)

        for i, z_val in enumerate(z_batch):
            is_neg = z_val < 0
            z_end_val = abs(z_val.item())
            z_end_tensor = z_val.abs()

            # Handle small z values using series expansion
            if z_end_val == 0:
                p_results.flatten()[i] = torch.inf
                dp_results.flatten()[i] = -torch.inf
                continue
            if z_end_val <= z_start:
                p_val = 1/z_end_tensor**2 + self.g2*z_end_tensor**2/20 + self.g3*z_end_tensor**4/28
                dp_val = -2/z_end_tensor**3 + self.g2*z_end_tensor/10 + self.g3*z_end_tensor**3/7
                p_results.flatten()[i] = p_val
                if is_neg:
                    dp_val = -dp_val
                dp_results.flatten()[i] = dp_val
                continue

            # Solve ODE for z_end_val > z_start
            t0 = z_start
            n_steps = int((z_end_val - t0) / dt)
            if n_steps <= 0:
                n_steps = 1
            current_dt = (z_end_val - t0) / n_steps
            
            solver = RK4(dt=current_dt)

            def weierstrass_ode_term(t, p, dp):
                g2_inv = self.g2
                p_double_prime = 6 * p**2 - g2_inv / 2
                return dp, p_double_prime
            
            # Unsqueeze to create a batch of 1
            Y0 = (p0.unsqueeze(0), dp0.unsqueeze(0))
            
            solution = solver.forward(weierstrass_ode_term, Y0, t0, n_steps, tqdm_bar=False)
            
            final_p = solution['Y'][0][0, -1]
            final_dp = solution['Y'][1][0, -1]
            
            p_results.flatten()[i] = final_p
            if is_neg:
                final_dp = -final_dp # p' is an odd function
            dp_results.flatten()[i] = final_dp

        return p_results.view_as(z), dp_results.view_as(z)

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        """Evaluate Weierstrass elliptic function at given z.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor for evaluation of Weierstrass elliptic function.
            Should be real-valued.
        
        Returns
        -------
        torch.Tensor
            Weierstrass elliptic function evaluated at given z.

        """
        p_vals, _ = self._solve_ode(z)
        return p_vals

    def d(self, z: torch.Tensor) -> torch.Tensor:
        """
        Derivative of Weierstrass elliptic function with respect to z.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor for evaluation of Weierstrass elliptic function.
            Should be real-valued.
        """
        _, dp_vals = self._solve_ode(z)
        return dp_vals