# from typing import Dict, List, Tuple

# import pytest
# import matplotlib.pyplot as plt

# import torch

# from bhtrace.geometry.exact.newtonian import KeplerianTrajectories

# N = 16
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# x_, y_ = 0, 0

# def plot_onesample(params: Dict[str, torch.Tensor], elliptic: KeplerianTrajectories):

#     ax.plot(x_, y_)
#     ax.grid("on")
#     ax.set_xlim(-25, 25)
#     ax.set_ylim(-25, 25)
#     ax.set_aspect("equal")
#     plt.show()


# @pytest.fixture
# def init_params() -> Dict[str, Dict[torch.Tensor]]:
#     return {
#         'minimal': {
#             'p': torch.tensor([10.0, 10.0, 8.0]),
#             'eps': torch.tensor([0.5, 0.5, 0.8]),
#         }
#     }

# def test_initialization(init_params):

#     elliptic = KeplerianTrajectories(
        
#     )

#     nu = torch.linspace(0, torch.pi * 1.8, 128).unsqueeze(-1)

#     traj = elliptic.phase_sample(nu, coords="cart2d")

#     x_, y_ = traj.unbind(dim=-1)

# # --- Kinematic test ---
# @pytest.fixture
# def init_kinematic_params() -> Dict[str,Dict[str, torch.Tensor]]:
#     return {
#         "var_radius": {
#             "r": torch.tensor([5.0, 10.0, 15.0]),
#             "nu": torch.tensor([0.0, 0.0, 0.0]),
#             "v_r": torch.tensor([0.05, 0.05, 0.05]),
#             "v_nu": torch.tensor([0.2, 0.2, 0.2]),
#             "mu": torch.tensor([100.0, 100.0, 100]),
#         },
#         "var_mass": {
#             "r": torch.tensor([10.0, 10.0, 10.0]),
#             "nu": torch.tensor([0.0, 0.0, 0.0]),
#             "v_r": torch.tensor([0.05, 0.05, 0.05]),
#             "v_nu": torch.tensor([0.2, 0.2, 0.2]),
#             "mu": torch.tensor([50.0, 100.0, 200.0]),
#         },
#     }

# @pytest.mark.parametrize(['case'], [])
# def test_kinematic_init(init_kinematic_params, case):

#     if test_case in kinematic_samples.keys():

#         kwargs = kinematic_samples[test_case]

#     elliptic = KeplerianTrajectories.from_kinematic(**kwargs)

#     nu = torch.linspace(0, torch.pi * 1.0, 128).unsqueeze(-1)
#     traj = elliptic.phase_sample(nu, coords="cart2d")
#     x_, y_ = traj.unbind(dim=-1)

# # --- Cartesian 2d test ---
# @pytest.fixture
# def init_cartesian_2d_params() -> Dict[str, Dict[str, torch.Tensor]]:
#     return {
#         "conj_ic": {
#             "x": torch.tensor([10.0, -10.0]),
#             "y": torch.tensor([10.0, 10.0]),
#             "v_x": torch.tensor([-0.2, -0.2]),
#             "v_y": torch.tensor([0.2, -0.2]),
#             "mu": torch.tensor([200.0, 200.0]),
#         },
#         "var_x_perp": {
#             "x": torch.tensor([10.0, 5.0, -5.0]),
#             "y": torch.tensor([1.0, 1.0, 1.0]),
#             "v_x": torch.tensor([0.0, 0.0, 0.0]),
#             "v_y": torch.tensor([0.2, 0.2, 0.2]),
#             "mu": torch.tensor([100.0, 100.0, 100.0]),
#         },
#         "var_y_perp": {
#             "x": torch.tensor([5.0, 5.0, 5.0,]),
#             "y": torch.tensor([-10, -5.0, 5.0,]),
#             "v_x": torch.tensor([-0.2, -0.2, -0.2,]),
#             "v_y": torch.tensor([0.0, 0.0, 0.0,]),
#             "mu": torch.tensor([100.0, 100.0, 100.0,]),
#         },
#         "var_y_par": {
#             "x": torch.tensor([10.0, 10.0, 10.0]),
#             "y": torch.tensor([-5.0, 5.0, 10.0]),
#             "v_x": torch.tensor([0.0, 0.0, 0.0]),
#             "v_y": torch.tensor([0.2, 0.2, 0.2]),
#             "mu": torch.tensor([100.0, 100.0, 100.0]),
#         },
#         'massive': {
#             'x': torch.linspace(8, 16, N),
#             'y': torch.linspace(0, 8, N),
#             'v_x': -torch.linspace(0.2, 0.1, N),
#             'v_y': torch.linspace(0.3, 0.1, N),
#             'mu': torch.linspace(200.0, 200.0, N),
#         },
#         'massively_par': {
#             'x': torch.linspace(16, 16, N),
#             'y': torch.linspace(2, 10, N),
#             'v_x': -torch.linspace(0.1, 0.2, N),
#             'v_y': torch.linspace(0.1, 0.2, N),
#             'mu': torch.linspace(200.0, 200.0, N),
#         }
#     }

# def test_init_cartesian_2d(init_cartesian_2d_params):


#     if test_case in cartesian_2d_samples.keys():

#     kwargs = cartesian_2d_samples[test_case]

#     elliptic = KeplerianTrajectories.from_cartesian_2d(**kwargs)

#     nu = torch.linspace(0, torch.pi * 1.5, 128).unsqueeze(-1)

#     traj = elliptic.phase_sample(nu, coords="cart2d")

#     x_, y_ = traj.unbind(dim=-1)
#     ax.quiver(kwargs["x"], kwargs["y"], kwargs["v_x"], kwargs["v_y"], pivot="mid")



