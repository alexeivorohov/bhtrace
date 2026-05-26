from typing import List, Dict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .trajectory import Trajectory

def join_trajectories( trajectories: List[Trajectory], fill_reprs: bool = True) -> Trajectory:
    """Joins a list of trajectories to the current one.

    All trajectories must have the same number of steps, same coordinate
    system, and originate from the same particle and spacetime
    configuration.

    Parameters
    ----------
    trajectories : List[Trajectory]
        A list of Trajectory objects to join.
    fill_reprs : bool, optional
        If True, fills all coordinate representations present in any of the
        trajectories. If False, only keeps representations present in all
        of them. Defaults to True.

    Returns
    -------
    Trajectory
        Joined trajectory

    """
    if not trajectories:
        raise ValueError('Attempted to join trajectories, but no trajectories provided')

    # Compatibility checks
    for t in trajectories:
        ...
        # if t.nsteps != self.nsteps:
        #     raise ValueError(
        #         "Cannot join trajectories with different number of steps."
        #     )
        # if t.solution_coordinates != self.solution_coordinates:
        #     raise ValueError(
        #         "Cannot join trajectories with different coordinate systems."
        #     )
        # if t.particle_state != self.particle_state:
        #     raise ValueError("Cannot join trajectories with different particle states.")
        # if t.spacetime_state != self.spacetime_state:
        #     raise ValueError("Cannot join trajectories with different spacetime states.")

    all_X = [t._X for t in trajectories]
    all_P = [t._P for t in trajectories]
    all_affine_t = [t.affine_t for t in trajectories]

    all_last_step = [t.last_step for t in trajectories if t.last_step is not None]

    if len(all_last_step) > 0:
        last_step = max(all_last_step)

    all_keys = set()
    for t in trajectories:
        _keys_ = set(t.__XP_reprs__.keys())
        if fill_reprs:
            all_keys.update(_keys_)
        else:
            all_keys.intersection_update(_keys_)

    # for key in all_keys:
    #     new_X, new_P = self.__getitem__(key)
    #     reprs = [t.__getitem__(key) for t in trajectories]
    #     X_ = [new_X] + [_r_[0] for _r_ in reprs]
    #     P_ = [new_P] + [_r_[1] for _r_ in reprs]

    #     new_X = torch.cat(X_)
    #     new_P = torch.cat(P_)

    #     self.__XP_reprs__[key] = new_X, new_P

    # self.ntraj += sum([t.ntraj for t in trajectories])
    # return self