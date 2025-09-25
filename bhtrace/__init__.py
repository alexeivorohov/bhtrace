"""
PyTorch-based library for modeling images of compact objects 


"""
# About variable notation:
# 1. We use _ to distinguish already computed quantities from computation methods (e.g. F vs _F, g vs _g and etc.)
# 2. For tensors, contravariant (vector) indexes come after variable, covariant - before variable (e.g. ik_g vs g_ik)
# 3. As ususal, we use i, k, j, m, n, l letters to designate spatial indexes and u, v, w, p, q letters for spacetime indexes. Common batch and time slice indexes are b and N.

from .trajectory import Trajectory

