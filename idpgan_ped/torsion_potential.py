import numpy as np
import torch
from idpgan.coords import torch_chain_dihedrals


tor_scores = np.array([
    2.1436, 2.1098, 2.0857, 2.0676, 2.0641, 2.0686, 2.0886, 2.1083,
    2.1327, 2.156, 2.1846, 2.2239, 2.2706, 2.3458, 2.4378, 2.5513,
    2.6816, 2.8086, 2.9103, 2.9885, 3.0347, 3.076, 3.1131, 3.1377,
    3.154, 3.1739, 3.1758, 3.1857, 3.1738, 3.179, 3.1645, 3.1512,
    3.1393, 3.1167, 3.0795, 3.0373, 2.9859, 2.907, 2.6695, 1.7389,
    1.5859, 2.1782, 2.3514, 2.4548, 2.5282, 2.5851, 2.6205, 2.6463,
    2.657, 2.6532, 2.6517, 2.6474, 2.6302, 2.6067, 2.5728, 2.5281,
    2.4855, 2.4303, 2.3729, 2.3164, 2.2634, 2.2173, 2.1764])

tor_bins = np.linspace(-np.pi, np.pi, tor_scores.shape[0]+1)

def score_torsion(xyz: torch.Tensor):
    tor = torch_chain_dihedrals(xyz)
    tor_discrete = np.digitize(tor, tor_bins)-1
    return tor_scores[tor_discrete].sum(axis=1)

def select_mirror_images(xyz: torch.Tensor):
    xyz_ori = xyz.clone()
    xyz_rev = xyz.clone()
    # First score the torsion potential on the original conformations.
    s_ori = score_torsion(xyz_ori)
    # Create a mirror image of the conformations.
    xyz_rev[:,:,0] = -xyz_ori[:,:,0]
    # Score the potential on the mirror images.
    s_rev = score_torsion(xyz_rev)
    # Select those conformations having a lower energy in the mirror image and
    # substitute.
    reflect_mask = s_ori > s_rev
    xyz_ori[reflect_mask] = xyz_rev[reflect_mask]
    s_sel = score_torsion(xyz_ori)
    return xyz_ori, (s_ori, s_sel)