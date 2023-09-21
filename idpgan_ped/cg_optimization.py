"""
Score and minimize the energy of the CG potential from the cg2aa project:
    https://github.com/huhlim/cg2all/tree/main
Code adapted from:
    https://github.com/huhlim/cg2all/blob/main/cg2all/lib/libloss.py
"""

import os
import time
import torch


AMINO_ACID_s = (
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HSD",
    "HSE",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
    "UNK",
)

aa_one_to_three_dict = {
    "G": "GLY", "A": "ALA", "L": "LEU", "I": "ILE", "R": "ARG", "K": "LYS",
    "M": "MET", "C": "CYS", "Y": "TYR", "T": "THR", "P": "PRO", "S": "SER",
    "W": "TRP", "D": "ASP", "E": "GLU", "N": "ASN", "Q": "GLN", "F": "PHE",
    "H": "HSD", "V": "VAL", "X": "UNK"
}

MAX_RESIDUE_TYPE = len(AMINO_ACID_s)

def calc_chain_bond_angles(xyz):
    ids = torch.tensor([[i, i+1, i+2] for i in range(xyz.shape[1]-2)],
                       dtype=torch.long)
    return calc_angles(xyz, ids)

def calc_angles(xyz, angle_indices):
    ix01 = angle_indices[:, [1, 0]]
    ix21 = angle_indices[:, [1, 2]]

    u_prime = xyz[:,ix01[:,1]]-xyz[:,ix01[:,0]]
    v_prime = xyz[:,ix21[:,1]]-xyz[:,ix01[:,0]]
    u_norm = torch.sqrt((u_prime**2).sum(-1))
    v_norm = torch.sqrt((v_prime**2).sum(-1))

    # adding a new axis makes sure that broasting rules kick in on the third
    # dimension
    u = u_prime / (u_norm[..., None])
    v = v_prime / (v_norm[..., None])

    return torch.arccos((u * v).sum(-1))

def calc_dmap(xyz, epsilon=1e-12):
    B = torch
    if len(xyz.shape) == 2:
        if xyz.shape[1] != 3:
            raise ValueError(xyz.shape)
    elif len(xyz.shape) == 3:
        if xyz.shape[2] != 3:
            raise ValueError(xyz.shape)
    else:
        raise ValueError(xyz.shape)
    if len(xyz.shape) == 3:
        dmap = B.sqrt(
                 B.sum(
                   B.square(xyz[:,None,:,:] - xyz[:,:,None,:]),
                 axis=3) + epsilon)
        exp_dim = 1
    else:
        dmap = B.sqrt(
                 B.sum(
                   B.square(xyz[None,:,:] - xyz[:,None,:]),
                 axis=2) + epsilon)
        exp_dim = 0
    return dmap.unsqueeze(exp_dim)

def calc_dmap_triu(input_data, epsilon=1e-12):
    # Check the shape.
    if len(input_data.shape) == 2:
        if input_data.shape[1] != 3:
            raise ValueError(input_data.shape)
        dmap = calc_dmap(input_data, epsilon, backend)
    elif len(input_data.shape) == 3:
        if input_data.shape[2] != 3:
            raise ValueError(input_data.shape)
        dmap = calc_dmap(input_data, epsilon, backend)
    elif len(input_data.shape) == 4:
        if input_data.shape[1] != 1:
            raise ValueError(input_data.shape)
        if input_data.shape[2] != input_data.shape[3]:
            raise ValueError(input_data.shape)
        dmap = input_data
    else:
        raise ValueError(input_data.shape)
    # Get the triu ids.
    l = dmap.shape[2]
    triu_ids = torch.triu_indices(l, l, offset=1)

    # Returns the values.
    if len(input_data.shape) != 2:
        return dmap[:,0,triu_ids[0],triu_ids[1]]
    else:
        return dmap[0,triu_ids[0],triu_ids[1]]


class CoarseGrainedGeometryEnergy(object):

    def __init__(self, seq, vdw_offset=0.1, device="cpu"):
        self.vdw_offset = vdw_offset
        self.set_param(device)
        self.seq = seq
        aa_ids = [AMINO_ACID_s.index(aa_one_to_three_dict[aa_i]) \
                  for aa_i in seq]
        self.aa_ids = torch.tensor(aa_ids, dtype=torch.long, device=device)

    def set_param(self, device):
        self.b_len0 = None
        self.b_ang0 = None
        self.vdw = torch.zeros(
            (MAX_RESIDUE_TYPE, MAX_RESIDUE_TYPE),
            dtype=torch.float, device=device)
        ca_pot_params_fp = os.path.join(os.path.dirname(__file__),
                                        "calpha_geometry_params.dat")
        with open(ca_pot_params_fp) as fp:
            for line in fp:
                x = line.strip().split()
                if x[0] == "BOND_LENGTH":
                    self.b_len0 = (float(x[1]), float(x[2]))
                elif x[0] == "BOND_ANGLE":
                    self.b_ang0 = (float(x[1]), float(x[2]))
                else:
                    i = AMINO_ACID_s.index(x[1])
                    j = AMINO_ACID_s.index(x[2])
                    self.vdw[i, j] = float(x[3]) + self.vdw_offset

    def eval(self, xyz, get_terms=False):
        bonded = self.eval_bonded(xyz, get_terms=True)
        bond_energy = bonded["bond"]
        angle_energy = bonded["angle"]
        vdw_energy = self.eval_vdw(xyz)
        if get_terms:
            return {"bond": bond_energy, "angle": angle_energy, "vdw": vdw_energy}
        else:
            return bond_energy + vdw_energy + vdw_energy

    def eval_bonded(self, xyz, get_terms=True):
        # Bond length.
        bond_l = torch.sqrt(torch.square(xyz[:,:-1,:] - xyz[:,1:,:]).sum(dim=2))
        bond_energy = torch.square((bond_l - self.b_len0[0]) / self.b_len0[1]).sum(dim=1)
        # Bond angles.
        angles = calc_chain_bond_angles(xyz)
        angle_energy = torch.square((angles - self.b_ang0[0]) / self.b_ang0[1]).sum(dim=1)
        if get_terms:
            return {"bond": bond_energy, "angle": angle_energy}
        else:
            return bond_energy + angle_energy

    def eval_vdw(self, xyz, weight=1000):
        dmap = calc_dmap(xyz)
        triu_ids = torch.triu_indices(xyz.shape[1], xyz.shape[1], offset=3)
        dmap_triu = dmap[:,0,triu_ids[0],triu_ids[1]]
        aa_ids_i = self.aa_ids[triu_ids[0]]
        aa_ids_j = self.aa_ids[triu_ids[1]]
        vdw_params = self.vdw[aa_ids_i, aa_ids_j]
        clashes = torch.clip(dmap_triu - vdw_params, min=None, max=0)*weight
        vdw_energy = torch.square(clashes).sum(axis=1)
        return vdw_energy


def optimize(xyz: torch.Tensor,
             seq: str,
             step: float = 1e-7,
             n_steps: int = 500):
    """
    Perform steepest descent using the CG potential energy function.
    """
    cg_energy = CoarseGrainedGeometryEnergy(seq=seq)

    xyz_init = xyz
    e_ini = None
    e_end = None
    for i in range(n_steps):
        xyz = xyz.detach().clone().requires_grad_(True)
        e_i = cg_energy.eval(xyz)
        if i == 0:
            _e_ini = e_i.detach().numpy()
        elif i == n_steps-1:
            _e_end = e_i.detach().numpy()
        
        e_i.backward(torch.ones_like(e_i))
        xyz = xyz - xyz.grad*step
        if i == n_steps-1:
            rmsd_init = torch.sqrt(
                torch.mean(
                    torch.sum(
                        torch.square(xyz_init - xyz.detach()),
                        dim=2),
                    dim=1
                )
            ).numpy()
    return xyz.detach(), (_e_ini, _e_end, rmsd_init)