import os
import pathlib
import argparse
import numpy as np
from Bio.PDB import PDBParser


def get_ca_xyz_array(in_dp: str):
    """
    Parse a directory with N PDB files (each one must have L residues) and
    returns a numpy array of shape (N, L, 3) storing the xyz coordinates of
    the C-alpha atoms in the PDB files.
    """
    in_path = pathlib.Path(in_dp)
    pdb_paths = list(in_path.glob("./*.pdb"))
    if not pdb_paths:
        raise FileNotFoundError(f"No PDB files in {in_dp}")
    parser = PDBParser(QUIET=True)
    ca_xyz = []
    for i, pdb_path in enumerate(pdb_paths):
        s_i = parser.get_structure(f"s_{i}", pdb_path)
        ca_xyz_i = []
        for m in s_i:
            for r_j in m.get_residues():
                if r_j.get_id()[0] == ' ':
                    if "CA" in r_j:
                        ca_pos_j = r_j["CA"]
                        ca_xyz_i.append(ca_pos_j.get_coord())
            break
        if not ca_xyz_i:
            raise ValueError(f"No CA atoms found for {pdb_path}")
        ca_xyz_i = np.stack(ca_xyz_i)[None,...]
        ca_xyz.append(ca_xyz_i)
    ca_xyz = np.concatenate(ca_xyz, axis=0)
    return ca_xyz

def score_kld_approximation(v_true, v_pred, n_bins=50, pseudo_c=0.001):
    """
    Scores an approximation of KLD by discretizing the values in
    'v_true' (data points from a reference distribution) and 'v_pred'
    (data points from a predicted distribution).
    """
    # Define bins.
    _min = min((v_true.min(), v_pred.min()))
    _max = max((v_true.max(), v_pred.max()))
    bins = np.linspace(_min, _max, n_bins+1)
    # Compute the frequencies in the bins.
    ht = (np.histogram(v_true, bins=bins)[0]+pseudo_c)/v_true.shape[0]
    hp = (np.histogram(v_pred, bins=bins)[0]+pseudo_c)/v_pred.shape[0]
    kl = -np.sum(ht*np.log(hp/ht))
    return kl, bins

def score_jsd_approximation(v_true, v_pred, n_bins=50, pseudo_c=0.001):
    """
    Scores an approximation of JS by discretizing the values in
    'v_true' (data points from a reference distribution) and 'v_pred'
    (data points from a predicted distribution).
    """
    # Define bins.
    _min = min((v_true.min(), v_pred.min()))
    _max = max((v_true.max(), v_pred.max()))
    bins = np.linspace(_min, _max, n_bins+1)
    # Compute the frequencies in the bins.
    ht = (np.histogram(v_true, bins=bins)[0]+pseudo_c)/v_true.shape[0]
    hp = (np.histogram(v_pred, bins=bins)[0]+pseudo_c)/v_pred.shape[0]
    hm = (ht + hp)/2
    kl_tm = -np.sum(ht*np.log(hm/ht))
    kl_pm = -np.sum(hp*np.log(hm/hp))
    js = 0.5*kl_pm + 0.5*kl_tm
    return js, bins

def score(xyz_ref: np.array,
          xyz_hat: np.array,
          n_bins: int = 50,
          method: str = "js"):
    """
    See the idpGAN article.
    """
    # Calculate distance maps.
    dmap_ref = calc_dmap(xyz_ref)
    dmap_hat = calc_dmap(xyz_hat)
    if dmap_ref.shape[1] != dmap_hat.shape[1]:
        raise ValueError(
            "Input trajectories have different number of residues:"
            f" ref={dmap_ref.shape[1]}, hat={dmap_hat.shape[1]}")
    n_akld_d = []
    if method == "kl":
        score_func = score_kld_approximation
    elif method == "js":
        score_func = score_jsd_approximation
    else:
        raise KeyError(method)
    for i in range(dmap_ref.shape[1]):
        for j in range(dmap_ref.shape[1]):
            if i+1 >= j:
                continue
            kld_d_ij = score_func(
                dmap_ref[:,i,j], dmap_hat[:,i,j],
                n_bins=n_bins)[0]
            n_akld_d.append(kld_d_ij)
    return np.mean(n_akld_d)

def calc_dmap(xyz: np.array,
              epsilon: float = 1e-12):
    """
    Takes as input xyz arrays of shape (N, L, 3) and returns a distance map of
    shape (N, L, L). 
    """
    if len(xyz.shape) != 3 and xyz.shape[2] != 3:
        raise ValueError(xyz.shape)
    dmap = np.sqrt(
                np.sum(
                    np.square(xyz[:,None,:,:] - xyz[:,:,None,:]),
                axis=3) + epsilon)
    return dmap


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    description='Score the aKLD_d of two structural ensembles.')
    parser.add_argument('-r', '--ref_dp', type=str, required=True,
        help='Directory with the PDB files for the reference ensemble.')
    parser.add_argument('-t', '--hat_dp', type=str, nargs='+', required=True,
        help='Directory (or directories) with the PDB files for the proposed'
            ' ensemble(s).')
    parser.add_argument('-b', '--n_bins', type=int, default=50,
        help='Number of bins to use in the discretized KLD calculation.')
    parser.add_argument('-m', '--method', type=str, default='js',
        choices=['kl', 'js'],
        help='Type of divergence to score.')
    args = parser.parse_args()

    ref_xyz = get_ca_xyz_array(args.ref_dp)
    for hat_dp in args.hat_dp:
        hat_xyz = get_ca_xyz_array(hat_dp)
        a_kdl_d = score(ref_xyz, hat_xyz, n_bins=args.n_bins,
                        method=args.method)
        print(f"# Scoring {hat_dp}")
        print(f"- aKLD_d: {a_kdl_d:.4f}")