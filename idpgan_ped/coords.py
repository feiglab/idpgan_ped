import numpy as np


def calc_chain_dihedrals(xyz):
    r_sel = xyz
    b0 = -(r_sel[:,1:-2,:] - r_sel[:,0:-3,:])
    b1 = r_sel[:,2:-1,:] - r_sel[:,1:-2,:]
    b2 = r_sel[:,3:,:] - r_sel[:,2:-1,:]
    b0xb1 = np.cross(b0, b1)
    b1xb2 = np.cross(b2, b1)
    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)
    y = np.sum(b0xb1_x_b1xb2*b1, axis=2)*(1.0/np.linalg.norm(b1, axis=2))
    x = np.sum(b0xb1*b1xb2, axis=2)
    dh_vals = np.arctan2(y, x)
    return dh_vals