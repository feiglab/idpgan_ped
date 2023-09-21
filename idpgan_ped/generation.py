import os
from idpgan import load_netg_article


def load_netg(data_dp="data"):
    model_fp = os.path.join(data_dp, "generator.pt")
    netg = load_netg_article(model_fp)
    return netg

def generate_ensemble(netg, seq, n_samples=1000, batch_size=256):
    """Generate the CG ensemble with COCOMO-based idpGAN."""
    xyz_gen = netg.predict_idp(n_samples=n_samples,
                               aa_seq=seq,
                               batch_size=batch_size,
                               get_a=False)
    return xyz_gen