import os
import argparse
import numpy as np


parser = argparse.ArgumentParser(
    description='Build a file with the parameters of a C-alpha torsional angle'
                ' potential for disordered protein structures.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-i', '--in_fp', type=str, required=True,
    help='Input file. It must be the ouput of the'
         ' scripts/torsion_potential_stats.py script of this package.')
parser.add_argument('-n', '--n_bins', type=int, default=64,
    help='Number of bins to use for collecting the data for the potential.')

# parser.add_argument('--foo', action='store_true')
args = parser.parse_args()


tor_data = np.load(args.in_fp)
tor_values = tor_data[:,0]
eps = 1e-6
bins = np.linspace(-np.pi, np.pi, args.n_bins)
counts = np.histogram(tor_values, bins)[0] + eps
scores = -0.59*np.log(counts/tor_values.shape[0])
print("# Parameters for the Ca torsion angle potential:")
print(scores.round(4).tolist())