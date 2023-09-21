import os
import argparse
import pathlib
import numpy as np
from Bio.PDB import PDBParser
from idpgan_ped.coords import calc_chain_dihedrals


parser = argparse.ArgumentParser(
    description='Build a file with values of a C-alpha torsional angles'
                ' for disordered protein structures.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-i', '--pdb_dp', type=str, required=True,
    help='Input directory with a set of AlphaFold structures.')
parser.add_argument('-p', '--pred_fp', type=str, required=True,
    help='Output file from the alphafold_disorder.py script from the'
         ' AlphaFold-disorder GitHub repository. The script must have been'
         ' use on the directory specified using the -i argument of this'
         ' script.')
parser.add_argument('-o', '--out_fp', type=str, required=True,
    help='Output file.')
parser.add_argument('-t', '--idr_score_t', type=float, default=0.581,
    help='Threshold score for defining an IDR residue. Obtained from:'
         ' https://pubmed.ncbi.nlm.nih.gov/36210722/')

# parser.add_argument('--foo', action='store_true')
args = parser.parse_args()


# Amino acids.
aa_three_to_one_dict = {
    "GLY": "G", "ALA": "A", "LEU": "L", "ILE": "I", "ARG": "R", "LYS": "K",
    "MET": "M", "CYS": "C", "TYR": "Y", "THR": "T", "PRO": "P", "SER": "S",
    "TRP": "W", "ASP": "D", "GLU": "E", "ASN": "N", "GLN": "Q", "PHE": "F",
    "HIS": "H", "VAL": "V", "UNK": "X"
}

aa_one_letter = tuple("QWERTYIPASDFGHKLCVNMX")


# Parse the output file of the 'alphafold_disorder.py' script.
print(f"# Reading the output of AlphaFold-disorder at: {args.pred_fp}")
idr_data = {}
tot_residues = 0
with open(args.pred_fp, "r") as i_fh:
    i_fh.readline()  # Exclude the first line.
    for line in i_fh:
        fields = line.split()
        filename = fields[0]
        res_num = int(fields[1])
        idr_score = float(fields[7])
        tot_residues += 1
        # Found an residue predicted to be in a IDR.
        if idr_score < args.idr_score_t:
            continue
        if filename not in idr_data:
            idr_data[filename] = [(res_num, idr_score)]
        else:
            idr_data[filename].append((res_num, idr_score))
print(f"- Found {len(idr_data)} models with IDR residues and"
      f" {sum([len(idr_data[k]) for k in idr_data])} IDR residues"
      f" (out of {tot_residues} total residues).")


# Collect statistics for each PDB file.
print("# Extracting torsion angle values from PDB files.")
tor_values = []
for k, filename in enumerate(sorted(idr_data.keys())):
    idr_residues = idr_data[filename]
    print(f"* {filename} {k}/{len(idr_data)}")

    pdb_fp = os.path.join(args.pdb_dp, filename + ".pdb")
    structure = PDBParser(QUIET=True).get_structure(filename, pdb_fp)
    residues = list(structure.get_residues())
    print(f"- n_idr_res={len(idr_residues)}, n_tot_res={len(residues)}")

    xyz_ca = [res["CA"].get_coord() for res in residues]
    xyz_ca = np.stack(xyz_ca)[None,...]
    tor_ca = calc_chain_dihedrals(xyz_ca)
    for idr_res_i in idr_residues:
        if idr_res_i[0] == 1:  # The first residue of a chain can't be used.
            continue 
        if idr_res_i[0] >= len(residues) - 1:  # Last two residues can't be used.
            continue
        tor_id = idr_res_i[0]-2
        res_im1 = residues[idr_res_i[0]-1-1]
        res_i = residues[idr_res_i[0]-1]
        res_ip1 = residues[idr_res_i[0]-1+1]
        res_ip2 = residues[idr_res_i[0]-1+2]
        tor_values.append([
            # Torsion angle value.
            tor_ca[0, tor_id],
            # Amino acid type indices.
            aa_one_letter.index(aa_three_to_one_dict[res_im1.get_resname()]),
            aa_one_letter.index(aa_three_to_one_dict[res_i.get_resname()]),
            aa_one_letter.index(aa_three_to_one_dict[res_ip1.get_resname()]),
            aa_one_letter.index(aa_three_to_one_dict[res_ip2.get_resname()])
        ])

tor_values = np.array(tor_values)
print(f"\n# Collected {tor_values.shape[0]} torsion angle values.")
print(f"- Saving values at {args.out_fp}.")
np.save(args.out_fp, tor_values)