import os
import sys
import json
import argparse
import shutil
import csv
import time
import pathlib
from datetime import datetime
import numpy as np
import mdtraj
import torch
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
sys.path.append(".")
from idpgan_ped import cg_optimization
try:
    from idpgan_ped import aa_optimization
    has_openmm = True
except ImportError:
    has_openmm = False
from idpgan_ped.utils import get_sel_ids, get_mod_seq
from idpgan_ped.generation import load_netg, generate_ensemble
from idpgan_ped.torsion_potential import score_torsion, select_mirror_images
from idpgan_ped.data import get_ca_topology
from idpgan_ped.cg2all_lib import run_cg2all


parser = argparse.ArgumentParser(
    description='Generate ensembles with idpGAN.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-i', '--input_fp', type=str, required=True,
    help='Input JSON file with a list of all PED entries to process. See the'
         ' input.template.json file in the data directory of this package for'
         ' an example input file.')
parser.add_argument('-c', '--config_fp', type=str, default='config/main.json',
    help='JSON configuration file. It specifies the output directory of this'
         ' script among other things.')
parser.add_argument('-n', '--n_samples', type=int, default=1000,
    help='Number of samples in the final ensembles.')
parser.add_argument('-N', '--n_ini_samples', type=int, default=None,
    help='Number of samples in the original ensemble. If used, idpGAN'
         ' will generate this number of samples, then they will be scored with'
         ' the torsion potential and the top --n_samples will be selected.')
parser.add_argument('--batch_size', type=int, default=256,
    help='IdpGAN batch size.')
parser.add_argument('--skip_cg_opt', action='store_true',
    help='Do not optimize the energy of the CG structures.')
parser.add_argument('--step_cg_opt', type=float, default=1e-7,
    help='Step size in the CG energy minimization procedure.')
parser.add_argument('--n_steps_cg_opt', type=int, default=500,
    help='Number of steps in the CG energy minimization procedure.')
parser.add_argument('--unpatched_cg2all', action='store_true',
    help='Use the original version of cg2all script. By default, it will use a'
         ' patched version of the script (August 23 2023).')
parser.add_argument('--skip_aa_opt', action='store_true',
    help='Do not optimize the energy of the all atom structures using OpenMM.')
parser.add_argument('--n_steps_aa_opt', type=int, default=250,
    help='Number of steps in the all atom energy minimization procedure.'
         ' A value of 0 corresponds to letting OpenMM minimize until'
         ' convergence.')
parser.add_argument('--selection', type=str, default=None,
    help='Selected entry ids. The ids are the indices of the entries in the'
         ' list found in the input.json file. Example value: 0-7,9,12-40.')
parser.add_argument('--openmm_device', type=str, default='cpu',
    choices=('cpu', 'cuda'), help='Device to use for OpenMM calculations.')
parser.add_argument('--seed', type=int, default=2023, help='Random seed.')
args = parser.parse_args()


# Parameters and I/O directories.
if not args.skip_aa_opt and not has_openmm:
    raise ImportError(
        "OpenMM is not installed, can not minimize the energy of all atom"
        " structures")

with open(args.config_fp, "r") as i_fh:
    config = json.load(i_fh)

n_ini_samples = args.n_ini_samples if args.n_ini_samples is not None \
                                   else args.n_samples
n_samples = args.n_samples
sel_ids = get_sel_ids(args.selection) if args.selection is not None else None

idpgan_data_dp = config["data"]["idpgan_data_dp"]
if not os.path.isdir(idpgan_data_dp):
    raise FileNotFoundError(idpgan_data_dp)
ensembles_dp = config["data"]["ensembles_dp"]
if not os.path.isdir(ensembles_dp):
    os.mkdir(ensembles_dp)

if args.unpatched_cg2all:
    cg2all_default_cmd = ["convert_cg2all"]
else:
    cg2all_default_cmd = ["python", "scripts/cg2all/convert_cg2all.patch.0.py"]


# Load files with PED entries and their sequences.
with open(args.input_fp) as i_fh:
    entries = json.load(i_fh)
print(f"# Found {len(entries)} input PED entries.")


# Load the idpGAN generator network.
print("# Loading idpGAN.")
netg = load_netg(data_dp=idpgan_data_dp)


# Columns for a csv file for storing ensemble data.
csv_cols = ["id"]
if n_ini_samples != n_samples:
    csv_cols.extend(["tor_e"])
else:
    csv_cols.extend(["tor_e_ini", "tor_e_end"])
if not args.skip_cg_opt:
    csv_cols.extend(["cg_e_ini", "cg_e_end", "cg_rmsd"])
if not args.skip_aa_opt:
    csv_cols.extend(["aa_e_ini", "aa_e_end", "aa_rmsd"])


# Generate the ensembles.
print("# Generating the ensembles.")

torch.manual_seed(args.seed)

if sel_ids is not None:
    entries = [e for (i, e) in enumerate(entries) if i in sel_ids]
    if not entries:
        raise ValueError("No entries selected")

for i, entry in enumerate(sorted(entries, key=lambda e: e["id"])):
    
    print(f"* Generating an ensemble for {entry['id']} {i+1}/{len(entries)}.")
    print(f"- Sequence has {len(entry['seq'])} residues.")

    # Create a directory for the ensemble of this entry.
    ensemble_dp = os.path.join(ensembles_dp, entry["id"])
    if os.path.isdir(ensemble_dp):
        shutil.rmtree(ensemble_dp)
    os.mkdir(ensemble_dp)

    # Store some data about the ensemble generation process.
    # Original sequence.
    raw_seq = entry["seq"]
    # Sequence with amino acid substitutions to account for physical effects of
    # residue modificatins (e.g.: the THR modified in TPO is substituted by
    # ASP). This is the input seque of idpGAN.
    modifications = entry.get("modifications", [])
    if modifications:
        print(f"- Found {len(modifications)} residue modifications, converting"
              " the raw sequence.")
    mod_seq = get_mod_seq(raw_seq, modifications)
    # Ensemble data.
    ensemble_data = {
        "id": entry["id"], "len": len(entry["seq"]),
        "n_modifications": len(modifications)
    }
    # Data for each structure.
    csv_data = {"id": list(range(0, args.n_samples))}

    # Generate the initial CG ensemble with idpGAN.
    print("- Generating CG conformations with idpGAN.")
    t_gen = time.time()
    xyz_gen = generate_ensemble(netg=netg,
                                seq=mod_seq,
                                n_samples=n_ini_samples,
                                batch_size=args.batch_size)
    t_gen = time.time() - t_gen
    ensemble_data["t_gen"] = t_gen
    print(f"- Generated {xyz_gen.shape[0]} samples. It took {t_gen:.2f} s.")

    # Select the correct mirror images of the conformations.
    print("- Selecting the most likely mirror images.")
    xyz_sel, sel_data = select_mirror_images(xyz=xyz_gen)
    print("- Selection results:"
          f" e_ini={sel_data[0].mean():.2f},"
          f" e_end={sel_data[1].mean():.2f}")

    if n_ini_samples != n_samples:
        score_sel = score_torsion(xyz_sel)
        score_ids = score_sel.argsort()
        xyz_sel = xyz_sel[score_ids[:n_samples]]
        csv_data["tor_e"] = score_sel[score_ids[:n_samples]]
        ensemble_data["tor_e"] = float(np.mean(csv_data["tor_e"]))
        print(f"- Selected top {xyz_sel.shape[0]} samples:"
              f" e_sel={ensemble_data['tor_e']:.2f}")
    else:
        xyz_sel = xyz_gen
        csv_data["tor_e_ini"] = sel_data[0]
        csv_data["tor_e_end"] = sel_data[1]
        ensemble_data["tor_e_ini"] = float(np.mean(csv_data["tor_e_ini"]))
        ensemble_data["tor_e_end"] = float(np.mean(csv_data["tor_e_end"]))

    # Optimize the conformations by minimizing a CG energy function.
    if not args.skip_cg_opt:
        print("- Minimizing CG energy.")
        t_cg_opt = time.time()
        xyz_opt, opt_data = cg_optimization.optimize(
            xyz=xyz_sel,
            seq=mod_seq,
            step=args.step_cg_opt,
            n_steps=args.n_steps_cg_opt)
        t_cg_opt = time.time() - t_cg_opt
        csv_data["cg_e_ini"] = opt_data[0]
        csv_data["cg_e_end"] = opt_data[1]
        csv_data["cg_rmsd"] = opt_data[2]
        ensemble_data["cg_e_ini"] = float(np.mean(csv_data["cg_e_ini"]))
        ensemble_data["cg_e_end"] = float(np.mean(csv_data["cg_e_end"]))
        ensemble_data["cg_rmsd"] = float(np.mean(csv_data["cg_rmsd"]))
        ensemble_data["t_cg_opt"] = t_cg_opt
        print("- Minimization results:"
            f" e_ini={ensemble_data['cg_e_ini']:.2f},"
            f" e_end={ensemble_data['cg_e_end']:.2f},"
            f" dist_ini={ensemble_data['cg_rmsd']:.2f} nm,"
            f" time={ensemble_data['t_cg_opt']:.2f} s")
    else:
        xyz_opt = xyz_sel

    # Save input files for cg2aa.
    cg_top_fp = os.path.join(ensemble_dp, "top.cg.pdb")
    aa_top_fp = os.path.join(ensemble_dp, "top.aa.pdb")
    cg_traj_fp = os.path.join(ensemble_dp, "traj.cg.dcd")
    aa_traj_fp = os.path.join(ensemble_dp, "traj.aa.dcd")

    print("- Saving input files for cg2aa.")
    cg_traj = mdtraj.Trajectory(
        xyz=xyz_opt.numpy(),
        # Save the topology PDB file with the original sequence.
        topology=get_ca_topology(raw_seq))
    cg_traj.center_coordinates()
    cg_traj[0].save(cg_top_fp)
    cg_traj.save(cg_traj_fp)

    # Use the cg2aa network to convert the CG ensemble to all atom.
    print("- Converting to all atom via cg2aa.")
    t_cg2all = time.time()
    cg2all_traj_args = cg2all_default_cmd + [
                       "-p", cg_top_fp,
                       "-d", cg_traj_fp,
                       "-o", aa_traj_fp,
                       "--cg", "CalphaBasedModel",
                       "--device", "cpu"]
    run_cg2all(cg2all_traj_args)
    t_cg2all = time.time() - t_cg2all
    ensemble_data["t_cg2all"] = t_cg2all
    print(f"- Converted. It took {t_cg2all:.2f} s.")

    # Use cg2aa to save an all atom topology file.
    cg2all_top_args = cg2all_default_cmd + [
                       "-p", cg_top_fp,
                       "-o", aa_top_fp,
                       "--cg", "CalphaBasedModel",
                       "--device", "cpu"]
    run_cg2all(cg2all_top_args)

    # Save all atom PDB files for each conformation in the ensemble.
    pdb_dp = os.path.join(ensemble_dp, "pdb")
    os.mkdir(pdb_dp)
    aa_traj = mdtraj.load(aa_traj_fp, top=aa_top_fp)

    print(f"- Writing ouput PDB files at {pdb_dp}.")
    for j in range(len(aa_traj)):
        aa_traj[j].save(os.path.join(pdb_dp, f"structure.{j}.pdb"))
    
    # Optimize the conformations by minimizing an all atom energy function.
    if not args.skip_aa_opt:
        print("- Minimizing all atom energy using OpenMM.")
        aa_opt_data = []
        t_aa_opt = 0
        for j in range(len(aa_traj)):
            # Optimize.
            t_aa_opt_j = time.time()
            aa_opt_data_j = aa_optimization.optimize(
                pdb_fp=os.path.join(pdb_dp, f"structure.{j}.pdb"),
                max_iterations=args.n_steps_aa_opt,
                device=args.openmm_device)
            t_aa_opt += time.time() - t_aa_opt_j
            # Distances to initial atom positions.
            rmsd_ini_j = np.sqrt(
                np.mean(
                    np.sum(
                        np.square(
                            aa_opt_data_j["pos"] - aa_opt_data_j["posinit"]
                        ),
                        axis=1),
                )
            )
            aa_opt_data_j.pop("posinit")
            aa_opt_data_j.pop("pos")
            aa_opt_data_j["rmsd_ini"] = rmsd_ini_j
            aa_opt_data.append(aa_opt_data_j)
        # Collect some results of the optimization.
        csv_data["aa_e_ini"] = [o["einit"] for o in aa_opt_data]
        csv_data["aa_e_end"] = [o["efinal"] for o in aa_opt_data]
        csv_data["aa_rmsd"] = [o["rmsd_ini"]*0.1 for o in aa_opt_data]
        ensemble_data["aa_e_ini"] = float(np.mean(csv_data["aa_e_ini"]))
        ensemble_data["aa_e_end"] = float(np.mean(csv_data["aa_e_end"]))
        ensemble_data["aa_med_e_ini"] = float(np.median(csv_data["aa_e_ini"]))
        ensemble_data["aa_med_e_end"] = float(np.median(csv_data["aa_e_end"]))
        ensemble_data["aa_rmsd"] = float(np.mean(csv_data["aa_rmsd"]))
        ensemble_data["t_aa_opt"] = t_aa_opt
        print("- Minimization results:"
              f" e_ini={ensemble_data['aa_e_ini']:.2e},"
              f" e_end={ensemble_data['aa_e_end']:.2e},"
              f" med_e_ini={ensemble_data['aa_med_e_ini']:.2e},"
              f" med_e_end={ensemble_data['aa_med_e_end']:.2e},"
              f" rmsd_ini={ensemble_data['aa_rmsd']:.2f} nm,"
              f" time={ensemble_data['t_aa_opt']:.2f} s")
        
        # Rewrite all PDB files with Biopython, since mdtraj files do not seem
        # to be compatible with some mkdssp versions.
        pdb_io = PDBIO()
        pdb_parser = PDBParser(QUIET=True)
        for pdb_path in pathlib.Path(pdb_dp).glob("./*.pdb"):
            pdb_io.set_structure(pdb_parser.get_structure(pdb_path.name,
                                                          str(pdb_path)))
            pdb_io.save(str(pdb_path))
            # Add an header with the current date.
            with open(pdb_path, "r") as i_fh:
                new_lines = i_fh.readlines()
                current_date = datetime.now().strftime("%d-%b-%y")
                header_line = "{:<50}{}\n".format("HEADER",
                                                  current_date.upper())
                new_lines.insert(0, header_line)
            with open(pdb_path, "w") as o_fh:
                o_fh.writelines(new_lines)

        # Output file with some properties of the ensemble.
        ensemble_data_fp = os.path.join(ensemble_dp, "stats.json")
        with open(ensemble_data_fp, "w") as o_fh:
            json.dump(ensemble_data, o_fh, indent=2)

        csv_fp = os.path.join(ensemble_dp, "scores.csv")
        with open(csv_fp, "w", newline="") as c_fh:
            writer = csv.writer(c_fh)
            # Write the header row with column names.
            writer.writerow(csv_cols)
            # Write the data.
            for row in range(0, args.n_samples):
                row_data = [round(csv_data[col][row], 4) for col in csv_cols]
                writer.writerow(row_data)

        seq_file = os.path.join(ensemble_dp, "seq.fasta")
        with open(seq_file, "w") as o_fh:
            o_fh.write(f">raw_seq\n{raw_seq}\n\n")
            o_fh.write(f">mod_seq\n{mod_seq}\n")
