# Instructions
Those are the instruction for deriving the parameters of the Ca torsion angle potential used in this package. Use the scripts in this directory if you want to derive them again.

The parameters were already derived and are stored in the `idpgan_ped/torsion_potential.py` module. The following instructions can be used to replicate the analysis for deriving them.
1. Download the human AF2DB proteome: https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v4.tar
2. Run the `alphafold_disorder.py` script from the `AlphaFold-disorder` repository: https://github.com/BioComputingUP/AlphaFold-disorder. Use the default options of the script. Rename the prediction ouput file as `af2_pred.tsv`, it will be used in the step below.
3. Run the following command to extract data for the torsion angle potential:
   ```bash
   torsion_potential_stats.py -i af2_pdb/ -p af2_pred.tsv -o torsion_pot.npy && torsion_potential_build.py -i torsion_pot.npy
   ```
   where `af2_pdb` is the directory where the PDB files downloaded in point 1 are. `torsion_pot.npy` is the name of the output file of the first here script.