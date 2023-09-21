import mdtraj


aa_one_to_three_dict = {
    "G": "GLY", "A": "ALA", "L": "LEU", "I": "ILE", "R": "ARG", "K": "LYS",
    "M": "MET", "C": "CYS", "Y": "TYR", "T": "THR", "P": "PRO", "S": "SER",
    "W": "TRP", "D": "ASP", "E": "GLU", "N": "ASN", "Q": "GLN", "F": "PHE",
    "H": "HIS", "V": "VAL", "X": "UNK"
}

def get_ca_topology(sequence):
    topology = mdtraj.Topology()
    chain = topology.add_chain()
    for res in sequence:
        res_obj = topology.add_residue(aa_one_to_three_dict[res], chain)
        topology.add_atom("CA", mdtraj.core.topology.elem.carbon, res_obj)
    return topology