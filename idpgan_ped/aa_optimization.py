# Adapted from:
#     https://github.com/deepmind/alphafold/blob/main/alphafold/relax/amber_minimize.py
# Original version license:
#
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Score and minimize the amber99sb energy of all atom models using OpenMM.
"""

import os
import openmm
import openmm.app
import openmm.unit

ENERGY = openmm.unit.kilocalories_per_mole
LENGTH = openmm.unit.angstroms


def optimize(
    pdb_fp: str,
    max_iterations: int = 50,
    tolerance: float = 2.39,
    # stiffness: float = 10.0,
    device: str = "cpu",
    overwrite: bool = False):
    """Minimize energy via openmm."""

    if not pdb_fp.endswith(".pdb"):
        raise OSError(f"Can only minimize PDB files, provided {pdb_fp}")

    with open(pdb_fp, "r") as i_fh:
        pdb = openmm.app.PDBFile(i_fh)

    force_field = openmm.app.ForceField("amber99sb.xml")
    constraints = openmm.app.HBonds
    system = force_field.createSystem(
        pdb.topology, constraints=constraints)

    tolerance = tolerance * ENERGY
    # stiffness = stiffness * ENERGY / (LENGTH**2)

    integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
    platform = openmm.Platform.getPlatformByName(device.upper())
    simulation = openmm.app.Simulation(
        pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)

    ret = {}
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    ret["einit"] = state.getPotentialEnergy().value_in_unit(ENERGY)
    ret["posinit"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
    simulation.minimizeEnergy(maxIterations=max_iterations,
                              tolerance=tolerance)
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    ret["efinal"] = state.getPotentialEnergy().value_in_unit(ENERGY)
    ret["pos"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)

    if overwrite:
        out_fp = pdb_fp
    else:
        out_fp = os.path.join(
            os.path.dirname(pdb_fp),
            os.path.basename(pdb_fp).rstrip(".pdb") + ".opt.pdb")
                              
    with open(out_fp, "w") as o_fh:
        openmm.app.PDBFile.writeFile(simulation.topology,
                                     state.getPositions(),
                                     o_fh)
    return ret