import gromacs  # GromacsWrapper for running GROMACS commands
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import os
import urllib.request

# Set up working directory
working_dir = "gromacs_test"
os.makedirs(working_dir, exist_ok=True)
os.chdir(working_dir)

# Step 1: Download a sample PDB file (ubiquitin, 1UBQ)
pdb_url = "https://files.rcsb.org/download/1UBQ.pdb"
pdb_file = "1ubq.pdb"
if not os.path.exists(pdb_file):
    urllib.request.urlretrieve(pdb_url, pdb_file)

# Step 2: Write a simple energy minimization .mdp file
minim_mdp = """
integrator  = steep        ; Steepest descent algorithm
emtol       = 1000.0       ; Stop when energy change < 1000 kJ/mol/nm
nsteps      = 5000         ; Max steps
nstenergy   = 100          ; Energy output frequency
energygrps  = Protein      ; Energy group
"""
with open("minim.mdp", "w") as f:
    f.write(minim_mdp)

# Step 3: Run GROMACS commands
print("Generating topology and structure...")
gromacs.pdb2gmx(f=pdb_file, o="processed.gro", p="topol.top", ff="amber99sb-ildn", water="tip3p")

print("Adding solvent...")
gromacs.editconf(f="processed.gro", o="boxed.gro", bt="cubic", d=1.0)
gromacs.solvate(cp="boxed.gro", cs="spc216.gro", p="topol.top", o="solvated.gro")

print("Preparing energy minimization...")
gromacs.grompp(f="minim.mdp", c="solvated.gro", p="topol.top", o="minim.tpr")
gromacs.mdrun(s="minim.tpr", c="minim.gro", v=True)

# Step 4: Analyze with MDAnalysis
print("Analyzing output with MDAnalysis...")
universe = mda.Universe("minim.gro")
protein = universe.select_atoms("protein")
coords = protein.positions
rmsd = rms.RMSD(protein, protein).run()  # Compare to itself (single frame)

print(f"Protein atoms: {len(protein)}")
print(f"Final coordinates (first 5 atoms): {coords[:5]}")
print(f"RMSD (dummy, single frame): {rmsd.rmsd[0]}")

print("Simulation and analysis completed!")