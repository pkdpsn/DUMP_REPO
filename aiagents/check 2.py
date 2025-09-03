# import gromacs
# import MDAnalysis as mda
# from MDAnalysis.analysis import rms
# import os
# import urllib.request

# # Set up working directory
# working_dir = "simple_sim"
# os.makedirs(working_dir, exist_ok=True)
# os.chdir(working_dir)

# # Download PDB (ubiquitin, 1UBQ)
# pdb_url = "https://files.rcsb.org/download/1UBQ.pdb"
# pdb_file = "1ubq.pdb"
# if not os.path.exists(pdb_file):
#     urllib.request.urlretrieve(pdb_url, pdb_file)

# # Energy minimization .mdp
# minim_mdp = """
# integrator  = steep
# emtol       = 10.0    ; Converge to 100 kJ/mol/nm
# nsteps      = 5000
# nstenergy   = 100
# """
# with open("minim.mdp", "w") as f:
#     f.write(minim_mdp)

# # Simple MD .mdp (NVT, 50 ps)
# md_mdp = """
# integrator  = md
# nsteps      = 50000    ; 50 ps at 1 fs timestep
# dt          = 0.0005    ; 1 fs for stability
# nstxout     = 100      ; Output every 100 steps
# nstenergy   = 100
# tcoupl      = v-rescale
# tc-grps     = System   ; Couple entire system
# tau-t       = 1.0
# ref_t       = 300
# gen_vel     = yes
# constraints = h-bonds
# """
# with open("md.mdp", "w") as f:
#     f.write(md_mdp)

# # GROMACS workflow
# print("Preparing system...")
# gromacs.pdb2gmx(f=pdb_file, o="processed.gro", p="topol.top", ff="amber99sb-ildn", water="tip3p")
# gromacs.editconf(f="processed.gro", o="boxed.gro", bt="cubic", d=1.0)
# gromacs.solvate(cp="boxed.gro", cs="spc216.gro", p="topol.top", o="solvated.gro")

# print("Running energy minimization...")
# gromacs.grompp(f="minim.mdp", c="solvated.gro", p="topol.top", o="minim.tpr")
# gromacs.mdrun(s="minim.tpr", c="minim.gro", v=True)

# print("Running MD simulation...")
# gromacs.grompp(f="md.mdp", c="minim.gro", p="topol.top", o="md.tpr")
# gromacs.mdrun(s="md.tpr", c="md.gro", x="traj.xtc", v=True)

# # Analyze with MDAnalysis
# print("Analyzing results...")
# u = mda.Universe("minim.gro", "traj.xtc")
# protein = u.select_atoms("protein")
# R = rms.RMSD(protein, reference=protein, ref_frame=0).run()
# final_rmsd = R.rmsd[-1, 2]  # Last frame RMSD in nm

# print(f"Protein atoms: {len(protein)}")
# print(f"Final RMSD: {final_rmsd:.3f} nm")
# print("Simulation completed!")

# from langchain_community.llms import Ollama
# llm = Ollama(model="deepseek-r1:1.5b")
# response = llm("What is LangChain?")
# print(response)


import cupy as cp

# Create random matrices
A = cp.random.randn(10000, 10000)
B = cp.random.randn(10000, 10000)

# Perform matrix multiplication on GPU
C = cp.dot(A, B)

print("Matrix multiplication using CuPy completed!")
