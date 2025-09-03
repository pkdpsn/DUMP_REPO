import MDAnalysis as mda
from MDAnalysis.analysis import rms
import subprocess

def run_gromacs():
    subprocess.run(['gmx', 'pdb2gmx', '-f', 'ABCDE.pdb', '-o', 'ABCDE.gro', '-p', 'topol.top', '-ff', 'amber99sb-ildn', '-water', 'tip3p'])
    with open('minim.mdp', 'w') as f:
        f.write('integrator = steep\n emtol = 1000\n nsteps = 50000')
    subprocess.run(['gmx', 'grompp', '-f', 'minim.mdp', '-c', 'ABCDE.gro', '-p', 'topol.top', '-o', 'minim.tpr'])
    subprocess.run(['gmx', 'mdrun', '-deffnm', 'minim'])
    with open('nvt.mdp', 'w') as f:
        f.write('integrator = md\n dt = 0.002\n nsteps = 50000\n tcoupl = v-rescale\n ref_t = 300 300')
    subprocess.run(['gmx', 'grompp', '-f', 'nvt.mdp', '-c', 'minim.gro', '-p', 'topol.top', '-o', 'nvt.tpr'])
    subprocess.run(['gmx', 'mdrun', '-deffnm', 'nvt'])
    with open('md.mdp', 'w') as f:
        f.write('integrator = md\n dt = 0.002\n nsteps = 500000\n tcoupl = v-rescale\n ref_t = 300 300')
    subprocess.run(['gmx', 'grompp', '-f', 'md.mdp', '-c', 'nvt.gro', '-p', 'topol.top', '-o', 'md.tpr'])
    subprocess.run(['gmx', 'mdrun', '-deffnm', 'md'])

run_gromacs()

u = mda.Universe('topol.top', 'md.xtc')
ref = mda.Universe('ABCDE.gro')
R = rms.RMSD(u, ref, select='backbone')
R.run()
rmsd = R.rmsd[:, 2][-1]
while rmsd >= 0.2:
    subprocess.run(['gmx', 'mdrun', '-deffnm', 'md', '-cpi', 'md.cpt'])
    u = mda.Universe('topol.top', 'md.xtc')
    R = rms.RMSD(u, ref, select='backbone')
    R.run()
    rmsd = R.rmsd[:, 2][-1]
print(f'Final RMSD: {rmsd} nm')