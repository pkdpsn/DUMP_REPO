import gromacs
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import os
import json

class LearningProteinAgent:
    def __init__(self, working_dir="agentic_protein"):
        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)
        os.chdir(self.working_dir)
        self.memory_file = "memory.json"
        self.memory = self.load_memory()
        self.protein_sequence = None
        self.conditions = {}
        self.target_output = {}

    def load_memory(self):
        """Load past simulation data."""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                return json.load(f)
        return {}

    def save_memory(self, request, params, result):
        """Save simulation data to memory."""
        self.memory[request] = {"params": params, "result": result}
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f)

    def parse_user_request(self, request):
        """Parse user input (simulated LLM)."""
        request = request.lower()
        if "simulate protein" in request:
            parts = request.split("simulate protein")[1].strip().split(" at ")
            self.protein_sequence = parts[0].strip().replace(" ", "")
            condition_part = parts[1] if len(parts) > 1 else "300K"
            self.conditions["temperature"] = float(condition_part.split("k")[0].strip())
            if "until rmsd <" in request:
                self.target_output["rmsd"] = float(request.split("rmsd <")[1].split("nm")[0].strip())
        print(f"Parsed: Sequence={self.protein_sequence}, Conditions={self.conditions}, Target={self.target_output}")

    def generate_pdb(self):
        """Generate a dummy PDB."""
        with open("input.pdb", "w") as f:
            f.write("ATOM      1  N   ALA A   1      0.000   0.000   0.000  1.00  0.00\n")
            f.write("ATOM      2  C   ALA A   1      1.500   0.000   0.000  1.00  0.00\n")
            f.write("TER\nEND\n")
        print("Generated input.pdb")

    def write_md_mdp(self, nsteps=50000, dt=0.001):
        """Generate MD .mdp with learned parameters."""
        md_mdp = f"""
integrator  = md
nsteps      = {nsteps}
dt          = {dt}
nstxout     = 100
nstenergy   = 100
tcoupl      = v-rescale
tc-grps     = System
tau-t       = 1.0
ref_t       = {self.conditions.get("temperature", 300)}
gen_vel     = yes
constraints = h-bonds
"""
        with open("md.mdp", "w") as f:
            f.write(md_mdp)
        return {"nsteps": nsteps, "dt": dt}

    def run_simulation(self, params):
        """Run GROMACS simulation with given parameters."""
        print("Preparing system...")
        gromacs.pdb2gmx(f="input.pdb", o="processed.gro", p="topol.top", ff="amber99sb-ildn", water="tip3p", ignh=True)
        gromacs.editconf(f="processed.gro", o="boxed.gro", bt="cubic", d=1.0)
        gromacs.solvate(cp="boxed.gro", cs="spc216.gro", p="topol.top", o="solvated.gro")
        
        print("Minimizing energy...")
        with open("minim.mdp", "w") as f:
            f.write("integrator = steep\nemtol = 100.0\nnsteps = 5000\nnstenergy = 100\n")
        gromacs.grompp(f="minim.mdp", c="solvated.gro", p="topol.top", o="minim.tpr")
        gromacs.mdrun(s="minim.tpr", c="minim.gro", v=True)
        
        print("Running MD...")
        self.write_md_mdp(**params)
        gromacs.grompp(f="md.mdp", c="minim.gro", p="topol.top", o="md.tpr")
        gromacs.mdrun(s="md.tpr", c="md.gro", x="traj.xtc", v=True)

    def analyze_output(self):
        """Analyze simulation output."""
        u = mda.Universe("minim.gro", "traj.xtc")
        protein = u.select_atoms("protein")
        R = rms.RMSD(protein, reference=protein, ref_frame=0).run()
        final_rmsd = R.rmsd[-1, 2]
        print(f"Final RMSD: {final_rmsd:.3f} nm")
        return {"rmsd": final_rmsd}

    def learn_parameters(self, request):
        """Decide parameters based on memory or defaults."""
        if request in self.memory:
            params = self.memory[request]["params"]
            print(f"Using learned parameters: {params}")
            return params
        return {"nsteps": 50000, "dt": 0.001}  # Default

    def adjust_parameters(self, params, result):
        """Adjust parameters based on results."""
        if "rmsd" in self.target_output and result["rmsd"] > self.target_output["rmsd"]:
            if params["dt"] > 0.0005:
                params["dt"] -= 0.0005
                print(f"Adjusted dt to {params['dt']}")
            else:
                params["nsteps"] += 50000
                print(f"Adjusted nsteps to {params['nsteps']}")
            return params, True
        return params, False

    def execute(self, request):
        """Main agent loop with learning."""
        self.parse_user_request(request)
        self.generate_pdb()
        max_attempts = 3
        params = self.learn_parameters(request)
        
        for attempt in range(max_attempts):
            print(f"Attempt {attempt + 1}/{max_attempts}")
            self.run_simulation(params)
            result = self.analyze_output()
            self.save_memory(request, params, result)
            if "rmsd" in self.target_output and result["rmsd"] <= self.target_output["rmsd"]:
                print("Target conditions met!")
                break
            params, should_continue = self.adjust_parameters(params, result)
            if not should_continue:
                break
        print("Agent task completed!")

# Example usage
if __name__ == "__main__":
    agent = LearningProteinAgent()
    request = "Simulate protein ABCDE at 300K until RMSD < 0.2 nm"
    agent.execute(request)