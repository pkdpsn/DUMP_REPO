import os
import json
import subprocess
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from prompts import initial_prompt  # Assuming initial_prompt is defined in prompts.py

# Initialize Ollama with DeepSeek 1.5B
os.environ["OLLAMA_NO_CUDA"] = "0"  # Ensure CUDA is used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU (GPU 0)
llm = Ollama(model="deepseek-r1:1.5b")  # Adjust if your model name differs

# Create the chain
chain = LLMChain(llm=llm, prompt=initial_prompt)

# Function to parse the LLM response
def parse_llm_response(response):
    """Parse the LLM response into thought, code, and final answer."""
    thought = ""
    code = ""
    final_answer = ""
    
    lines = response.splitlines()
    current_section = None
    
    for line in lines:
        line = line.strip()
        if line.startswith("**Thought:**"):
            current_section = "thought"
            thought = line[len("**Thought:**"):].strip()
        elif line.startswith("**Code:**"):
            current_section = "code"
        elif line.startswith("**Final Answer:**"):
            current_section = "final_answer"
        elif current_section == "thought" and line:
            thought += " " + line
        elif current_section == "code" and "```python" in line:
            continue
        elif current_section == "code" and "```" in line and not "```python" in line:
            current_section = None
        elif current_section == "code" and line:
            code += line + "\n"
        elif current_section == "final_answer" and "```output" in line:
            continue
        elif current_section == "final_answer" and "```" in line and not "```output" in line:
            current_section = None
        elif current_section == "final_answer" and line:
            final_answer += line + "\n"
    
    # Extract JSON from code and final_answer
    try:
        if code:
            code_json = json.loads(code.strip())
            code = code_json.get("code", "")
        if final_answer:
            answer_json = json.loads(final_answer.strip())
            final_answer = answer_json.get("output", "")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in response: {e}")
        return thought, None, None
    
    return thought, code, final_answer

# Function to execute simulation code
def execute_simulation_code(code_str):
    """Execute the provided code and save files to the current directory."""
    # Save the code to a file in the current directory
    script_file = "run_sim.py"
    with open(script_file, "w") as f:
        f.write(code_str)
    
    try:
        # Run as subprocess in the current directory
        result = subprocess.run(["python", script_file], capture_output=True, text=True, check=True)
        output = result.stdout
        # Files like .gro, .xtc, etc., will remain in the current directory
        return True, output
    except subprocess.CalledProcessError as e:
        error = e.stderr
        return False, error
    except Exception as e:
        return False, str(e)

# Main agent function
def md_simulation_agent(user_prompt, context="", max_attempts=10):
    """Run the LLM agent with error feedback and iteration."""
    current_context = context
    current_error = ""
    
    for attempt in range(max_attempts):
        print(f"\n=== Attempt {attempt + 1}/{max_attempts} ===")
        
        # Generate response from LLM
        response = chain.run(input=user_prompt, context=current_context, error=current_error)
        print("Raw LLM Response:")
        print(response)
        
        # Parse the response
        thought, code, final_answer = parse_llm_response(response)
        
        print("Thought:", thought)
        if code:
            print("Generated Code:")
            print(code)
        if final_answer:
            print("Final Answer:", final_answer)
        
        # Handle the response
        if final_answer:
            print("Simulation completed with final answer!")
            return final_answer
        elif code:
            print("Executing Code...")
            success, result = execute_simulation_code(code)
            
            if success:
                print("Output:", result)
                print("Simulation completed successfully!")
                # All files (e.g., run_sim.py, md.gro, traj.xtc) remain in the current directory
                return result
            else:
                print("Error:", result)
                current_error = result
                current_context += f"\nPrevious attempt failed with error: {current_error}"
                print("Feeding error back to LLM for correction...")
        else:
            print("No valid code or final answer provided.")
            return None
    
    print(f"Failed to generate working code or answer after {max_attempts} attempts.")
    return None

# Test the agent
if __name__ == "__main__":
    # Example prompt
    user_prompt = "Simulate protein ABCDE at 300K until RMSD < 0.2 nm"
    context = "No prior runs."
    
    # Run the agent
    result = md_simulation_agent(user_prompt, context)