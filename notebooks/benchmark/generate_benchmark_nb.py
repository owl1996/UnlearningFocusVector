import nbformat as nbf
import os

# Create a new Jupyter notebook
nb = nbf.v4.new_notebook()

# Define the cells
cells = []

# Cell 1: Title and Markdown Description
cells.append(nbf.v4.new_markdown_cell("# Unlearning Benchmarking Pipeline\nThis notebook automates testing multiple unlearning methods via MLFlow tracking. It verifies the existence of both the base model and its corresponding ideal model (based on `num_indexes_to_replace` and `class_to_replace`) before triggering the unlearn script."))

# Cell 2: Imports and Setup
code_setup = """
import os
import subprocess
import glob

# Ensure we're executing commands from the project root
# If the notebook is inside 'notebooks/', we adjust the Python working directory
current_dir = os.getcwd()
if current_dir.endswith('notebooks'):
    project_root = os.path.dirname(current_dir)
    os.chdir(project_root)
else:
    project_root = current_dir

print(f"Project root set to: {project_root}")
"""
cells.append(nbf.v4.new_code_cell(code_setup))

# Cell 3: Configuration block
cells.append(nbf.v4.new_markdown_cell("## Configuration\nModify these parameters to control the benchmark. The script will automatically link the chosen `seed`, `arch`, `dataset`, and retention parameters to the specific ideal model checkpoint requirements."))

code_config = """
# Unlearning methods to benchmark sequentially
methods_to_test = [
    "LDA",
    "fisher",
    "OS_unlearn",
]

# Shared Parameters for the Unlearning Script
dataset = "cifar10"
arch = "vgg16_bn"
seed = 1

# Forgetting setup (these define which 'ideal' model is targeted)
class_to_replace = -1
num_indexes_to_replace = 4500

# Training Hyperparams
batch_size = 128
unlearn_epochs = 1
unlearn_lr = 0.0001
epochs = 100
beta = 0.95

# The directory where datasets and model checkpoints reside
data_dir = "./data"
save_dir = f"./results/{dataset}"
"""
cells.append(nbf.v4.new_code_cell(code_config))

# Cell 4: Model Discovery Logic
cells.append(nbf.v4.new_markdown_cell("## Model Discovery\nVerifying that the required base model and its corresponding `ideal` model are present in the `results/` folder."))

code_discovery = """
base_model_file = f"{dataset}_{arch}_{seed}model.pth.tar"
ideal_model_file = f"ideal_{num_indexes_to_replace}_{class_to_replace}_{dataset}_{arch}_{seed}model.pth.tar"

base_path = os.path.join(save_dir, base_model_file)
ideal_path = os.path.join(save_dir, ideal_model_file)

if not os.path.exists(base_path):
    raise FileNotFoundError(f"Base model not found: {base_path}")
    
if not os.path.exists(ideal_path):
    raise FileNotFoundError(f"Corresponding Ideal model not found: {ideal_path}")

print(f"Found Base Model: {base_model_file}")
print(f"Found Ideal Model: {ideal_model_file}")
print(f"Ready to benchmark {len(methods_to_test)} methods.")
"""
cells.append(nbf.v4.new_code_cell(code_discovery))

# Cell 5: Execution Loop
cells.append(nbf.v4.new_markdown_cell("## Execution Loop\nRunning `mlflow_forget.py` for each defined method using `subprocess`. Live output will be streamed below."))

code_execution = """
for method in methods_to_test:
    print(f"\\n{'='*50}\\nStarting benchmark for method: {method}\\n{'='*50}")
    
    command = [
        "python", "mlflow_forget.py",
        "--save_dir", save_dir,
        "--mask", base_path,
        "--unlearn", method,
        "--unlearn_epochs", str(unlearn_epochs),
        "--unlearn_lr", str(unlearn_lr),
        "--data", data_dir,
        "--dataset", dataset,
        "--seed", str(seed),
        "--arch", arch,
        "--epochs", str(epochs),
        "--num_indexes_to_replace", str(num_indexes_to_replace),
        "--class_to_replace", str(class_to_replace),
        "--beta", str(beta),
        "--batch_size", str(batch_size)
    ]
    
    print("Command Executed:")
    print(" ".join(command))
    print("-" * 50)
    
    try:
        # Run process and stream output
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=project_root
        )
        
        # Stream logs line by line
        for line in process.stdout:
            print(line, end="")
            
        process.wait()
        
        if process.returncode == 0:
            print(f"\\n✅ Successfully completed unlearning for {method}.")
        else:
            print(f"\\n❌ Error executing {method}. Exit code: {process.returncode}")
            
    except Exception as e:
        print(f"\\n⚠️ Exception occurred while running {method}: {e}")
"""
cells.append(nbf.v4.new_code_cell(code_execution))


# Write notebook
nb.cells = cells
output_file = 'notebooks/generate_benchmark_nb.py'

# Output the notebook into notebooks dir relative to project root
notebook_out_path = 'notebooks/benchmark.ipynb'
os.makedirs('notebooks', exist_ok=True)

with open(notebook_out_path, 'w') as f:
    nbf.write(nb, f)

print(f"Benchmarking notebook generated at: {notebook_out_path}")
