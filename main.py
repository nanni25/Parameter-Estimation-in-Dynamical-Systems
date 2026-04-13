import argparse
import subprocess
import sys
import os

def run_pipeline(args):
    print("========================================")
    print("  STARTING PIPELINE ")
    print("========================================\n")

    # Setup directories and dynamic file paths
    os.makedirs(args.cured_dir, exist_ok=True)
    os.makedirs(args.targets_dir, exist_ok=True) # <-- Creates the Targets folder
    
    base_filename = os.path.basename(args.input_model)
    cured_model_path = os.path.join(args.cured_dir, f"Cured_{base_filename}")
    
    # Extract the name without extension to create a unique JSON file name
    model_name_no_ext = os.path.splitext(base_filename)[0]
    targets_path = os.path.join(args.targets_dir, f"{model_name_no_ext}_targets.json")

    try:
        # Phase 1: The Target Generator (LLM)
        print(">>> Generating Targets via LLM...")
        subprocess.run([sys.executable, "targets.py", 
                        "--input_sbml", args.input_model, 
                        "--output_json", targets_path], check=True)

        # Phase 2: The SBML Modifier 
        print(">>> Modifying SBML file...")
        subprocess.run([sys.executable, "modifier.py", 
                        "--input_sbml", args.input_model, 
                        "--output_sbml", cured_model_path], check=True)

        # Phase 3: The Optimizer
        print(">>> Running the Optimizer...")
        optimizer_cmd = [
            sys.executable, "optimizer.py",
            "--model_path", cured_model_path,
            "--targets_path", targets_path,
            "--sim_time", str(args.sim_time),
            "--sim_steps", str(args.sim_steps),
            "--pop_size", str(args.pop_size),
            "--generations", str(args.generations),
            "--learning_rate", str(args.learning_rate),
            "--sigma", str(args.sigma),
            "--patience", str(args.patience),
            "--min_delta", str(args.min_delta)
        ]
        subprocess.run(optimizer_cmd, check=True)

        print("========================================")
        print("  FINISHED SUCCESSFULLY  ")
        print("========================================")

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Program crashed during execution. {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SBML Optimization Pipeline.")
    
    # File handling
    parser.add_argument("--input_model", type=str, required=True, help="Path to the original SBML file.")
    parser.add_argument("--cured_dir", type=str, default="Cured Models", help="Directory to save cured models.")
    parser.add_argument("--targets_dir", type=str, default="Targets", help="Directory to save target JSON files.")
    
    # Optimizer Hyperparameters
    parser.add_argument("--sim_time", type=float, default=1000.0, help="Total simulation time.")
    parser.add_argument("--sim_steps", type=int, default=1000, help="Number of simulation steps.")
    parser.add_argument("--pop_size", type=int, default=100, help="Population size for the optimizer.")
    parser.add_argument("--generations", type=int, default=500, help="Number of optimization generations.")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--sigma", type=float, default=0.05, help="Exploration variance (sigma).")    
    parser.add_argument("--patience", type=int, default=50, help="Generations to wait for improvement before stopping.")
    parser.add_argument("--min_delta", type=float, default=1e-5, help="Minimum decrease in loss to count as improvement.")
    
    args = parser.parse_args()
    run_pipeline(args)