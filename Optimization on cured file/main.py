import subprocess
import sys

def run_pipeline():
    print("========================================")
    print("  STARTING  ")
    print("========================================\n")

    try:
        # Phase 1: The Target Generator (LLM)
        print(">>> Generating Targets via LLM...")
        subprocess.run([sys.executable, "targets.py"], check=True)

        # Phase 2: The SBML Modifier 
        # print(">>> Modifying SBML file...")
        # subprocess.run([sys.executable, "modifier.py"], check=True)

        # Phase 3: The Optimizer
        print(">>> Running the Optimizer...")
        subprocess.run([sys.executable, "optimizer.py"], check=True)

        print("========================================")
        print("  FINISHED SUCCESSFULLY  ")
        print("========================================")

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Program crashed during execution. {e}")

if __name__ == "__main__":
    run_pipeline()