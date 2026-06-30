"""
Mega-Heracross: Full Pipeline Runner
Part A → Part B → Part C
"""
import subprocess
import os
import sys

def main():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    print("[A] Running Vision Engine...")
    subprocess.run([sys.executable, "part_a_vision/run_part_a.py", "--occlusion", "all"], env=env, check=True)

    print("\n[B] Running Skeletonization Engine...")
    subprocess.run([sys.executable, "part_b_skeleton/run.py"], env=env, check=True)

    print("\n[C] Running Resilience Engine...")
    subprocess.run([sys.executable, "part_c_resilience/main.py"], env=env, check=True)

    print("\nPipeline complete. Check outputs/.")

if __name__ == "__main__":
    main()
