"""
Mega-Heracross: Full Pipeline Runner
Part A -> Part B -> Part C
"""
import subprocess
import os
import sys

# ITEM 6: prevent cp1252 UnicodeEncodeError on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

def main():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    # Suppress HuggingFace Hub / Transformers noise from Part A subprocess
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["HF_HUB_VERBOSITY"] = "error"
    env["HUGGINGFACE_HUB_VERBOSITY"] = "error"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["TRANSFORMERS_VERBOSITY"] = "error"

    print("[A] Running Vision Engine...")
    subprocess.run([sys.executable, "part_a_vision/run_part_a.py", "--occlusion", "all"], env=env, check=True)

    print("\n[B] Running Skeletonization Engine...")
    subprocess.run([sys.executable, "part_b_skeleton/run.py"], env=env, check=True)

    print("\n[C] Running Resilience Engine...")
    subprocess.run([sys.executable, "part_c_resilience/main.py"], env=env, check=True)

    print("\nPipeline complete. Check outputs/.")

if __name__ == "__main__":
    main()
