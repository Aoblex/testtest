import subprocess
import sys
from pathlib import Path

def run_script(script_path: str) -> None:
    """Run a Python script and print its output."""
    print(f"\nRunning {script_path}...")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

def main():
    # Get all example scripts except this one
    example_dir = Path(__file__).parent
    scripts = [
        str(p) for p in example_dir.glob("run_*.py")
        if p.name != "run_all.py"
    ]
    
    # Run each script
    for script in sorted(scripts):
        run_script(script)

if __name__ == "__main__":
    main() 