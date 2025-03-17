import os
import subprocess
import itertools

def main():
    # Define parameter values to test
    # window_sizes = [5, 10, 15]
    # time_shifts = [-1, 0, 1]
    window_sizes = [15]
    time_shifts = [0]
    
    # Define algorithms to test - using a smaller set for faster tuning
    algorithms = "ddpg,a2c,ppo,td3"  # You can change this to include more algorithms if needed
    
    # Base command elements that will be the same for all runs
    base_cmd = [
        "python", "train_rl_agent.py",
        "--algorithms", algorithms,
        "--normalize-values",
        "--sort-results",
        # Add any other fixed parameters here
    ]
    
    print("Starting parameter tuning...")
    
    # Create all combinations of parameters
    param_combinations = list(itertools.product(window_sizes, time_shifts))
    total_combinations = len(param_combinations)
    
    # Run training for each parameter combination
    for i, (window_size, time_shift) in enumerate(param_combinations):
        print(f"\n[{i+1}/{total_combinations}] Testing window_size={window_size}, time_shift={time_shift}")
        
        # Create directory path based on parameters
        save_dir = f"saves/window={window_size}_shift={time_shift}"
        agent_dir = f"{save_dir}/agents"
        plot_dir = f"{save_dir}/plots"
        data_dir = f"{save_dir}/data"
        
        # Create command with these specific parameters
        cmd = base_cmd + [
            "--save-dir", save_dir,
            "--agent-dir", agent_dir,
            "--plot-dir", plot_dir,
            "--data-dir", data_dir,
            "--window-size", str(window_size),
            "--time-shift", str(time_shift)
        ]
        
        # Ensure directories exist
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(agent_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        print(f"Running: {' '.join(cmd)}")
        try:
            # Run the process and capture output
            process = subprocess.run(cmd, check=True, text=True)
            print(f"Completed window_size={window_size}, time_shift={time_shift} successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error running window_size={window_size}, time_shift={time_shift}: {e}")
    
    print("\nParameter tuning completed!")
    
    # Generate a summary report
    generate_summary(window_sizes, time_shifts)

def generate_summary(window_sizes, time_shifts):
    """Generate a summary report of all the parameter combinations"""
    print("\nGenerating summary report...")
    
    with open("parameter_tuning_summary.txt", "w") as f:
        f.write("Parameter Tuning Summary\n")
        f.write("=======================\n\n")
        
        for window_size, time_shift in itertools.product(window_sizes, time_shifts):
            save_dir = f"saves/window={window_size}_shift={time_shift}"
            f.write(f"Window Size: {window_size}, Time Shift: {time_shift}\n")
            
            # Check if the directory exists and has results
            if os.path.exists(save_dir):
                f.write(f"  Results directory: {save_dir}\n")
                
                # Check for performance table
                performance_table = os.path.join(save_dir, "plots", "performance_table_*.png")
                if any(os.path.exists(os.path.join(save_dir, "plots", f)) for f in os.listdir(os.path.join(save_dir, "plots")) if f.startswith("performance_table_")):
                    f.write("  Performance table: Available\n")
                else:
                    f.write("  Performance table: Not found\n")
            else:
                f.write("  No results found\n")
            
            f.write("\n")
    
    print(f"Summary report generated: parameter_tuning_summary.txt")

if __name__ == "__main__":
    main() 