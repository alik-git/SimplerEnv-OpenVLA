import os
import json
import pandas as pd

# Define the root directory where logs are saved
logging_root = "./results_simple_full_eval"

# Define specific version paths for each policy to handle subpath repetition
POLICY_PATHS = {
    "openvla": "openvla/openvla-7b/openvla-7b",
    "rt1": "rt1/rt_1_x_tf_trained_for_002272480_step"
}

# Initialize a list to store each task's result summary
results = []

# Loop over each task directory in the logging root
for task_dir in os.listdir(logging_root):
    task_path = os.path.join(logging_root, task_dir)
    
    if not os.path.isdir(task_path):
        continue  # Skip if not a directory

    # Loop over each policy type in POLICY_PATHS
    for policy, subpath in POLICY_PATHS.items():
        version_path = os.path.join(task_path, subpath)
        
        if not os.path.isdir(version_path):
            print(f"Warning: Version path missing for task '{task_dir}' with policy '{policy}'.")
            continue  # Skip if expected version path is missing

        # Initialize counters for successful and total trajectories
        success_count = 0
        total_count = 0
        
        # Path to the evaluation log within the version directory
        log_file = os.path.join(version_path, "evaluation_log.json")
        
        # Check if log file exists
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    log_data = json.load(f)
                
                # Process each episode in the log data
                for episode in log_data.get("episodes", []):
                    success = episode.get("success")
                    if success is not None:
                        total_count += 1
                        if success:
                            success_count += 1
                    else:
                        print(f"Warning: 'success' key missing in an episode in {log_file}")

            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading {log_file}: {e}")
        else:
            print(f"Log file not found for {task_dir} - {policy}")

        # Calculate success rate if total_count is greater than zero
        success_rate = success_count / total_count if total_count > 0 else None
        
        # Append to results, marking incomplete tasks if no valid log files were found
        results.append({
            "Task": task_dir,
            "Policy": policy,
            "Success Rate": success_rate,
            "Total Trajectories": total_count,
            "Successful Trajectories": success_count,
            "Incomplete": total_count == 0  # Mark as incomplete if no logs were valid
        })

# Create a DataFrame from the results
df_results = pd.DataFrame(results)

# Save the full results to a CSV
df_results.to_csv("evaluation_summary.csv", index=False)
print(df_results)

# Create summary for Google Robot and WidowX tasks
summary_data = []

# Calculate average success rates and total trajectories for Google Robot and WidowX tasks
for policy in ["rt1", "openvla"]:
    for task_type in ["google_robot", "widowx"]:
        # Filter by task type and policy
        filtered_df = df_results[(df_results["Task"].str.startswith(task_type)) & (df_results["Policy"] == policy)]
        
        # Calculate metrics
        avg_success_rate = filtered_df["Success Rate"].mean()  # Average success rate
        valid_trajectories = filtered_df["Total Trajectories"].sum()  # Sum of valid trajectories
        
        # Append summary data
        summary_data.append({
            "Task Type": "Google Robot" if task_type == "google_robot" else "WidowX",
            "Policy": policy,
            "Average Success Rate": avg_success_rate,
            "Total Valid Trajectories": valid_trajectories
        })

# Create DataFrame for summary results
df_summary = pd.DataFrame(summary_data)

# Save and display the summary DataFrame
df_summary.to_csv("evaluation_summary_summary.csv", index=False)
print(df_summary)
