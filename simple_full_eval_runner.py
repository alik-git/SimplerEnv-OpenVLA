import subprocess

from simpler_env import ENVIRONMENTS


# Define the policies with corresponding checkpoint paths
policies = [
    {"name": "rt1", "ckpt_path": "./checkpoints/rt_1_x_tf_trained_for_002272480_step"},
    {"name": "openvla/openvla-7b", "ckpt_path": None}
]

# Parameters for all evaluations
logging_root = "./results_simple_full_eval"
n_trajs = 10

# Function to run a single evaluation with error handling
def run_evaluation(policy_name, ckpt_path, task):
    command = [
        "python", "simpler_env/simple_inference_visual_matching_prepackaged_envs.py",
        "--policy", policy_name,
        "--task", task,
        "--logging-root", logging_root,
        "--n-trajs", str(n_trajs)
    ]
    
    if ckpt_path:
        command.extend(["--ckpt-path", ckpt_path])
    else:
        command.extend(["--ckpt-path", "None"])

    # Run the command with error handling
    try:
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"Successfully completed {policy_name} on {task}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {policy_name} on {task}")
        print(f"Return code: {e.returncode}")
        print("Output:", e.output)
        print("Error Output:", e.stderr)
    except Exception as e:
        print(f"Unexpected error with {policy_name} on {task}: {str(e)}")

# Loop over each task and each policy, running the evaluations
for task in ENVIRONMENTS:
    for policy in policies:
        run_evaluation(policy["name"], policy["ckpt_path"], task)
