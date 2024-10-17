"""
Simple script for real-to-sim eval using the prepackaged visual matching setup in ManiSkill2.
Example:
    cd {path_to_simpler_env_repo_root}
    python simpler_env/simple_inference_visual_matching_prepackaged_envs.py --policy rt1 \
        --ckpt-path ./checkpoints/rt_1_tf_trained_for_000400120  --task google_robot_pick_coke_can  --logging-root ./results_simple_eval/  --n-trajs 10
    python simpler_env/simple_inference_visual_matching_prepackaged_envs.py --policy octo-small \
        --ckpt-path None --task widowx_spoon_on_towel  --logging-root ./results_simple_eval/  --n-trajs 10
    python simpler_env/simple_inference_visual_matching_prepackaged_envs.py --policy openvla/openvla-7b \
        --ckpt-path None --task google_robot_move_near_v1  --logging-root ./results_simple_eval/  --n-trajs 10
"""

import argparse
import os
import json
import mediapy as media
import numpy as np
import tensorflow as tf
import simpler_env
from simpler_env import ENVIRONMENTS
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

# Function to handle both printing and logging
def log_message(log, message):
    print(message)
    log["messages"].append(message)

# Convert numpy types to native Python types
def convert_numpy_types(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# Set up argument parsing and logging
parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="rt1", choices=["rt1", "octo-base", "octo-small", "openvla/openvla-7b"])
parser.add_argument("--ckpt-path", type=str, default="./checkpoints/rt_1_x_tf_trained_for_002272480_step/") # Replace with your checkpoint!
parser.add_argument("--task", default="google_robot_pick_horizontal_coke_can", choices=ENVIRONMENTS)
parser.add_argument("--logging-root", type=str, default="./results_simple_random_eval")
parser.add_argument("--tf-memory-limit", type=int, default=3072)
parser.add_argument("--n-trajs", type=int, default=10)

args = parser.parse_args()

# Initialize the log dictionary
log = {
    "args": vars(args),
    "episodes": [],
    "messages": []
}

# Prepare logging directory
if args.policy in ["octo-base", "octo-small", "openvla/openvla-7b"]:
    if args.ckpt_path in [None, "None"] or "rt_1_x" in args.ckpt_path:
        args.ckpt_path = args.policy
if args.ckpt_path[-1] == "/":
    args.ckpt_path = args.ckpt_path[:-1]
logging_dir = os.path.join(args.logging_root, args.task, args.policy, os.path.basename(args.ckpt_path))
os.makedirs(logging_dir, exist_ok=True)

# Prevent GPU memory over-allocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
gpus = tf.config.list_physical_devices("GPU")
if len(gpus) > 0:
    # prevent a single tf process from taking up all the GPU memory
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
    )

# Build environment and policy setup
env = simpler_env.make(args.task)
policy_setup = "google_robot" if "google_robot" in args.task else "widowx_bridge"

# Initialize model
if args.policy == "rt1":
    from simpler_env.policies.rt1.rt1_model import RT1Inference
    model = RT1Inference(saved_model_path=args.ckpt_path, policy_setup=policy_setup)
elif "octo" in args.policy:
    from simpler_env.policies.octo.octo_model import OctoInference
    model = OctoInference(model_type=args.ckpt_path, policy_setup=policy_setup, init_rng=0)
elif "openvla" in args.policy:
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    model = OpenVLAInference(saved_model_path=args.ckpt_path, policy_setup=policy_setup)
else:
    raise NotImplementedError()

# Run evaluation
success_arr = []
for ep_id in range(args.n_trajs):
    obs, reset_info = env.reset()
    instruction = env.unwrapped.get_language_instruction()
    is_final_subtask = env.unwrapped.is_final_subtask()

    model.reset(instruction)
    log_message(log, f"Episode {ep_id}: {instruction}")

    episode_log = {"instruction": instruction, "timesteps": [], "episode_stats": {}, "success": False}
    image = get_image_from_maniskill2_obs_dict(env.unwrapped, obs)
    images = [image]
    predicted_terminated, success, truncated = False, False, False
    timestep = 0

    while not (predicted_terminated or truncated):
        raw_action, action = model.step(image, instruction)
        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        
        # Handle subtask advancement if required
        if predicted_terminated and not is_final_subtask:
            predicted_terminated = False
            env.unwrapped.advance_to_next_subtask()

        # Perform environment step
        obs, reward, success, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),
        )

        # Log details for current timestep
        timestep_log = {
            "timestep": timestep,
            "info": info,
            "action": {
                "world_vector": action["world_vector"].tolist(),
                "rot_axangle": action["rot_axangle"].tolist(),
                "gripper": action["gripper"].tolist(),
                "terminate_episode": action["terminate_episode"][0]
            }
        }
        episode_log["timesteps"].append(timestep_log)
        log_message(log, f"Timestep {timestep}, Info: {info}")

        # Update instruction if required for long-horizon tasks
        new_instruction = env.unwrapped.get_language_instruction()
        if new_instruction != instruction:
            instruction = new_instruction
            model.reset(instruction)
            log_message(log, f"New Instruction: {instruction}")
        is_final_subtask = env.unwrapped.is_final_subtask()

        # Update image
        image = get_image_from_maniskill2_obs_dict(env.unwrapped, obs)
        images.append(image)
        timestep += 1

    # Store episode stats
    episode_log["episode_stats"] = info.get("episode_stats", {})
    episode_log["success"] = bool(success)  # Convert success to a standard bool
    success_arr.append(success)
    log_message(log, f"Episode {ep_id} success: {success}")
    log["episodes"].append(episode_log)

    # Save video for each episode
    media.write_video(f"{logging_dir}/episode_{ep_id}_success_{success}.mp4", images, fps=5)

# Final success summary
overall_success = np.mean(success_arr)
summary_message = f"**Overall Success** {overall_success} ({np.sum(success_arr)}/{len(success_arr)})"
log_message(log, summary_message)

# Save log as JSON with converted types
with open(f"{logging_dir}/evaluation_log.json", "w") as f:
    json.dump(log, f, indent=4, default=convert_numpy_types)
