import os
import numpy as np
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.policies.rt1.rt1_model import RT1Inference
import mediapy as media

# Ali Debug prefix for logging
p = "ALI DEBUG: "

print(f"{p} starting script")

# Set up environment
task_name = "google_robot_pick_coke_can"
env = simpler_env.make(task_name)
print(f"{p} made env")
obs, reset_info = env.reset()
print(f"{p} before getting instruction")
instruction = env.unwrapped.get_language_instruction()
print("Reset info", reset_info)
print("Instruction", instruction)

print("Control mode:", env.unwrapped.agent.control_mode)

# Load the RT-1 checkpoint path
checkpoint_dir = "/home/kasm-user/SimplerEnv-OpenVLA/checkpoints"
rt_1_checkpoint = os.path.join(checkpoint_dir, "rt_1_x_tf_trained_for_002272480_step")
policy_setup = "google_robot" if "google" in task_name else "widowx_bridge"

# Initialize RT-1 policy
rt1_policy = RT1Inference(saved_model_path=rt_1_checkpoint, policy_setup=policy_setup)
rt1_policy.reset(instruction)

# Array to store video frames
frames = []

# Main loop
predicted_terminated, success, truncated = False, False, False
timestep = 0
while not (predicted_terminated or truncated):
    # Get the observation image for RT-1
    image = get_image_from_maniskill2_obs_dict(env.unwrapped, obs)
    
    # Step the RT-1 model; "raw_action" is raw model action output; "action" is the processed action
    raw_action, action = rt1_policy.step(image)
    
    # Determine if the episode should terminate based on RT-1's output
    predicted_terminated = bool(action["terminate_episode"][0] > 0)

    # Perform environment step with RT-1 action directly
    obs, reward, success, truncated, info = env.step(
        np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
    )

    # Append the current frame to the frames list
    frames.append(image)

    # Log info for each timestep
    print(f"Step {timestep}: info={info}")
    
    # Update image observation and increment timestep
    timestep += 1

# Save the video of the episode
video_path = "rt1_policy_video.mp4"
media.write_video(video_path, np.array(frames), fps=10)
print(f"Video saved at {video_path}")

# Episode stats
episode_stats = info.get('episode_stats', {})
print("Episode stats", episode_stats)
