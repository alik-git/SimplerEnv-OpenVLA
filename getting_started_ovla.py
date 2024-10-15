import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.policies.openvla.openvla_model import OpenVLAInference
import mediapy as media
import numpy as np

# Ali Debug prefix for logging
p = "ALI DEBUG: "

print(f"{p} starting script")

# Set up environment
# env = simpler_env.make('google_robot_pick_coke_can')
env = simpler_env.make('widowx_spoon_on_towel')
print(f"{p} made env")
obs, reset_info = env.reset()
print(f"{p} before getting instruction")
instruction = env.unwrapped.get_language_instruction()
print("Reset info", reset_info)
print("Instruction", instruction)

print("Control mode:", env.unwrapped.agent.control_mode)

# Initialize OpenVLA policy
openvla_policy = OpenVLAInference(
    saved_model_path="openvla/openvla-7b",  # Use Hugging Face model ID instead of local path
    # policy_setup="google_robot"
    policy_setup="widowx_bridge"
)
openvla_policy.reset(instruction)

# Array to store video frames
frames = []

# Main loop
done, truncated = False, False
while not (done or truncated):
    # Get the observation image for OpenVLA
    image = get_image_from_maniskill2_obs_dict(env.unwrapped, obs)
    
    # Generate action using OpenVLA policy
    openvla_out, action = openvla_policy.step(image=image, task_description=instruction)
    
    # Extract values in the same structure as the sample action
    # Position deltas (x, y, z)
    position_deltas = action['world_vector'].tolist()

    # Rotation (x, y, z) as axis-angle
    rotation_deltas = action['rot_axangle'].tolist()

    # Gripper state
    gripper_state = [action['gripper'][0]]

    # Construct a flat list matching the sample action format
    transformed_action = position_deltas + rotation_deltas + gripper_state

    # Convert the action list to a NumPy array
    transformed_action = np.array(transformed_action, dtype=np.float32)

    # Append the current frame to the frames list
    frames.append(image)

    # Perform environment step with the transformed action
    obs, reward, done, truncated, info = env.step(transformed_action)
    
    # Check for new instructions
    new_instruction = env.unwrapped.get_language_instruction()
    if new_instruction != instruction:
        instruction = new_instruction
        print("New Instruction", instruction)
        openvla_policy.reset(instruction)

# Save the video of the episode
video_path = "openvla_policy_video_widowX_spoon_towel.mp4"
media.write_video(video_path, np.array(frames), fps=10)
print(f"Video saved at {video_path}")

# Episode stats
episode_stats = info.get('episode_stats', {})
print("Episode stats", episode_stats)

