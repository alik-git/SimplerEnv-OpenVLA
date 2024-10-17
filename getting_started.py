import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import mediapy as media
import numpy as np

# Set up environment
env = simpler_env.make('google_robot_pick_coke_can')
obs, reset_info = env.reset()

# Fetch instruction from the environment
instruction = env.unwrapped.get_language_instruction()
print("Reset info", reset_info)
print("Instruction", instruction)

# Video frames array
frames = []

done, truncated = False, False
while not (done or truncated):
    # Capture frame from environment
    image = get_image_from_maniskill2_obs_dict(env.unwrapped, obs)
    
    # Sample an action
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    
    # Append frame
    frames.append(image)

# Save the video to an .mp4 file
video_path = "basic_env_video.mp4"
media.write_video(video_path, np.array(frames), fps=10)
print(f"Video saved at {video_path}")

episode_stats = info.get('episode_stats', {})
print("Episode stats", episode_stats)
