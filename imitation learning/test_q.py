from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from diambra.arena import SpaceTypes, Roles, EnvironmentSettings
import wandb
from wandb.integration.sb3 import WandbCallback
import os

# Settings
settings = EnvironmentSettings()
settings.frame_shape = (60, 60, 0)
settings.characters = ("Ken")
settings.action_space = SpaceTypes.DISCRETE
settings.role = Roles.P1
settings.continue_game = 0.0

# Wrappers Settings
wrappers_settings = WrappersSettings()
wrappers_settings.normalize_reward = True
wrappers_settings.stack_frames = 3
wrappers_settings.add_last_action = True
wrappers_settings.stack_actions = 6
wrappers_settings.scale = True
wrappers_settings.exclude_image_scaling = True
wrappers_settings.role_relative = True
wrappers_settings.flatten = True
wrappers_settings.filter_keys = ["action", "own_health", "opp_health", "own_side", "opp_side", "opp_character", "stage", "timer"]

# Wandb configuration
config = {
    "algorithm": "DQN",
    "policy": "MultiInputPolicy",
    "total_timesteps": 1000000,
    "environment": "diambra-arena-sb3",
    'Frame Size': "60, 60, 0"
}
# Create environment
env, num_envs = make_sb3_env("sfiii3n", settings, wrappers_settings, render_mode='human')
print("Activated {} environment(s)".format(num_envs))

# Instantiate the agent
agent = DQN("MultiInputPolicy", env, verbose=1, buffer_size=50000, batch_size=32)

agent.load("dqn_sfiii3n.zip")
mean_reward, std_reward = evaluate_policy(agent, agent.get_env(), n_eval_episodes=3)

print("Reward: {} (avg) ± {} (std)".format(mean_reward, std_reward))

# Run trained agent
observation = env.reset()
cumulative_reward = 0
while True:
    env.render()

    action, _state = agent.predict(observation, deterministic=True)
    observation, reward, done, info = env.step(action)

    cumulative_reward += reward
    if (reward != 0):
        print("Cumulative reward =", cumulative_reward)

    if done:
        observation = env.reset()
        break
