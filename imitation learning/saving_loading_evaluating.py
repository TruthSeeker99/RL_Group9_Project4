from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from diambra.arena import SpaceTypes, Roles, EnvironmentSettings
import wandb
from wandb.integration.sb3 import WandbCallback


def main():
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
    run = wandb.init(
        project="sfiii3n_DQN",
        config=config,
        sync_tensorboard=True,
        save_code=True,
    )


    # Create environment
    env, num_envs = make_sb3_env("sfiii3n", settings, wrappers_settings)
    print("Activated {} environment(s)".format(num_envs))

    # Instantiate the agent
    agent = DQN("MultiInputPolicy", env, verbose=1, buffer_size=50000, batch_size=32, tensorboard_log=f"runs/{run.id}")
    # Train the agent
    agent.learn(
            total_timesteps=config["total_timesteps"],
            callback=WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{run.id}",
                verbose=2,
                # every
                model_save_freq=100000,
            ),
        )
    # Save the agent
    agent.save("dqn_sfiii3n")
    del agent  # delete trained agent to demonstrate loading

    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the agent was trained vs the current one
    # agent = A2C.load("a2c_doapp", env=env, print_system_info=True)
    agent = DQN.load("dqn_sfiii3n", env=env)

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    mean_reward, std_reward = evaluate_policy(agent, agent.get_env(), n_eval_episodes=3)
    print("Reward: {} (avg) ± {} (std)".format(mean_reward, std_reward))

    #Log evaluation metrics:
    wandb.log({"mean_reward": mean_reward, "std_reward": std_reward})

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

    # Close the environment
    env.close()
    run.finish()
    # Return success
    return 0

if __name__ == "__main__":
    main()