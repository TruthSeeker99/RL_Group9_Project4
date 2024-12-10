#!/usr/bin/env python3
import argparse
import os

import diambra.arena
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings, \
    RecordingSettings
from diambra.arena.stable_baselines3.sb3_utils import AutoSave
from diambra.engine import SpaceTypes
from stable_baselines3.common.vec_env import SubprocVecEnv

from settings import all_settings
from utils import record_video, record_single_video, build_env

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--model', default='PPO_final', help='the function name.')
    args = parser.parse_args()

    if 'PPO' in args.model:
        sb3 = True
        env, num_envs = build_env(False, sb3=sb3, render_mode='rgb_array', all_settings=all_settings)
        # env = None
        agent = PPO('MultiInputPolicy', env, verbose=1, device='cuda')
        env.close()
        test_env, num_envs = build_env(True, sb3=sb3, render_mode='rgb_array',
                                       all_settings=all_settings, test=True)
        # print(agent.policy.action_net)
        # Train the agent
        # for _ in range(200):

        model_folder = './ckpts/'
        # model_checkpoint = 'PPO'
        # new_model_checkpoint = model_checkpoint + "_final"
        new_model_checkpoint = args.model
        # new_model_checkpoint = model_checkpoint + "_autosave_1450800"
        model_path = os.path.join(model_folder, new_model_checkpoint)
        agent.load(model_path)
        record_single_video(test_env, agent,
                            video_length=10240 * 4,
                            video_folder='./videos/', env_id=args.model, all_settings=all_settings, episodes=1)
