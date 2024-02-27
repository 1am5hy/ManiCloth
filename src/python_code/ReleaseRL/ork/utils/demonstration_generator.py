import argparse
import datetime
import json
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import cv2
import numpy as np
import shutup

shutup.please()

import torch

from plb.algorithms.logger import Logger
from plb.engine.taichi_env import TaichiEnv
from plb.envs import make
from plb.optimizer.solver import Solver

RL_ALGOS = ['sac', 'td3', 'ppo']
DIFF_ALGOS = ['action', 'nn']


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Rollingpin")
    parser.add_argument("--obs_mode", type=str, default="Particle")  # Particle, Image, Graph
    parser.add_argument("--variant", type=int, default=1)
    parser.add_argument("--algo", type=str, default="action")  # ppo, sac, td3, action, nn
    # parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sdf_loss", type=float, default=10)
    parser.add_argument("--density_loss", type=float, default=10)
    parser.add_argument("--contact_loss", type=float, default=1)
    parser.add_argument("--soft_contact_loss", action='store_true')

    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--horizon", type=int, default=50)
    # differentiable physics parameters
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--softness", type=float, default=666.)
    parser.add_argument("--optim", type=str, default='Adam', choices=['Adam', 'Momentum'])

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    if args.num_steps is None:
        if args.algo in DIFF_ALGOS:
            args.num_steps = 50 * 200
        else:
            args.num_steps = 500000

    env_name = "{}-{}-v{}".format(args.task, args.obs_mode, args.variant)
    log_path = "runs/{}{}{}_{}".format(args.task, args.obs_mode, args.algo,
                                       datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    logger = Logger(log_path)
    set_random_seed(args.seed)

    env = make(env_name, nn=(args.algo == 'nn'), sdf_loss=args.sdf_loss,
               density_loss=args.density_loss, contact_loss=args.contact_loss,
               soft_contact_loss=args.soft_contact_loss)
    env.seed(args.seed)

    solve_action(env, log_path, logger, args, env_name)


def solve_action(env, path, logger, args, env_name):
    os.makedirs(path, exist_ok=True)
    env.reset()
    taichi_env: TaichiEnv = env.unwrapped.taichi_env
    T = env._max_episode_steps
    solver = Solver(taichi_env, logger, None,
                    n_iters=(args.num_steps + T - 1) // T, softness=args.softness, horizon=T,
                    **{"optim.lr": args.lr, "optim.type": args.optim, "init_range": 0.0001})

    action = solver.solve()

    demo_dict = []
    for idx, act in enumerate(action):
        obs, r, _, loss_info = env.step(act)
        demo_dict.append({'obs': obs[:-7].tolist(), 'r': r, 'loss_info': loss_info, 'tool_pose': obs[-7:].tolist()})
        img = env.render(mode='rgb_array')
        cv2.imwrite(f"{path}/{idx:04d}.png", img[..., ::-1])

        with open(f"{path}/{env_name}_demo_data.json", mode='w') as json_file:
            json.dump(demo_dict, json_file, indent=4)


if __name__ == '__main__':
    main()
