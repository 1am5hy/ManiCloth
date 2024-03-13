from src.python_code.OAMP.eval.main_eval import inference
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser()
# DoughRolling, DoughCutting, DoughGathering, DoughShaping
parser.add_argument("--task", type=str, default="ClothHang")
# Available agent: SoftGPT, GraphPPO, GraphSAC, PPO, SAC
parser.add_argument("--agent", type=str, default="ORK")
parser.add_argument("--policy", type=str, default="ppo")  # Available policy agent: ppo, sac, unless agent=SoftGPT
parser.add_argument("--mode", type=str, default="Particle")  # Available modes: Particle, Image, Graph
parser.add_argument("--sim_device", type=int, default=0)
parser.add_argument("--rl_device", type=int, default=0)
parser.add_argument("--inference", action="store_false", help="turn to inference mode while adding this argument")
parser.add_argument("--ckpt_path", type=str, default="/home/ubuntu/Github/ManiCloth/src/python_code/OAMP/runs/RofuncRL_ORKTrainer_ClothHang_24-02-27_09-53-01-768486/checkpoints/best_ckpt.pth")
custom_args = parser.parse_args()

# new = inference(custom_args, np.array([10, 12, 13]))
# print(new)

exploration_steps = 100

init_cloth_params = np.array([689.16, 0.09])
param = []
for i in range(exploration_steps):
    param.append(init_cloth_params)
param = np.array(param)

random_param_1 = np.random.rand(exploration_steps, 1) * 2000
random_param_2 = np.random.rand(exploration_steps, 1) * 10
random = np.hstack([random_param_1, random_param_2])
random = random + param

bar_pos = []
error = []