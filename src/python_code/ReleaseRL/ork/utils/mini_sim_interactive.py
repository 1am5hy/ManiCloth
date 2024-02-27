import os

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import taichi as ti

from simulator.plb.config import load
from simulator.plb.engine.taichi_env import TaichiEnv
from simulator.test.interactive import set_env_parameters

task_list = ['doughgathering']
data_dir = '/home/ubuntu/Data/interactive/interactive'

for task in task_list:
    ti.init()
    print(task)
    cfg_path = f'simulator/plb/envs/{task}.yml'
    cfg = load(cfg_path)
    task_name = cfg_path.split('_')[-1].split('.')[0]
    env = TaichiEnv(cfg, nn=False, loss=False)
    set_env_parameters(env, yield_stress=200, E=5e3, nu=0.2, task_name=task_name)
    state = env.get_state()  # [x: (2000, 3), v: (2000, 3), F: (2000, 3, 3), C: (2000, 3, 3), primitives: (7)]
    for folder in tqdm.tqdm(os.listdir(data_dir)):
        if task =='doughrolling':
            task = 'doughstretchv'
        if folder.startswith(task):
            try:
                trial_path = os.path.join(data_dir, folder)
                # action_list = np.load(os.path.join(trial_path, 'z_actions.npy'))
                manipulator_pose_list = np.load(os.path.join(trial_path, 'z_manipulator_poses.npy'), allow_pickle=True)
                # real_rotation_list = np.load(os.path.join(trial_path, 'z_real_rotations.npy'))
                shape_files = [i for i in os.listdir(trial_path) if i.startswith('shape')]
                num = len(shape_files)
                for i in range(num):
                    file_path = os.path.join(trial_path, f'shape_{i:04d}.npy')
                    data = np.load(file_path)
                    state['state'][0] = data
                    state['state'][4] = manipulator_pose_list[i]
                    env.set_state(**state)
                    obs = env.render_for_train()
                    np.save(os.path.join(trial_path, f'rgbd_{i:04d}.npy'), obs)

                    plt.figure()
                    plt.imshow((np.array(obs[:, :, :3]) * 255).astype(np.uint8))
                    plt.axis('off')
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"{trial_path}/rgb_{i:04d}.png", bbox_inches='tight', pad_inches=0)
                    plt.cla()
                    plt.close("all")
            except Exception as e:
                print(e)
