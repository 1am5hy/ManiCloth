import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

data_dir = '/home/ubuntu/Github/Soft-Manipulation/SoftGPT/softgpt/utils/runs'


def get_img(z):
    exp_dir, rgbd_file = z
    rgbd = torch.load(os.path.join(exp_dir, rgbd_file))
    start_num = int(rgbd_file.split('_')[1].split('.')[0]) - 250
    for i in range(rgbd.shape[0]):
        if f'rgb_{start_num + i:08d}.png' in os.listdir(exp_dir):
            continue
        plt.figure()
        plt.imshow((np.array(rgbd[i, :3, :, :].cpu()).transpose((2, 1, 0)) * 255).astype(np.uint8))
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        # plt.show()
        # print()
        plt.savefig(f"{exp_dir}/rgb_{start_num + i:08d}.png", bbox_inches='tight', pad_inches=0)
        plt.cla()
        plt.close("all")


def main():
    for exp in os.listdir(data_dir):
        with Pool(12) as pool:
            exp_dir = os.path.join(data_dir, exp, 'data')
            rgbd_files = [f for f in os.listdir(exp_dir) if f.startswith('rgbd')]
            z = [(exp_dir, rgbd_file) for rgbd_file in rgbd_files]
            list((tqdm.tqdm(pool.imap(get_img, z), total=len(z))))


if __name__ == '__main__':
    main()
