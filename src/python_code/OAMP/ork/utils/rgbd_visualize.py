import matplotlib.pyplot as plt
import numpy as np


def update_camera(env):
    env.renderer.camera_pos[0] = 0.5
    env.renderer.camera_pos[1] = 2.5
    env.renderer.camera_pos[2] = 0.5
    env.renderer.camera_rot = (1.57, 0.0)
    env.render_cfg.defrost()
    env.render_cfg.camera_pos_1 = (0.5, 2.5, 2.8)
    env.render_cfg.camera_rot_1 = (0.8, 0.)
    env.render_cfg.camera_pos_2 = (2.6, 2.5, 0.)
    env.render_cfg.camera_rot_2 = (0.8, 1.8)
    env.render_cfg.camera_pos_3 = (-1.9, 2.5, 0)
    env.render_cfg.camera_rot_3 = (0.8, -1.8)
    env.render_cfg.camera_pos_4 = (0.5, 2.5, -1.8)
    env.render_cfg.camera_rot_4 = (0.8, 3.14)


def update_test_camera(env):
    env.renderer.camera_pos[0] = 0.5
    env.renderer.camera_pos[1] = 2.5
    env.renderer.camera_pos[2] = 2.8
    env.renderer.camera_rot = (0.8, 0.)


def plot_multi(rgb, depth, save_path=None):
    plt.figure(figsize=(8, 16))

    plt.subplot(421)
    plt.imshow(rgb[0])
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plt.subplot(422)
    plt.imshow(depth[0])
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plt.subplot(423)
    plt.imshow(rgb[1])
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plt.subplot(424)
    plt.imshow(depth[1])
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plt.subplot(425)
    plt.imshow(rgb[2])
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plt.subplot(426)
    plt.imshow(depth[2])
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plt.subplot(427)
    plt.imshow(rgb[3])
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plt.subplot(428)
    plt.imshow(depth[3])
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.cla()
    plt.close("all")


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
