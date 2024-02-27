"""
Cloth (RofuncRL)
===========================
Cloth task, trained by RofuncRL
"""

import argparse

from rofunc.config.utils import omegaconf_to_dict, get_config
from rofunc.learning.RofuncRL.tasks import task_map
from rofunc.learning.RofuncRL.trainers import trainer_map
from rofunc.learning.pre_trained_models.download import model_zoo
from rofunc.learning.utils.utils import set_seed

from cloth_env import ClothEnv
cenv = ClothEnv()

def train(custom_args):
    # Config task and trainer parameters for Isaac Gym environments

    args_overrides = ["task={}".format(custom_args.task),
                      "train={}{}".format(custom_args.task, custom_args.agent.upper()),
                      "sim_device={}".format(custom_args.sim_device),
                      "rl_device={}".format(custom_args.rl_device),
                      "graphics_device_id={}".format(custom_args.graphics_device_id),
                      "headless={}".format(custom_args.headless),
                      "num_envs={}".format(custom_args.num_envs)]
    # cfg = get_config('./learning/rl', 'config', args=args_overrides)
    # cfg_dict = omegaconf_to_dict(cfg.task)
    cfg = get_config(absl_config_path="/home/ubuntu/Github/DiffCloth/src/python_code/config", config_name="config", args=args_overrides)
    set_seed(cfg.train.Trainer.seed)

    # Instantiate the Isaac Gym environment
    env = ClothEnv()

    # Instantiate the RL trainer
    trainer = trainer_map[custom_args.agent](cfg=cfg.train,
                                             env=env,
                                             device=cfg.rl_device,
                                             env_name=custom_args.task)
    # Start training
    trainer.train()


def inference(custom_args):
    # Config task and trainer parameters for Isaac Gym environments
    args_overrides = ["task={}".format(custom_args.task),
                      "train={}{}".format(custom_args.task, custom_args.agent.upper()),
                      "sim_device={}".format(custom_args.sim_device),
                      "rl_device={}".format(custom_args.rl_device),
                      "graphics_device_id={}".format(custom_args.graphics_device_id),
                      "headless={}".format(False),
                      "num_envs={}".format(16)]

    # cfg = get_config('./learning/rl', 'config', args=args_overrides)
    # cfg_dict = omegaconf_to_dict(cfg.task)

    cfg = get_config(absl_config_path="/home/ubuntu/Github/DiffCloth/src/python_code/config", config_name="config", args=args_overrides)
    # cfg_dict = omegaconf_to_dict(cfg.task)

    set_seed(cfg.train.Trainer.seed)

    # Instantiate the Isaac Gym environment
    infer_env = ClothEnv()

    # Instantiate the RL trainer
    trainer = trainer_map[custom_args.agent](cfg=cfg.train,
                                             env=infer_env,
                                             device=cfg.rl_device,
                                             env_name=custom_args.task)

    # load checkpoint
    if custom_args.ckpt_path is None:
        custom_args.ckpt_path = "/home/ubuntu/Github/DiffCloth/src/python_code/runs/RofuncRL_PPOTrainer_Cloth_23-09-13_18-35-20-451470/checkpoints/ckpt_8760.pth"
    trainer.agent.load_ckpt(custom_args.ckpt_path)

    # Start inference
    trainer.inference()


if __name__ == '__main__':
    gpu_id = 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Cloth")
    parser.add_argument("--agent", type=str, default="ppo")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--sim_device", type=str, default="cuda:{}".format(gpu_id))
    parser.add_argument("--rl_device", type=str, default="cuda:{}".format(gpu_id))
    parser.add_argument("--graphics_device_id", type=int, default=gpu_id)
    parser.add_argument("--headless", type=str, default="True")
    parser.add_argument("--inference", action="store_true", help="turn to inference mode while adding this argument")
    # parser.add_argument("--inference", action="store_true", help="turn to inference mode while adding this argument", default="True")
    parser.add_argument("--ckpt_path", type=str, default=None)
    custom_args = parser.parse_args()
    # inference steps = 100!

    if not custom_args.inference:
        train(custom_args)
    else:
        inference(custom_args)