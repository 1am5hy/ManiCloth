import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from rofunc.config.utils import omegaconf_to_dict, get_config

from ork.trainers import trainer_map
# from ork.utils.env_utils import make_env
# from ork.tasks.dough_deformation import DoughDeformationTask
# from task import ClothHangTask
from rofunc.learning.utils.utils import set_seed
from task_table import ClothTableTask
from env_table import cloth_env


def train(custom_args):
    args_overrides = ["task={}".format(custom_args.task),
                      "train={}{}{}".format(custom_args.task, custom_args.mode, custom_args.agent),
                      "device_id={}".format(custom_args.sim_device),
                      "rl_device=cuda:{}".format(custom_args.rl_device)]
    cfg = get_config(absl_config_path='/home/ubuntu/Github/DiffCloth/src/python_code/OAMP/ork/config',
                     config_name='config', args=args_overrides)
    set_seed(cfg.train.Trainer.seed)

    env = cloth_env()

    # env = make_env(custom_args, cfg.task)
    # env.seed(cfg.train.Trainer.seed)
    env = ClothTableTask(cfg=omegaconf_to_dict(cfg.task),
                        sim_device=f'cuda:{cfg.device_id}',
                        cloth_env=env)

    trainer = trainer_map[custom_args.agent](cfg=cfg,
                                             env=env,
                                             device=cfg.rl_device,
                                             env_name=custom_args.task)

    # Start training
    trainer.train()


def inference(custom_args):
    args_overrides = ["task={}".format(custom_args.task),
                      "train={}{}{}".format(custom_args.task, custom_args.mode, custom_args.agent),
                      "device_id={}".format(custom_args.sim_device),
                      "rl_device=cuda:{}".format(custom_args.rl_device)]
    cfg = get_config(absl_config_path='/home/ubuntu/Github/DiffCloth/src/python_code/OAMP/ork/config',
                     config_name='config', args=args_overrides)
    set_seed(cfg.train.Trainer.seed)

    env = cloth_env()
    # env.seed(cfg.train.Trainer.seed)
    env = ClothTableTask(cfg=omegaconf_to_dict(cfg.task),
                        sim_device=f'cuda:{cfg.device_id}',
                        cloth_env=env)

    trainer = trainer_map[custom_args.agent](cfg=cfg,
                                             env=env,
                                             device=cfg.rl_device,
                                             env_name=custom_args.task,
                                             inference=True)

    trainer.agent.load_ckpt(custom_args.ckpt_path)
    trainer.inference()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DoughRolling, DoughCutting, DoughGathering, DoughShaping
    parser.add_argument("--task", type=str, default="ClothTable")
    # Available agent: SoftGPT, GraphPPO, GraphSAC, PPO, SAC
    parser.add_argument("--agent", type=str, default="ORK")
    parser.add_argument("--policy", type=str, default="ppo")  # Available policy agent: ppo, sac, unless agent=SoftGPT
    parser.add_argument("--mode", type=str, default="Particle")  # Available modes: Particle, Image, Graph
    parser.add_argument("--sim_device", type=int, default=0)
    parser.add_argument("--rl_device", type=int, default=0)
    parser.add_argument("--inference", action="store_true", help="turn to inference mode while adding this argument")
    parser.add_argument("--ckpt_path", type=str, default="/home/ubuntu/Github/DiffCloth/src/python_code/OAMP/runs/RofuncRL_ORKTrainer_ClothTable_24-02-26_07-07-12-607141/checkpoints/best_ckpt.pth")
    custom_args = parser.parse_args()
    #
    if not custom_args.inference:
        train(custom_args)
    else:
        inference(custom_args)

    # Not sure if this will work
    # RofuncRL_ORKTrainer_ClothTable_24-02-05_16-15-49-936984

