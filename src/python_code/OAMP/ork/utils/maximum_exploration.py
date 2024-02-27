import argparse
import os

import rofunc as rf
import torch
import tqdm
from rofunc.config.utils import get_config

from softgpt.trainer import trainer_map
from softgpt.utils.env_utils import make_env

gpu_id = 3
os.environ["CUDA_VISIBLE_DEVICES"] = f'{gpu_id}'

parser = argparse.ArgumentParser()
# DoughRolling, DoughCutting, DoughGathering, DoughShaping
parser.add_argument("--task", type=str, default="DoughRollingExp")
# Available agent: SoftGPT, GraphPPO, GraphSAC, PPO, SAC
parser.add_argument("--agent", type=str, default="PPOHorizon")
parser.add_argument("--policy", type=str, default="ppo")  # Available policy agent: ppo, sac, unless agent=SoftGPT
parser.add_argument("--mode", type=str, default="Particle")  # Available modes: Particle, Image, Graph
parser.add_argument("--device", type=str, default="cuda:0")
custom_args = parser.parse_args()


class ExplorationTrainer(trainer_map[custom_args.agent]):
    def __init__(self, cfg, env, device, env_name):
        self.__class__.__name__ = f'{custom_args.agent}Trainer'
        super().__init__(cfg, env, device, env_name)
        self.states = []
        self.actions = []
        self.poses = []
        self.real_rotations = []
        rf.utils.create_dir(os.path.join(self.exp_dir, 'data'))

        self.eval_flag = False

    def train(self):
        """
        Main training loop.
        - Reset the environment
        - For each step:
            - Pre-interaction
            - Obtain action from agent
            - Interact with environment
            - Store transition
            - Reset the environment
            - Post-interaction
        - Close the environment
        """
        # reset env
        states, infos = self.env.reset()
        with tqdm.trange(self.maximum_steps, ncols=80, colour='green') as self.t_bar:
            for i in self.t_bar:
                if i % 50 == 0 and custom_args.agent in ['PPOHorizon', 'SACHorizon']:
                    states = torch.unsqueeze(
                        torch.as_tensor(self.env.reset_primitives().copy(), dtype=torch.float32), dim=0).to(self.device)

                self.pre_interaction()
                # Obtain action from agent
                with torch.no_grad():
                    actions = self.get_action(states)
                    self.states.append(states)
                    self.actions.append(actions)
                    current_pose, rotation_real = self.env.taichi_env.primitives[0].get_state(0)
                    self.poses.append(torch.tensor([current_pose]))
                    if rotation_real is not None:
                        self.real_rotations.append(torch.tensor([rotation_real]))

                # Interact with environment
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                with torch.no_grad():
                    # Store transition
                    self.agent.store_transition(states=states, actions=actions, next_states=next_states,
                                                rewards=rewards, terminated=terminated, truncated=truncated,
                                                infos=infos)
                self.post_interaction()
                self._step += 1

                with torch.no_grad():
                    # Reset the environment
                    if terminated.any() or truncated.any():
                        self.states.append(next_states)
                        torch.save(torch.cat(self.states, dim=0),
                                   os.path.join(self.exp_dir, 'data', f'rgbd_{self._step:08}.pt'))
                        torch.save(torch.cat(self.actions, dim=0),
                                   os.path.join(self.exp_dir, 'data', f'action_{self._step:08}.pt'))
                        torch.save(torch.cat(self.poses, dim=0),
                                   os.path.join(self.exp_dir, 'data', f'pose_{self._step:08}.pt'))
                        if rotation_real is not None:
                            torch.save(torch.cat(self.real_rotations, dim=0),
                                       os.path.join(self.exp_dir, 'data', f'rotation_{self._step:08}.pt'))

                        self.states = []
                        self.actions = []
                        self.poses = []
                        self.real_rotations = []

                        states, infos = self.env.reset()
                    else:
                        states = next_states.clone()

        # close the environment
        self.env.close()
        # close the logger
        self.writer.close()
        self.rofunc_logger.info('Training complete.')


def explore(custom_args):
    args_overrides = ["task={}".format(custom_args.task),
                      "train={}{}{}".format(custom_args.task, custom_args.mode, custom_args.agent)]
    cfg = get_config(absl_config_path='/home/ubuntu/Github/Soft-Manipulation/SoftGPT/softgpt/config',
                     config_name='config', args=args_overrides)

    env = make_env(custom_args, cfg.task)
    trainer = ExplorationTrainer(cfg=cfg.train, env=env, device=custom_args.device, env_name=custom_args.task)
    trainer.train()


if __name__ == '__main__':
    explore(custom_args)
