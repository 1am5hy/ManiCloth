"""
 Copyright 2023, Junjia LIU, jjliu@mae.cuhk.edu.hk

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import rofunc as rf
import torch
import tqdm
from rofunc.learning.RofuncRL.trainers.sac_trainer import SACTrainer as RofuncSACTrainer


class SACTrainer(RofuncSACTrainer):
    def train(self):
        """
        Main training loop. \n
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
            values = {'loss': 0}
            for _ in self.t_bar:
                self.pre_interaction()
                # Obtain action from agent
                with torch.no_grad():
                    actions = self.get_action(states)

                # Interact with environment
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                next_states, rewards, terminated, truncated = self.agent.multi_gpu_transfer(next_states, rewards,
                                                                                            terminated,
                                                                                            truncated)


                with torch.no_grad():
                    # Store transition
                    self.agent.store_transition(states=states, actions=actions, next_states=next_states,
                                                rewards=rewards, terminated=terminated, truncated=truncated,
                                                infos=infos)
                self.post_interaction()
                self._step += 1

                # values['last_iou'] = infos['incremental_iou']
                # values['total_iou'] += infos['incremental_iou']
                # values['sdf'] += infos['sdf_loss']
                # values['density'] += infos['density_loss']
                # values['contact'] += infos['contact_loss']

                values['loss'] += infos['loss']
                if not self._step % self.write_interval and self.write_interval > 0:
                    # self.rofunc_logger.info(f"Step: {self._step}, last_iou: {values['last_iou']:.4f}, "
                    #                         f"total_iou: {values['total_iou']:.4f}, sdf: {values['sdf']:.4f}, "
                    #                         f"density: {values['density']:.4f}, contact: {values['contact']:.4f}, "
                    #                         f"loss: {values['loss']:.4f}", local_verbose=False)
                    # values = {'last_iou': 0, 'total_iou': 0, 'sdf': 0, 'density': 0, 'contact': 0, 'loss': 0}
                    self.rofunc_logger.info(f"Step: {self._step}, "
                                            f"loss: {values['loss']:.4f}", local_verbose=False)
                    values = {'loss': 0}

                with torch.no_grad():
                    # Reset the environment
                    if terminated.any() or truncated.any():
                        states, infos = self.env.reset()
                    else:
                        states = next_states.clone()

        # close the environment
        self.env.close()
        # close the logger
        self.writer.close()
        self.rofunc_logger.info('Training complete.')

    def eval(self):
        # reset env
        img_log_dir = os.path.join(self.exp_dir, 'step_{}'.format(self._rollout))
        if 'Image' in self.cfg.Trainer.task_name:
            rf.utils.create_dir(img_log_dir)

        eval_rew_list = []
        states, infos = self.env.reset()
        for i in range(self.eval_steps):
            self.pre_interaction()
            # Obtain action from agent
            with torch.no_grad():
                actions = self.get_action(states)

            # Interact with environment
            next_states, rewards, terminated, truncated, infos = self.env.step(actions)

            eval_rew_list.append(rewards.item())
            if i < 250 and 'Image' in self.cfg.Trainer.task_name:
                plt.figure()
                plt.imshow((np.array(next_states[0, :3].detach().cpu()).transpose((2, 1, 0)) * 255).astype(np.uint8))
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"{img_log_dir}/{i:04d}.png", bbox_inches='tight', pad_inches=0)
                plt.cla()
                plt.close("all")

            with torch.no_grad():
                # Reset the environment
                if terminated.any() or truncated.any():
                    states, infos = self.env.reset()
                else:
                    states = next_states.clone()

        self.eval_rew_mean = sum(eval_rew_list) / 5
