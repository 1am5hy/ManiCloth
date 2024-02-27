import os
import wandb
import logging
import numpy as np
import torch
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset, MultiviewDataset
from sim_for import SimDeform

class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()
        self.conf = ConfigFactory.parse_string(conf_text)
        # self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        if 'general.base_run_dir' in self.conf:
            self.base_run_dir = self.conf['general.base_run_dir']
        else:
            self.base_run_dir = './'
        os.makedirs(self.base_exp_dir, exist_ok=True)
        # self.is_multiview = self.conf.get_bool('general.is_multiview')
        # if self.is_multiview:
        #     self.dataset = MultiviewDataset(self.conf['dataset'])
        # else:
        # self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.max_pe_iter = self.conf.get_int('train.max_pe_iter')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_phy = 500 * self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.learning_rate_alpha_phy = self.conf.get_float('train.learning_rate_alpha_phy')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.warm_up_end_phy = self.conf.get_float('train.warm_up_end_phy', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.use_mesh = self.conf.get_bool('train.use_mesh', default=True)

        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        params_to_train = []

        self.sim_deform = SimDeform(**self.conf['model.sim_deform']).to(self.device)

        # print(self.sim_deform.parameters())
        params_to_train += [{'name':'sim_deform', 'params':self.sim_deform.parameters(), 'lr':self.learning_rate_phy}]
        # print(params_to_train)

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate, amsgrad=True)

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

    def train(self):
        # torch.autograd.set_detect_anomaly(True)
        # self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        num_epochs = res_step  # // self.dataset.n_images
        # wandb.watch(self.coord_map)
        wandb.init()
        wandb.watch(self.sim_deform)

        # self.coord_map.init_pose = self.dataset.fabric_init.detach().clone().to(self.device)
        #
        # scale_mat = self.dataset.scales_all[0][0]
        # fab_init = self.dataset.fabric_init.to(self.device)
        # fab_init = torch.matmul(scale_mat[:3, :3], fab_init.transpose(1, 0)).transpose(1, 0) + scale_mat[:3, 3]
        # position_control = self.dataset.get_trajectory_control().to(self.device)
        # offsets = torch.stack((fab_init[36] - position_control[0], fab_init[52] - position_control[0]), dim=0)
        # self.sim_deform.offsets = offsets

        max_decay = 1.0

        for epoch_i in tqdm(range(num_epochs)):
            if self.iter_step <= 3000:
                prg = self.iter_step / 3000
                delta = max_decay - 1
                decay = max_decay - 3 * delta * prg ** 2 + 2 * delta * prg ** 3
            else:
                decay = 1.0

            wandb_dict = {"epoch": epoch_i}

            position_control = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/real2sim/optitrack_data/npfiles/gp_dyndemo_scaled.npy')
            # position_control = position_control[:-1]
            # print(position_control.shape)

            if len(position_control) > 150:
                position_control = position_control[:150]

            x_init = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/x_init_dyndemo_scaled.npy')
            # x_init = x_init.reshape(-1)

            x = self.sim_deform(x_init, position_control, decay)

            wandb_dict.update({'Statistics/stretch_stiffness': self.sim_deform.phy_params[2].item(),
                               'Statistics/bend_stiffness': self.sim_deform.phy_params[3].item(),
                               'Statistics/density': self.sim_deform.phy_params[4].item()})

            markers = np.load('/home/ubuntu/Github/DiffCloth/src/python_code/real2sim/optitrack_data/npfiles/marker_path_dyndemo_scaled.npy')
            if len(markers) > 150:
                markers = markers[:150]

            markers = torch.tensor(markers).float()

            # x = np.array(x)
            # x = np.vstack([x_init, x])
            part = x.clone().detach()
            # part = x
            markers = self.upsample(markers, part)
            # x = x * 2

            # marker_part_loss = markers - x
            # print(marker_part_loss[:, -1])

            loss = torch.log(torch.nn.functional.mse_loss(markers, x, reduction='sum'))

            # print(loss)
            #
            self.optimizer.zero_grad()
            # loss.retain_grad()
            loss.backward()

            self.optimizer.step()

            # self.update_learning_rate()

            # if self.iter_step % 30000 == 0:
            #     decay -= 0.01
            self.iter_step += 1

            wandb_dict.update({'iter_step': self.iter_step})
            wandb_dict.update({'Loss/loss': loss})
            wandb_dict.update({'learning rate': self.optimizer.param_groups[0]['lr']})
            wandb_dict.update({'decay': decay})

            wandb.log(wandb_dict)

            # if self.iter_step % self.report_freq == 0:
            #     os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
            #     mesh_file = os.path.join(self.base_exp_dir, 'meshes', 'mesh_{:06d}.npz'.format(self.iter_step))
            #     np.savez(mesh_file, gt=self.dataset.parts_gt.detach().cpu().numpy(),
            #              pred=self.dataset.sim_meshes.detach().cpu().numpy())
            #     print(self.base_exp_dir)
            #     print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, self.loss,
            #                                                1))  # , self.optimizer.param_groups[0]['lr'], decay))




    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name),
                                map_location=self.device)

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def upsample(self, marker, particle):

        """
        Input: - the marker points
               - the simulation point cloud

        Output: Point Cloud of marker points mixed into the simulation point cloud
        """

        # particle = particle + 0.001
        particle[:, 0] = marker[:, 0]
        particle[:, 7] = marker[:, 1]
        particle[:, 14] = marker[:, 2]
        particle[:, 84] = marker[:, 3]
        particle[:, 105] = marker[:, 4]
        particle[:, 119] = marker[:, 5]
        particle[:, 156] = marker[:, 6]
        particle[:, 210] = marker[:, 7]
        particle[:, 217] = marker[:, 8]
        particle[:, 224] = marker[:, 9]
        # print(marker[:, 7])
        # print(particle[:, 870])


        return particle


if __name__ == '__main__':

    run = Runner("/home/ubuntu/Github/DiffCloth/src/python_code/real2sim/cloth.conf", case='cloth')
    run.train()