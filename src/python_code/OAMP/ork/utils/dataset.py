import torch
import os
import tqdm


class InteractionDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, device, seq_len=5):
        self.file_list = file_list
        self.device = device
        self.data = []
        self.seq_len = seq_len
        self.data_prepare()

    def data_prepare(self):
        for i in range(len(self.file_list)):
            exp_dir, rgbd_files = self.file_list[i]
            for rgbd_file in tqdm.tqdm(rgbd_files):
                post = rgbd_file.split('_')[1]
                rgbd = torch.load(os.path.join(exp_dir, rgbd_file)).to(self.device)
                action = torch.load(os.path.join(exp_dir, f'action_{post}')).to(self.device)
                pose = torch.load(os.path.join(exp_dir, f'pose_{post}')).to(self.device)
                rotation = torch.load(os.path.join(exp_dir, f'rotation_{post}')).to(self.device)
                if 'Horizon' in exp_dir:
                    for rollout in range(5):
                        for j in range(50 - self.seq_len - 2):
                            self.data.append((rgbd[rollout * 50 + j:rollout * 50 + j + self.seq_len],
                                              action[rollout * 50 + j:rollout * 50 + j + self.seq_len],
                                              rgbd[rollout * 50 + j + 1:rollout * 50 + j + self.seq_len + 1],
                                              pose[rollout * 50 + j:rollout * 50 + j + self.seq_len],
                                              rotation[rollout * 50 + j:rollout * 50 + j + self.seq_len]))
                else:
                    for j in range(rgbd.shape[0] - self.seq_len - 1):
                        self.data.append((rgbd[j:j + self.seq_len], action[j:j + self.seq_len],
                                          rgbd[j + 1:j + self.seq_len + 1], pose[j:j + self.seq_len],
                                          rotation[j:j + self.seq_len]))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    data_dir = '/home/ubuntu/Github/Soft-Manipulation/SoftGPT/softgpt/utils/runs'
    file_list = []
    for exp in tqdm.tqdm(os.listdir(data_dir)):
        exp_dir = os.path.join(data_dir, exp, 'data')
        rgbd_files = [f for f in os.listdir(exp_dir) if f.startswith('rgbd')]
        file_list.append((exp_dir, rgbd_files))

    dataset = InteractionDataset(file_list, device='cuda:0')
