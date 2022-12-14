import os.path as osp

import torch
import torch.utils.data as data
import numpy as np


class TRNHDDDataLayer(data.Dataset):
    def __init__(self, args, phase='train'):
        self.data_root = args.data_root
        self.camera_feature = args.camera_feature
        self.optical_feature = args.optical_feature
        self.sessions = getattr(args, phase+'_session_set')
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.training = phase == 'train'

        self.inputs = []
        for session in self.sessions:
            sensor = np.load(osp.join(self.data_root, 'sensor', session+'.npy'))
            target = np.load(osp.join(self.data_root, 'target', session+'.npy'))
            seed = np.random.randint(self.enc_steps) if self.training else 90
            for start, end in zip(
                range(seed, target.shape[0] - self.dec_steps, self.enc_steps),
                range(seed + self.enc_steps, target.shape[0] - self.dec_steps, self.enc_steps)):
                enc_target = target[start:end]
                dec_target = self.get_dec_target(target[start:end + self.dec_steps])
                self.inputs.append([
                    session, start, end, sensor[start:end],
                    enc_target, dec_target,
                ])

    def get_dec_target(self, target_vector):
        target_matrix = np.zeros((self.enc_steps, self.dec_steps))
        for i in range(self.enc_steps):
            for j in range(self.dec_steps):
                # 0 -> [1, 2, 3]
                # target_matrix[i,j] = target_vector[i+j+1]
                # 0 -> [0, 1, 2]
                target_matrix[i,j] = target_vector[i+j]
        return target_matrix

    def __getitem__(self, index):
        session, start, end, sensor_inputs, enc_target, dec_target = self.inputs[index]

        camera_inputs = np.load(
            osp.join(self.data_root, self.camera_feature, session+'.npy'), mmap_mode='r')[start:end]
        camera_inputs = torch.as_tensor(camera_inputs.astype(np.float32))
        optical_inputs = np.load(
            osp.join(self.data_root, self.optical_feature, session + '.npy'), mmap_mode='r')[start:end]
        optical_inputs = torch.as_tensor(optical_inputs.astype(np.float32))
        sensor_inputs = torch.as_tensor(sensor_inputs.astype(np.float32))
        enc_target = torch.as_tensor(enc_target.astype(np.int64))
        dec_target = torch.as_tensor(dec_target.astype(np.int64))

        return camera_inputs, sensor_inputs, optical_inputs, enc_target, dec_target.view(-1)

    def __len__(self):
        return len(self.inputs)
