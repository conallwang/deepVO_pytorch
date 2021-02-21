from torch.utils.data import Dataset, DataLoader

from glob import glob
from config import *

from PIL import Image

import numpy as np

class DeepVODataset(Dataset):
    def __init__(self, seq=2, interval=1, phase='train'):
        super().__init__()

        self.seq = seq
        self.interval = interval
        self.phase = phase

        self.img_list, self.start_idx, self.poses = self.load_data()

    def load_data(self):
        start_idx = []
        img_list = []
        poses = []
        count = 0

        if self.phase == 'train':
            # choose sequence 00, 02, 08, 09
            for i in train_seq:
                # get relative poses
                pose = np.loadtxt(f'{poses_path}/{i:02}_rp.txt')
                pose = np.concatenate((pose, -100 * np.ones((1, 6))), axis=0)

                poses.append(pose)

                # get all image name in special sequence
                img_path = glob(f'{dataset_path}/{i:02}/image_2/*.png')
                img_path.sort()

                img_list.extend(img_path)
                for j in range(len(img_path)):
                    if j <= len(img_path) - self.seq and j % self.seq == 0:
                        start_idx.append(count)
                    count += 1
        else:
            for i in vaild_seq:
                # get relative poses
                pose = np.loadtxt(f'{poses_path}/{i:02}_rp.txt')
                pose = np.concatenate((pose, -100 * np.ones((1, 6))), axis=0)

                poses.append(pose)

                # get all image name in special sequence
                img_path = glob(f'{dataset_path}/{i:02}/image_2/*.png')
                img_path.sort()

                img_list.extend(img_path)
                for j in range(len(img_path)):
                    if j <= len(img_path) - self.seq and j % self.seq == 0:
                        start_idx.append(count)
                    count += 1
        
        res = poses[0]
        for i in range(1, len(poses)):
            res = np.concatenate((res, poses[i]), axis=0)

        return img_list, start_idx, res
    
    def __getitem__(self, index):
        # get start index
        idx = self.start_idx[index]

        idxs = []
        for id in range(idx, idx + self.seq + 1):
            idxs.append(id)

        imgs_1 = []
        imgs_2 = []
        labels = []
        for idx_1, idx_2 in zip(idxs[:-1], idxs[1:]):
            imgs_1.append(np.array(Image.open(self.img_list[idx_1]).resize((1280, 384))).astype(np.float32))
            imgs_2.append(np.array(Image.open(self.img_list[idx_2]).resize((1280, 384))).astype(np.float32))
            labels.append(self.poses[idx_1])

        sample = dict()
        sample['img_1'] = np.stack(imgs_1, 0)       # [T, H, W, C]
        sample['img_1'] = np.transpose(sample['img_1'], [0, 3, 1, 2])       # [T, C, H, W]

        sample['img_2'] = np.stack(imgs_2, 0)
        sample['img_2'] = np.transpose(sample['img_2'], [0, 3, 1, 2])   

        sample['label'] = np.array(labels)

        return sample            

    def __len__(self):
        return len(self.start_idx)

if __name__ == '__main__':
    dataset = DeepVODataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    for step, sample in enumerate(dataloader):
        print(step)
        print(sample['img_1'])