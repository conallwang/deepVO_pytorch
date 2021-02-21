from torch.utils.data import Dataset

from glob import glob
from config import *

from PIL import Image

import numpy as np

class DeepVODataset(Dataset):
    def __init__(self, seq=2, interval=1, phase='train'):
        super().__init__()

        self.seq = 2
        self.interval = 1
        self.phase = phase

        self.img_list, self.start_idx = self.load_data()

    def load_data(self):
        start_idx = []
        img_list = []
        poses = []
        count = 0

        if self.phase == 'train':
            # choose sequence 00, 02, 08, 09
            for i in [0, 2, 8, 9]:
                # get all image name in special sequence
                img_path = glob(f'{dataset_path}/{i:02}/image_2/*.png')
                img_path.sort()

                img_list.extend(img_path)
                l = -self.seq
                for j in range(len(img_path)):
                    if j <= len(img_path) - self.seq and j - l >= self.seq:
                        start_idx.append(count)
                        l = j
                    count += 1
        else:
            # choose sequence 03, 04, 05, 06, 07, 10
            for i in [3, 4, 5, 6, 7, 10]:
                pass
        
        return img_list, start_idx
    
    def __getitem__(self, index):
        # get start index
        idx = self.start_idx[index]

        imgs = []
        for img_path in self.img_list[idx:idx+self.seq+1]:
            imgs.append(np.array(Image.open(img_path).resize((1280, 384))))
        
        imgs_1 = []
        imgs_2 = []
        labels = []
        for img_1, img_2 in zip(imgs[-1:], imgs[1:]):
            pass

    def __len__(self):
        pass

if __name__ == '__main__':
    dataset = DeepVODataset()
    print(len(dataset.img_list))
    # print(dataset.start_idx)