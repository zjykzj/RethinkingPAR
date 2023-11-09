# -*- coding: utf-8 -*-

"""
@Time    : 2023/11/6 11:18
@File    : dataset.py
@Author  : zj
@Description: 
"""

import os.path
import pickle

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class RethinkingPARDataset(Dataset):
    """
    The architecture of the PETA dataset is as follows:
    .
    ├── dataset_zs_run0.pkl
    ├── images/
        00001.png
        00002.png
        ...

    """

    def __init__(self, data_root, dataset='PETA', split="trainval", height=256, width=192):
        assert dataset == 'PETA', "Now, Only support PETA_sz datast"
        assert os.path.isdir(data_root), data_root

        self.data_root = data_root
        self.split = split
        self.height = height
        self.width = width

        self.image_root = os.path.join(self.data_root, "images")
        pkl_path = os.path.join(data_root, "dataset_zs_run0.pkl")
        assert os.path.isfile(pkl_path), pkl_path
        with open(pkl_path, 'rb') as f:
            dataset_info = pickle.load(f)

        assert 'label_idx' in dataset_info.keys(), dataset_info.keys()
        # PETA_zs use 35 attributes to train/eval
        self.eval_attr_idx = dataset_info.label_idx.eval
        self.attr_name = [dataset_info.attr_name[i] for i in self.eval_attr_idx]

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'
        self.image_idx_list = dataset_info.partition[split]
        if isinstance(self.image_idx_list, list):
            self.image_idx_list = self.image_idx_list[0]  # default partition 0
        self.image_name_list = [dataset_info.image_name[i] for i in self.image_idx_list]

        # label in PETA: [19000, 105]
        attr_label = dataset_info.label
        attr_label[attr_label == 2] = 0
        # attr_label: [19000, 35]
        attr_label = attr_label[:, self.eval_attr_idx]
        self.label_list = attr_label[self.image_idx_list]  # [:, [0, 12]]

        if 'train' in split:
            self.transform = transforms.Compose([
                transforms.Resize((height, width)),
                # Better in ResNet101
                # transforms.Pad(10),
                # transforms.RandomCrop((height, width)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((height, width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        img_name, label = self.image_name_list[index], self.label_list[index]

        img_path = os.path.join(self.image_root, img_name)
        image = Image.open(img_path)
        if self.transform is not None:
            image = self.transform(image)

        target = torch.from_numpy(label.astype(np.float32))
        # if self.target_transform:
        #     gt_label = gt_label[self.target_transform]

        return image, target

    def __len__(self):
        return len(self.image_name_list)


if __name__ == '__main__':
    data_root = "/mnt/c/zj/repos/Rethinking_of_PAR/data/PETA/"
    dataset = RethinkingPARDataset(data_root, split='trainval')
    print(dataset, len(dataset))
    print(dataset.attr_name)

    for _ in range(5):
        index = np.random.choice(range(len(dataset)))
        img, gt_label = dataset.__getitem__(index)
        print(index, img.shape, type(img))
        print(gt_label, len(gt_label))
