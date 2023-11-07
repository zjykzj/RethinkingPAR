# -*- coding: utf-8 -*-

"""
@Time    : 2023/11/7 15:59
@File    : eval.py
@Author  : zj
@Description: 
"""

import argparse

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils.model.baseline import Baseline, backbone_dict
from utils.dataset import RethinkingPARDataset
from utils.evaluator import Evaluator


def parse_opt():
    parser = argparse.ArgumentParser(description='Eval CRNN with EMNIST')
    parser.add_argument('val_root', metavar='DIR', type=str, help='path to val dataset')

    parser.add_argument('pretrained', metavar='PRETRAINED', type=str, default="runs/emnist_ddp/crnn-emnist-e100.pth",
                        help='path to pretrained model')

    parser.add_argument('--backbone', metavar='BACKBONE', type=str, default='resnet18', choices=backbone_dict.keys(),
                        help='model architecture: ' + ' | '.join(backbone_dict.keys()) + ' (default: resnet18)')
    parser.add_argument('--num_attr', metavar='ATTR', type=int, default=35,
                        help='number of attributes. Default: 35')

    args = parser.parse_args()
    print(f"args: {args}")
    return args


@torch.no_grad()
def val(opt):
    val_root, pretrained, backbone, num_attr = opt.val_root, opt.pretrained, opt.backbone, opt.num_attr

    model = Baseline(backbone, num_attr)
    print(f"Loading CRNN pretrained: {pretrained}")
    ckpt = torch.load(pretrained, map_location='cpu')
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    val_dataset = RethinkingPARDataset(val_root, dataset='PETA', split="test", height=256, width=192)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=False,
                                pin_memory=True)
    emnist_evaluator = Evaluator()

    pbar = tqdm(val_dataloader)
    for idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images).cpu()

        acc = emnist_evaluator.update(outputs.numpy(), targets.numpy())
        info = f"Batch:{idx} ACC:{acc * 100:.3f}"
        pbar.set_description(info)
    acc = emnist_evaluator.result()
    print(f"ACC:{acc * 100:.3f}")


if __name__ == '__main__':
    opt = parse_opt()
    val(opt)
