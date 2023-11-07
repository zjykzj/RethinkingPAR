# -*- coding: utf-8 -*-

"""
@Time    : 2023/11/7 20:26
@File    : predict.py
@Author  : zj
@Description: 
"""

import argparse
import os

import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

from utils.model.baseline import Baseline, backbone_dict

# cp assets/fonts/simhei.ttf /usr/share/fonts/truetype/noto/
# rm -rf ~/.cache/matplotlib/*
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

PETA_zs_NAMES = ['accessoryHat', 'accessoryMuffler', 'accessoryNothing', 'accessorySunglasses', 'hairLong',
                 'upperBodyCasual', 'upperBodyFormal', 'upperBodyJacket', 'upperBodyLogo', 'upperBodyPlaid',
                 'upperBodyShortSleeve', 'upperBodyThinStripes', 'upperBodyTshirt', 'upperBodyOther', 'upperBodyVNeck',
                 'lowerBodyCasual', 'lowerBodyFormal', 'lowerBodyJeans', 'lowerBodyShorts', 'lowerBodyShortSkirt',
                 'lowerBodyTrousers', 'footwearLeatherShoes', 'footwearSandals', 'footwearShoes', 'footwearSneaker',
                 'carryingBackpack', 'carryingOther', 'carryingMessengerBag', 'carryingNothing', 'carryingPlasticBags',
                 'personalLess30', 'personalLess45', 'personalLess60', 'personalLarger60', 'personalMale']


def parse_opt():
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('image_path', metavar='IMAGE', type=str, help='path to image path')
    parser.add_argument('save_dir', metavar='DST', type=str, help='path to save dir')

    parser.add_argument('pretrained', metavar='PRETRAINED', type=str, default="rethinking_par-e100.pth",
                        help='path to pretrained model')
    parser.add_argument('--backbone', metavar='BACKBONE', type=str, default='resnet50', choices=backbone_dict.keys(),
                        help='model architecture: ' + ' | '.join(backbone_dict.keys()) + ' (default: resnet50)')
    parser.add_argument('--num_attr', metavar='ATTR', type=int, default=35,
                        help='number of attributes. Default: 35 attributes for PETA_zs')

    args = parser.parse_args()
    print(f"args: {args}")
    return args


@torch.no_grad()
def predict(opt):
    image_path, pretrained, save_dir, backbone, num_attr = opt.image_path, opt.pretrained, opt.save_dir, opt.backbone, opt.num_attr

    model = Baseline(backbone, num_attr)
    print(f"Loading Baseline pretrained: {pretrained}")
    ckpt = torch.load(pretrained, map_location='cpu')
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Data
    height = 256
    width = 192
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path)
    data = transform(image)

    # Infer
    data = data.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(data).cpu()
        output = torch.sigmoid(outputs).numpy()[0]

    thresh = 0.5
    pred_label = output > thresh
    pred_names = np.array(PETA_zs_NAMES)[pred_label]
    pred_str = '\n'.join(pred_names)

    # Draw
    title = f"Pred: \n{pred_str}"
    print(title)

    plt.figure()
    plt.imshow(image)
    plt.text(-60, 100, title, fontsize=10, color='green')

    image_name = os.path.basename(image_path)
    plt.savefig(os.path.join(save_dir, f"baseline_{image_name}"))
    plt.show()


if __name__ == '__main__':
    opt = parse_opt()
    predict(opt)
