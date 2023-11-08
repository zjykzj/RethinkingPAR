# -*- coding: utf-8 -*-

"""
@Time    : 2023/11/6 14:42
@File    : baseline.py
@Author  : zj
@Description: 
"""

import os
import sys

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch
import torch.nn as nn

from backbone.resnet import resnet18, resnet34, resnet50, resnet101

backbone_dict = {
    'resnet34': 512,
    'resnet18': 512,
    'resnet50': 2048,
    'resnet101': 2048,
}


class Baseline(nn.Module):

    def __init__(self, backbone_type, num_attr):
        super().__init__()
        self.backbone_type = backbone_type
        self.num_attr = num_attr

        assert backbone_type in backbone_dict.keys(), backbone_type
        output_d = backbone_dict[backbone_type]
        if 'resnet18' == backbone_type:
            self.backbone = resnet18(pretrained=True)
        elif 'resnet50' == backbone_type:
            self.backbone = resnet50(pretrained=True)
        elif 'resnet101' == backbone_type:
            self.backbone = resnet101(pretrained=True)
        else:
            raise ValueError(f"{backbone_type} does not supports")

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(output_d, num_attr)

    def forward(self, x):
        x = self.backbone(x)
        # [N, C, H, W] -> [N, C, 1, 1] -> [N, C]
        x = self.pool(x).view(x.size(0), -1)
        # [N, C] -> [N, num_attr]
        x = self.fc(x)

        # outputs = torch.sigmoid(x)
        # return outputs
        return x


if __name__ == '__main__':
    model = Baseline('resnet18', 100)
    print(model)

    import torch

    data = torch.randn(1, 3, 224, 224)
    output = model(data)
    print(data.shape, output.shape)
