
# Train

## Image_resize

```text
# Height/Width
Size_A: 256/192
Size_B: 256/128
```

## Train_transform

```text
# transform_A
            self.transform = transforms.Compose([
                transforms.Resize((height, width)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
# transform_B
            self.transform = transforms.Compose([
                transforms.Resize((height, width)),
                transforms.Pad(10),
                transforms.RandomCrop((height, width)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
```

## Results

|                      | Dataset |   Model  |   mA   |   Acc  |  Prec  |   Rec  |   F1   |
|:--------------------:|:-------:|:--------:|:------:|:------:|:------:|:------:|:------:|
| Size_A + Transform_A | PETA_zs | ResNet50 | 70.374 | 59.106 | 75.239 | 69.822 | 72.429 |
| Size_B + Transform_A | PETA_zs | ResNet50 | 70.427 | 58.909 | 74.963 | 69.597 | 72.180 |
| Size_B + Transform_B | PETA_zs | ResNet50 | 69.969 | 58.185 | 74.235 | 69.278 | 71.671 |

* It seems that `Pad+RandomCrop` have no effect
* `256x192` works better