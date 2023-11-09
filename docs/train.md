
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

## Batch_size

```text
Batch_A: 64
    * LR: 1e-4
Batch_B: 256
    * LR: 4e-4
```

## Results

|                      | Dataset |   Model   |     mA     |    Acc     |    Prec    |    Rec     |     F1     |
|:--------------------:|:-------:|:---------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| Size_A + Transform_A | PETA_zs | ResNet50  |   70.374   |   59.106   |   75.239   |   69.822   |   72.429   |
| Size_B + Transform_A | PETA_zs | ResNet50  |   70.427   |   58.909   |   74.963   |   69.597   |   72.180   |
| Size_B + Transform_B | PETA_zs | ResNet50  |   69.969   |   58.185   |   74.235   |   69.278   |   71.671   |
| Size_A + Transform_A | PETA_zs | ResNet101 |   71.076   |   59.189   |   74.534   |   70.196   |   72.300   |
| Size_A + Transform_B | PETA_zs | ResNet101 | **71.980** | **59.809** | **75.486** | **70.583** | **72.952** |

* `Pad+RandomCrop` is not helpful for `ResNet50`, but it can improve the performance of `ResNet101`
* `256x192` works better than `256x128`

|         | Dataset |  Model   |   mA   |  Acc   |  Prec  |  Rec   |   F1   |
|:-------:|:-------:|:--------:|:------:|:------:|:------:|:------:|:------:|
| Batch_A | PETA_zs | ResNet50 | 70.374 | 59.106 | 75.239 | 69.822 | 72.429 |
| Batch_B | PETA_zs | ResNet50 | 70.512 | 59.106 | 75.327 | 69.551 | 72.324 |

* Increasing `batch_size` has no effect on training results