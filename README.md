<!-- <div align="right">
  Language:
    🇺🇸
  <a title="Chinese" href="./README.zh-CN.md">🇨🇳</a>
</div> -->

<div align="center"><a title="" href="https://github.com/zjykzj/RethinkingPAR"><img align="center" src="assets/icons/RethinkingPAR.svg" alt=""></a></div>

<p align="center">
  «RethinkPAR» implements a strong PyTorch baseline for pedestrian attribute recognition 
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square" alt=""></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg" alt=""></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg" alt=""></a>
</p>

|                                             | Dataset |   Model  |   mA   |   Acc  |  Prec  |   Rec  |   F1   |
|:-------------------------------------------:|:-------:|:--------:|:------:|:------:|:------:|:------:|:------:|
| valencebond/Rethinking_of_PAR(Origin Paper) | PETA_zs | ResNet50 |  71.43 |  58.69 |  74.41 |  69.82 |  72.04 |
|                  This Repos                 | PETA_zs | ResNet50 | 70.374 | 59.106 | 75.239 | 69.822 | 72.429 |

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Latest News](#latest-news)
- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
  - [Train](#train)
  - [Eval](#eval)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## Latest News

* ***[2023/11/08][v0.1.1](https://github.com/zjykzj/RethinkingPAR/releases/tag/v0.1.1). Update Loss and Train.***
* ***[2023/11/07][v0.1.0](https://github.com/zjykzj/RethinkingPAR/releases/tag/v0.1.0). ResNet50 + PETA_zs.***

## Background

The paper [Rethinking of Pedestrian Attribute Recognition: A Reliable Evaluation under Zero-Shot Pedestrian Identity Setting](https://arxiv.org/abs/2107.03576) provides a detailed definition of existing research in the field of pedestrian attribute recognition, not only providing a clear definition of pedestrian attribute recognition, but also providing a new baseline method.

I tried the code repository provided in the paper, but there were several obvious issues, such as key dependency libraries not adapting to the latest version (unable to use Pytorch 2. x.x), and the overall implementation of the project being too heavy for further development and integration.

In order to facilitate better research and application of the methods proposed in this paper, I have implemented this warehouse to simplify training, evaluation, and prediction as much as possible.

## Installation

```shell
pip install -r requirements.txt
```

## Usage

### Train

```shell
CUDA_VISIBLE_DEVICES=0 python train.py ../datasets/PETA/ runs/r50_train_b64/ --backbone resnet50 --num_attr 32
```

### Eval

```shell
python eval.py ../datasets/PETA/ runs/r50_train_b64/rethinking_par-e95.pth 
```

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

* [valencebond/Rethinking_of_PAR](https://github.com/valencebond/Rethinking_of_PAR)
* [zjykzj/crnn-ctc](https://github.com/zjykzj/crnn-ctc)

## Contributing

Anyone's participation is welcome! Open an [issue](https://github.com/zjykzj/RethinkingPAR/issues) or submit PRs.

Small note:

* Git submission specifications should be complied
  with [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)
* If versioned, please conform to the [Semantic Versioning 2.0.0](https://semver.org) specification
* If editing the README, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme)
  specification.

## License

[Apache License 2.0](LICENSE) © 2023 zjykzj