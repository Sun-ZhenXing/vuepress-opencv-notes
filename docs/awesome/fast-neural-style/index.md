---
title: OpenCV 部署快速风格迁移
description: OpenCV 部署快速风格迁移
---

# OpenCV 部署快速风格迁移

[[TOC]]

## 1. 快速风格迁移简介

将一张图片的风格迁移到另一张图片上是耗时任务，但是在 2016 年 [*Perceptual Losses for Real-Time Style Transfer and Super-Resolution*](https://arxiv.org/abs/1603.08155) 论文的提出，实现了实时完成这项任务。

在 [此项目的官方网站](https://cs.stanford.edu/people/jcjohns/eccv16/) 上，可以查看论文和其效果，推荐阅读。

此项目的原始代码托管在 GitHub 上：[jcjohnson/fast-neural-style](https://github.com/jcjohnson/fast-neural-style)，可以下载其预训练权重直接部署。

## 2. OpenCV 部署 Torch 模型

直接下载模型：
- [instance_norm/candy.t7](http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/candy.t7)
- [instance_norm/la_muse.t7](http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/la_muse.t7)
- [instance_norm/mosaic.t7](http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/mosaic.t7)
- [instance_norm/feathers.t7](http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/feathers.t7)
- [instance_norm/the_scream.t7](http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/the_scream.t7)
- [instance_norm/udnie.t7](http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/udnie.t7)
- [eccv16/the_wave.t7](http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/eccv16/the_wave.t7)
- [eccv16/starry_night.t7](http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/eccv16/starry_night.t7)
- [eccv16/la_muse.t7](http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/eccv16/la_muse.t7)
- [eccv16/composition_vii.t7](http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/eccv16/composition_vii.t7)

然后我们新建 `main.py`，然后再任意找一张图片 `test.jpg`，将文件保存如下：

- `main.py`
- `models/`
    - `eccv16/`
        - `composition_vii.t7`
        - `la_muse.t7`
        - `starry_night.t7`
        - `the_wave.t7`
    - `instance_norm/`
        - `candy.t7`
        - `feathers.t7`
        - `la_muse.t7`
        - `mosaic.t7`
        - `the_scream.t7`
        - `udnie.t7`
- `test.jpg`

然后我们对每一个模型都进行推理测试：

@[code python](./src/main.py)

如果需要保存，可以使用下面的代码：

```python
out = (out * 255).clip(0, 255).astype(np.uint8)
cv2.imwrite('out.jpg', out)
```
