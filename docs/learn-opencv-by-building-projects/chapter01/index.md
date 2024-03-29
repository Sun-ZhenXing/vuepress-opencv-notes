# 1. OpenCV 入门

<!-- markdownlint-disable MD045 -->

[[TOC]]

计算机视觉应用程序很有趣，而且很有用，但是其底层算法是计算密集型的。

OpenCV 库提供实时高效的计算机视觉算法，已经是该领域的事实标准。它可以在几乎所有平台上使用。

本节将主要关注下面几个主题：

- [x] 人类如何处理视觉数据，如何理解图像内容
- [x] OpenCV 能做什么，OpenCV 中可以用于实现这些目标的各种模块是什么
- [x] 如何在 Windows、Linux 和 Mac OS 上安装 OpenCV

## 1.1 了解人类视觉系统

人眼可以捕获视野内所有信息，例如 **颜色**、**形状**、**亮度**。

一些事实：

- 人眼的视觉系统对低频的内容更敏感，而高度纹理化的表面人眼不容易发现目标
- 人眼对于亮度的变化比对颜色更敏感
- 人眼对于运动的事物很敏感，即使没有被直接看到
- 人类擅长记忆对象的特征点，以便在下一次看见它的时候能够快速识别出来

**人类视觉系统**（HVS）模型的一些事实：

- 对颜色：$30\degree \sim 60\degree$
- 对形状：$5\degree \sim 30\degree$
- 对文本：$5\degree \sim 10\degree$

## 1.2 人类如何理解图像内容

人在看见一个椅子，立刻就能识别出它是椅子。而计算机执行这项任务缺异常困难，研究人员还在不断研究为什么计算机在这方面做的没有我们好。

人的视觉数据处理发生在腹侧视觉流中，它基本上是大脑中的一个区域层次结构，帮助我们识别对象。

人类在识别对象时产生了某种不变性。大脑会自动提取对象的特征并以某种方式储存起来。这种方式不受视角、方向、大小和光照等因素影响。

腹侧流更进一步是更复杂的细胞，它们被训练为识别更加复杂的对象。

### 为什么机器难以理解图像内容

机器并不容易做到这一点，如果换一个角度拍摄图像，图片将产生很大的不同。图片有很多变化，如形状、大小、视角、角度、照明、遮挡。计算机不可能记住对象的所有可能的特征，我们需要一些实用算法来处理。

当对象的一部分被遮挡时，计算机仍然无法识别它，计算机可能认为这是一个新对象。因此，我们要想构建底层功能模块，就需要组合不同的处理算法来形成更复杂的算法。

OpenCV 提供了很多这样的功能，它们经过高度优化，一旦了解了 OpenCV 的功能，就可以构建更加复杂的应用程序。

## 1.3 你能用 OpenCV 做什么

### 1.3.1 内置数据结构和输入/输出

`imgcodecs` 模块包含了图像的读取和写入。

`videoio` 模块处理视频的输入输出操作，可以从文件或者摄像头读取，甚至获取或设置帧率、帧大小等信息。

### 1.3.2 图像处理操作

`imgproc` 模块提供了图像处理操作，例如 **图像过滤**、**形态学操作**、**几何变换**、**颜色转换**、**图像绘制**、**直方图**、**形状分析**、**运动分析**、**特征检测** 等。

`ximgproc` 模块包含高级图像处理算法，可以用于如 **结构化森林的边缘检测**、**域变换滤波器**、**自适应流形滤波器** 等处理。

### 1.3.3 GUI

`highgui` 模块，用于处理 GUI。

### 1.3.4 视频分析

`video` 模块可以 **分析连续帧之间的运动**，**目标追踪**，创建 **视频监控模型** 等任务。

`videostab` 模块用于处理视频稳定性问题。

### 1.3.5 3D 重建

`calib3d` 计算 2D 图像中各种对象之间的关系，计算其 3D 位置，该模块也可用于摄像机校准。

### 1.3.6 特征提取

`features2d` 的 OpenCV 模块提供了特征检测和提取功能，例如 **尺度不变特征转换**（Scale Invariant Feature Transform, SIFT）算法、**加速鲁棒特征**（Speeded Up Robust Features, SURF）和 **加速分段测试特征**（Features From Accelerated Segment Test, FAST）等算法。

`xfeatures2d` 提供了更多特征提取器，其中一部分仍然处于实验阶段。

`bioinspired` 模块提供了受到生物学启发的计算机视觉模型算法。

### 1.3.7 对象检测

对象检测是检测图像中对象的位置，与对象类型无关。

`objdetect` 和 `xobjdetect` 提供了对象检测器框架。

### 1.3.8 机器学习

`ml` 模块提供了很多机器学习方法，例如 **贝叶斯分类器**（Bayes Classifier）、**K 邻近**（KNN）、支持向量机（SVM）、决策树（Decision Tree）、**神经网络**（Neural Network）等。

此外，它还包含了 **快速近似最临近搜索库**，即 `flann`，用于在大型数据集中进行快速最近近搜索的算法。

### 1.3.9 计算摄影

`photo` 和 `xphoto` 提供了计算摄影学有关的算法，包括 HDR 成像、图像补光、光场相机等算法。

`stitching` 模块提供创建全景图像的算法。

### 1.3.10 形状分析

`shape` 模块提供了提取形状、测量相似性、转换对象形状等操作相关的算法。

### 1.3.11 光流算法

光流算法用于在视频中跟踪连续帧中的特征。

`optflow` 模块用于执行光流操作，`tracking` 模块包含跟踪特征的更多算法。

### 1.3.12 人脸和对象识别

`face` 用于识别人脸的位置。

`saliency` 模块用于检测静态图像和视频中的显著区域。

### 1.3.13 表面匹配

`surface_matching` 用于 3D 对象的识别方法，以及 3D 特征姿势估计算法。

### 1.3.14 文本检测和识别

`text` 模块包含了含有文字的检测和识别的算法。

### 1.3.15 深度学习

`dnn` 模块包含了深度学习相关的内容，包括 TensorFlow、Caffe、ONNX 等模型的导入器和部分网络的推理。

## 1.4 安装 OpenCV

OpenCV 发行版下载：[OpenCV Releases](https://opencv.org/releases/)。

### 1.4.1 Windows

::: tip Windows 预编译版本

目前 OpenCV 提供 64 位不同版本的 OpenCV，这些不同版本的 OpenCV 预编译包是在不同版本的 Visual Studio 中编译的。如果你需要 32 位的 OpenCV 或者兼容更高版本的 VS，则需要自己编译源代码。

:::

例如，你将 OpenCV 安装到 `D:\OpenCV4.6\` 文件夹下。

可以使用下面的命令创建变量：

```bat
setx -m OPENCV_DIR D:\OpenCV4.6\opencv\build\x64\vc15
```

安装后，在路径（PATH）上加入以下路径 `%OPENCV_DIR%/bin`。

也可以将完整 `bin/` 路径加入 PATH：

```bat
D:\OpenCV4.6\opencv\build\x64\vc15\bin
```

### 1.4.2 Mac OS

需要安装 CMake，可以从官网查找发行版，如果不满足可以从源码开始编译，过程大致和 Linux 类似。

### 1.4.3 Linux

可以编译安装，可以先安装编译依赖项，然后下载构建即可。

在编译前需要预先安装一些依赖库，请检查最新版本要求。

```bash
# 去官网或者 GitHub 查找下载最新版
wget "https://github.com/opencv/opencv/<最新版>.tar.gz" -O opencv.tar.gz
wget "https://github.com/opencv/opencv_contrib/<最新版>.tar.gz" -O opencv_contrib.tar.gz
tar -zxvf opencv.tar.gz
tar -xzvf opencv_contrib.tar.gz
```

进入各自目录，然后选择合适的参数，使用 CMake 编译：

```bash
cmake -D<各种配置> ../
make -j 4
sudo make install
```

然后复制配置文件：

```bash
cp <路径>/build/lib/pkgconfig/opencv.pc /usr/local/lib/pkgconfig.opencv4.pc
```

## 1.5 总结

本章讨论了计算机视觉系统，以及人类如何处理视觉数据。解释了为什么机器难以做到这一点，以及在设计计算机视觉库时需要考虑的因素。

我们学习了 OpenCV 可以完成的工作，以及可用于完成这些任务的各种模块。最后学习了如何在各种操作系统中安装 OpenCV。
