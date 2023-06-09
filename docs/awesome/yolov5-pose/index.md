# 使用 YOLOv5 进行姿态估计和行为检测

[[TOC]]

## 1. 姿态估计和行为检测概述

**行为识别**（Action Recognition）是指对视频中人的行为动作进行识别。行为识别是一项具有挑战性的任务，受光照条件各异、视角多样性、背景复杂、类内变化大等诸多因素的影响。[^1]

[^1]: 一文了解通用行为识别 Action Recognition：了解及分类，<https://zhuanlan.zhihu.com/p/103566134>

对行为识别的研究可以追溯到 1973 年，Johansson 通过实验观察发现，人体的运动可以通过一些主要关节点的移动来描述[^2]。因此，只要 10-12 个关键节点的组合与追踪便能形成对诸多行为例如跳舞、走路、跑步等的刻画，做到通过人体关键节点的运动来识别行为。

[^2]: Johansson, G. Visual perception of biological motion and a model for its analysis. *Perception & Psychophysics* 14, 201–211 (1973). <https://doi.org/10.3758/BF03212378>

**姿态估计**（Pose Estimation）是指检测图像和视频中的人物形象的计算机视觉技术，可以确定某人的某个身体部位出现在图像中的位置，也就是在图像和视频中对人体关节的定位问题，也可以理解为在所有关节姿势的空间中搜索特定姿势。简言之，姿态估计的任务就是重建人的关节和肢干。[^3]

[^3]: 姿态估计与行为识别（行为检测、行为分类）的区别，<https://cloud.tencent.com/developer/article/2029260>

姿态估计可输出一个高维的姿态向量表示关节点的位置，即一整组关节点的定位，从图像背景中分离出人体前景，然后重建人物的关节、肢体，以便作为行为识别的输入，进行动作的识别，如跑步，跳跃等。

当我们使用姿态估计的结果时，行为识别可认为是典型的分类问题。姿态估计得到了特征点在图片中的位置信息，这些信息可全部进行归一化，然后利用最流行的分类器来对行为进行分类。

## 2. 姿态估计的方法

目前人体姿态估计总体分为 Top-down 和 Bottom-up 两种，与目标检测不同，无论是基于热力图或是基于检测器处理的关键点检测算法，都较为依赖计算资源，推理耗时略长，2022 年出现了以 YOLO 为基线的关键点检测器。[^4]

[^4]: *YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss*，<https://arxiv.org/abs/2204.06806>

在 ECCV 2022 和 CVPRW 2022 会议上，YoLo-Pose 和 KaPao 都基于流行的 YOLO 目标检测框架提出一种新颖的无热力图的方法[^4][^5]，YOLO 类型的姿态估计方法不使用检测器进行二阶处理，也不使用使用热力图拼接，虽然是一种暴力回归关键点的检测算法，但在处理速度上具有一定优势。

[^5]: *Rethinking Keypoint Representations: Modeling Keypoints and Poses as Objects for Multi-Person Human Pose Estimation*，<https://arxiv.org/abs/2111.08557>

对于人的姿势估计，它可以归结为一个单个类别检测器（对于人）。每个人有 $17$ 个关键点，而每个关键点又被确定为识别位置和置信度。因此，$17$ 个关键点有 $51$ 个元素与一个锚点（anchor）。因此，对于每个锚点需要预测 $51$ 个元素，预测框需要 $6$ 个元素。对于一个有 $n$ 个关键点的锚，整个预测向量被定义为

$$
P_v = \{
    C_x,\,C_y,\,W,\,H,\,\mathrm{box}_{conf},\,\mathrm{class}_{conf},\,K^1_x,\,K^1_y,\,K^1_{conf},\,\dots,\,K^n_x,\,K^n_y,\,K^n_{conf}
\}
$$

YOLO-Pose 使用的数据集是 Keypoints Labels of MS COCO 2017，数据集中每一行表示一个人的姿态标注。第一个值恒为 $0$，表示类别为人。后面的四个值分别是 $x,\,y$ 和宽高的归一化值，接下来是 $17$ 个关键点的位置。每一个关键点是一个长度为 $3$ 的数组，第一和第二个元素分别是 $x$ 和 $y$ 归一化坐标值，第三个元素是个标志位 $v$，$v$ 为 $0$ 时表示这个关键点没有标注（这种情况下 $x=y=v=0$），$v$ 为 $1$ 时表示这个关键点标注了但是不可见（被遮挡了），$v$ 为 $2$ 时表示这个关键点标注了同时也可见。

网络中每一个锚点（anchor）的输出值是 $P_v$，对于 YOLO，通常使用非极大值抑制来获取最终的输出结果。也就是说，我们最终会得到一个人的目标框和关键点信息。我们取所有关键点信息的归一化值来给下面的行为检测器使用。我们提取人的检测框，并使用检测框对 $17$ 个关键点进行归一化，这样我们就得到了 $51$ 维度的训练数据。

现在 YOLOv7 Pose[^6] 和 YOLOv8[^7] 都已经实现了这个算法，并且提供了相应的预训练模型。后续将提供相应的代码示例。

[^6]: YOLOv7-Pose，GitHub，<https://github.com/WongKinYiu/yolov7/tree/pose>

[^7]: YOLOv8，GitHub，<https://github.com/ultralytics/ultralytics>

## 2. YOLOv5 姿态估计

[下载 ONNX 预训练模型](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/latest/edgeai-yolov5/pretrained_models/models/keypoint/coco/edgeai-yolov5/yolov5s6_pose_640_ti_lite_54p9_82p2.onnx)，得到文件 `yolov5s6_pose_640_ti_lite_54p9_82p2.onnx`。

下面使用 ONNX Runtime 进行推理。

::: details 查看代码

@[code python](./src/main.py)

:::

如果我们需要对每个关键点基于检测框进行归一化，可以在 `post_process` 函数中添加如下代码：

```python
if kpts.size > 0:
    det_bbox = det_bboxes[0]
    x1, y1, x2, y2 = map(int, det_bbox)
    w, h = x2 - x1, y2 - y1
    kpts[0, 0::3] = (kpts[0, 0::3] - x1) / w
    kpts[0, 1::3] = (kpts[0, 1::3] - y1) / h
```

如果需要推理某个文件夹下的全部文件，修改 `build_train_data` 函数，最终会构建 CSV 文件。

## 3. 行为分类

有了关键点数据，我们就可以对行为进行分类。我们可以使用 [Kaggle 瑜伽姿态数据集](https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset)，这个数据集包含了 5 种不同的瑜伽姿势，每种姿势有 100~200 个样本[^8]。我们可以使用 SVM 分类器来对这些数据进行分类。

[^8]: 在 Python 中使用机器学习进行人体姿势估计，深度学习与计算机视觉——微信公众号，<https://mp.weixin.qq.com/s/D_sTpTp_pkLeO2nrcjgpaA>

SVM 分类器的工作流程如下：

1. 收集训练数据集：收集一组已经标记好的训练数据集，其中每个样本都有一个标签，表示它所属的类别。
2. 特征提取：从每个样本中提取出一组特征向量，用于描述该样本的特征。
3. 标准化：对特征向量进行标准化处理，使其在数值上具有相同的尺度。
4. 寻找最优超平面：通过求解一个优化问题，找到一个最优的超平面，使得该超平面能够将不同类别的样本分开，并且在两侧的分类边界上的距离最大。
5. 核函数选择：如果数据集不是线性可分的，需要使用核函数将数据映射到高维空间中，使其成为线性可分的。
6. 参数调优：选择合适的参数，如正则化参数和核函数参数，以达到更好的分类效果。
7. 模型评估：使用测试数据集对模型进行评估，检验其泛化能力。
8. 应用模型：将训练好的模型应用于新的未知数据进行分类。

依据上面的流程，我们设计的训练流程如下：

```mermaid
graph TD
    A(["摄像机"])
    B(["YOLOv7-Pose 检测器"])
    C(["PCA 降维"])
    D(["SVM 分类器"])
    E(["终端"])

    A -- 图像帧 --> B
    B -- 关键点数据 --> C
    C -- 低维数据 --> D
    D -- 行为类别 --> E
```

由摄像机提取的图像帧数据经过预处理后经过 YOLOv7-Pose 网络检测后，得到每一个图像的特征点数据，然后降维到低维度后训练 SVM 分类器，通过 SVM 分类器实现行为检测，从而判断具体行为。

下面是一个二分类的示例，用于分类摔倒和没有摔倒的图片，预处理方式相似，需要对检测框进行归一化，请参考上方代码，将数据保存为 CSV 文件。使用网格搜索查找最优参数，训练后绘制混淆矩阵，并打印准确率、精度、召回率和 F1 值。

::: details SVM 分类器示例

@[code python](./src/svm.py)

:::

本次测试每个类别 116 个样本，训练集 80%，测试集 20%，最终结果如下。

最优参数：

| C   | gamma | kernel |
| --- | ----- | ------ |
| 10  | 1     | `rbf`  |

| 参数表       | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| 0.0          | 0.91      | 0.95   | 0.93     | 21      |
| 1.0          | 0.96      | 0.92   | 0.94     | 26      |
| accuracy     | -         | -      | 0.94     | 47      |
| macro avg    | 0.93      | 0.94   | 0.94     | 47      |
| weighted avg | 0.94      | 0.94   | 0.94     | 47      |

混淆矩阵：

![matrix](./images/matrix.svg)
