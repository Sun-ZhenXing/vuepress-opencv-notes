# 矩阵变换

[[TOC]]

## 1. 仿射变换

**仿射变换**（Affine Transformation）是一种二维平面几何变换，它保持了平行线的性质，但不一定保持长度和角度。仿射变换可以由以下几种基本变换组合得到：平移、缩放、旋转和错切。

### 1.1 公式

在二维空间中，仿射变换可以表示为一个 2x2 矩阵 $A$ 和一个 2x1 向量 $B$ 的形式：

$$
x' = Ax + B
$$

其中，$x$ 和 $x'$ 分别表示变换前后的二维坐标向量，具体形式如下：

$$
x' = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix} + \begin{bmatrix}
e \\
f
\end{bmatrix}
$$

在三维空间中，仿射变换可以表示为一个 3x3 矩阵 $A$ 和一个 3x1 向量 $B$ 的形式：

$$
x' = Ax + B
$$

其中，$x$ 和 $x'$ 分别表示变换前后的三维坐标向量，具体形式如下：

$$
x' = \begin{bmatrix}
a & b & c \\
d & e & f \\
g & h & i
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
z
\end{bmatrix} + \begin{bmatrix}
j \\
k \\
l
\end{bmatrix}
$$

### 1.2 原理

仿射变换的基本原理是通过矩阵乘法实现对坐标的变换。对于二维空间，2x2 矩阵 $A$ 负责实现缩放、旋转和错切变换，2x1 向量 $B$ 负责实现平移变换。对于三维空间，同样的原理适用于 3x3 矩阵 $A$ 和 3x1 向量 $B$。

### 1.3. 用途

仿射变换在许多领域都有广泛应用，例如计算机图形学、图像处理、机器人学和地理信息系统等。以下是一些常见的应用场景：

- 图像处理：仿射变换可以用于图像的变换操作，如旋转、缩放、平移以及透视矫正等。
- 计算机图形学：在 3D 渲染中，仿射变换可以用于实现场景中物体的变换，如旋转、缩放和平移等。
- 机器人学：仿射变换可用于表示机器人关节的运动和变换。
- 地理信息系统：仿射变换可用于地图投影和坐标转换等操作。

总之，仿射变换作为一种通用的几何变换，在许多领域都有重要的应用价值。

## 2. 投影变换

设 $(X,\, Y,\, Z)$ 和 $(x,\, y,\, z)$ 分别表示世界坐标系和相机坐标系中的点，投影变换可以表示为：

$$
\begin{bmatrix}
    X \\
    Y \\
    Z
\end{bmatrix} =
\begin{bmatrix}
    a_{11} & a_{12} & a_{13} \\
    a_{21} & a_{22} & a_{23} \\
    a_{31} & a_{32} & a_{33}
\end{bmatrix}
\begin{bmatrix}
    x \\
    y \\
    1
\end{bmatrix}
$$

可以得出

$$
\begin{cases}
    X &= a_{11}x + a_{12}y + a_{13} \\
    Y &= a_{21}x + a_{22}y + a_{23} \\
    Z &= a_{31}x + a_{32}y + a_{33}
\end{cases}
$$

投影后的点 $(x',\, y')$ 可以表示为：

$$
\begin{bmatrix}
    x' \\
    y'
\end{bmatrix} =
\begin{bmatrix}
    X / Z \\
    Y / Z
\end{bmatrix}
$$

透视变换可以通过四个点的变换前后位置来确定，在 OpenCV 中可以使用 `getPerspectiveTransform(srcQuad, dstQuad, mat)` 函数来计算透视变换矩阵，使用 `warpPerspective(src, dst, mat)` 函数来实现透视变换。
