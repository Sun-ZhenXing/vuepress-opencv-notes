---
title: 第 3 章：核心操作
description: OpenCV 中的核心操作
---

# 3. 核心操作

[[TOC]]

## 3.1 图片

::: details 学习目标

- 获取像素值并修改
- 获取图像的属性
- 获取图像的 ROI
- 图像通道的拆分及合并
- 图像的填充
- 函数
    - 学习使用 NumPy 的函数库处理图像

:::

::: info NumPy

几乎所有这些操作与 NumPy 的关系都比与 OpenCV 的关系更加紧密，因此熟练 NumPy 可以帮助我们写出性能更好的代码。

:::

### 3.1.1 获取并修改像素值

我们注意到，一张打开的图片就是 `np.ndarray` 对象，可以使用 NumPy 对图像进行任意的操作：

```python
img = cv2.imread('messi5.jpg')
print(img[10, 10])

img[100, 100] = 255, 255, 255
```

::: warning 大量修改像素操作

NumPy 是经过优化了的进行快速矩阵运算的软件包。所以我们不推荐
逐个获取像素值并修改，这样会很慢，能有矩阵运算就不要用循环。

:::

更优雅的方法：

```python
img: np.ndarray = cv2.imread('messi5.jpg')

img.item(10, 10, 2)
img.itemset((10, 10, 2), 100)
```

### 3.1.2 获取图像属性

可以参考 `np.ndarray` 属性：
- `img.shape` 形状，即一个包含行、列和通道的元组
- `img.size` 元素的数目，即像素乘通道的总数
- `img.dtype` 元素类型

### 3.1.3 图像 ROI

*@def* **ROI**（Region Of Interest），即感兴趣区域，对 ROI 的操作通常指的是对图像的特定区域进行操作。

```python
ball = img[280:340, 330:390]
img[273:333, 100:160] = ball
```

### 3.1.4 拆分及合并图像通道

```python
b, g, r = cv2.split(img)
img = cv2.merge(b, g, r)
```

使用索引更加快速、简单：

```python
b = img[:, :, 0]
img[:, :, 2] = 0
```

### 3.1.5 为图像扩边（填充）

如果你想在图像周围创建一个边，就像相框一样，你可以使用 `cv2.copyMakeBorder()` 函数。这经常在卷积运算或 `0` 填充时被用到。

| 语法 | `cv2.copyMakeBorder(src, top, bottom, left, right, borderType, dst=..., value=...)` |
| ---- | ----------------------------------------------------------------------------------- |

这个函数包括如下参数：
- `src` 输入图像
- `top, bottom, left, right` 对应边界的像素数目
- `borderType` 要添加那种类型的边界，类型如下
    - `cv2.BORDER_CONSTANT` 添加有颜色的常数值边界，还需要下一个参数 `value`
    - `cv2.BORDER_REFLECT` 边界元素的镜像。比如 `fedcba|abcdefgh|hgfedcb`
    - `cv2.BORDER_REFLECT_101` 或 `cv2.BORDER_DEFAULT` 跟上面一样，但稍作改动。例如 `gfedcb|abcdefgh|gfedcba`
    - `cv2.BORDER_REPLICATE` 重复最后一个元素。例如 `aaaaaa|abcdefgh|hhhhhhh`
    - `cv2.BORDER_WRAP` 就像这样 `cdefgh|abcdefgh|abcdefg`
- `value` 边界颜色，如果边界的类型是 `cv2.BORDER_CONSTANT`

为了更好的理解这几种类型请看下面的演示程序：

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

BLUE = 255, 0, 0

img: np.ndarray = cv2.imread('messi5.jpg')

replicate = cv2.copyMakeBorder(
    img, 10, 10, 10, 10, cv2.BORDER_REPLICATE)

reflect = cv2.copyMakeBorder(
    img, 10, 10, 10, 10, cv2.BORDER_REFLECT)

reflect_101 = cv2.copyMakeBorder(
    img, 10, 10, 10, 10, cv2.BORDER_REFLECT_101)

wrap = cv2.copyMakeBorder(
    img, 10, 10, 10, 10, cv2.BORDER_WRAP)

constant = cv2.copyMakeBorder(
    img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=BLUE)

plt.subplot(231)
plt.imshow(img[:, :, ::-1], 'gray')
plt.title('original')

plt.subplot(232)
plt.imshow(replicate[:, :, ::-1], 'gray')
plt.title('replicate')

plt.subplot(233)
plt.imshow(reflect[:, :, ::-1], 'gray')
plt.title('reflect')

plt.subplot(234)
plt.imshow(reflect_101[:, :, ::-1], 'gray')
plt.title('reflect_101')

plt.subplot(235)
plt.imshow(wrap[:, :, ::-1], 'gray')
plt.title('wrap')

plt.subplot(236)
plt.imshow(constant[:, :, ::-1], 'gray')
plt.title('constant')

plt.show()
```

## 3.2 图像上的算术运算

::: details 学习目标

- 图像上的加法、减法、位运算等
- 图像混合
- 函数
    - `cv2.add()` 加法
    - `cv2.addWeighted()` 带权加法，即图像混合
    - `cv2.bitwise_not()` 按位翻转
    - `cv2.bitwise_and()` 按位与

:::

### 3.2.1 图像加法

::: warning 加法操作

OpenCV 中的加法与 NumPy 的加法是有所不同的。OpenCV 的加法
是一种饱和操作，而 NumPy 的加法是一种模操作。

:::

例如，在 `dtype=np.uint8` 的情况下
```python
import cv2
import numpy as np

UINT8 = np.uint8

x = UINT8([250])
y = UINT8([10])

print(cv2.add(x, y))    # [[255]]
print(x + y)            # [4]
```

### 3.2.2 图像混合

图像混合，其实也是加法，但是不同的是两幅图像的权重不同，这就会给人一种混合或者透明的感觉。图像混合的计算公式如下

$$
g(x) = (1 - \alpha)f_0(x) + \alpha f_1(x)
$$

`cv2.addWeighted()` 的混合公式

$$
\mathrm{dst} = \alpha \cdot \mathrm{img_1} +
\beta \cdot \mathrm{img_2} + \gamma
$$

```python
import cv2

img1 = cv2.imread('t1.jpg')
img2 = cv2.imread('t2.jpg')

dst = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 3.2.3 按位运算

这里包括的按位操作有：AND，OR，NOT，XOR 等。当我们提取图像的一部分，选择非矩形 ROI 时这些操作会很有用。下面的例子就是教给我们如何改变一幅图的特定区域。

我想把 OpenCV 的标志放到另一幅图像上。如果我使用加法，颜色会改变，如果使用混合，会得到透明效果，但是我不想要透明。如果他是矩形我可以象上一章那样使用 ROI。但是他不是矩形。但是我们可以通过下面的按位运算实现：

```python
import cv2
import numpy as np

## 加载图像
img1: np.ndarray = cv2.imread('roi.jpg')
img2: np.ndarray = cv2.imread('opencv_logo.png')

## 选择 logo 的 ROI
rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]

## 创建一个遮罩 mask，并创建一个翻转的遮罩
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 175, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

## 取 ROI 中与 mask 中不为零的值对应的像素的值，其他值为 0
## 其中的 mask= 不能忽略
img1_bg = cv2.bitwise_and(roi, roi, mask=mask)

## 取 ROI 中与 mask_inv 中不为零的值对应的像素的值，其他值为 0。
img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)

dst = cv2.add(img1_bg, img2_fg)
img1[0:rows, 0:cols] = dst
cv2.imshow('res', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 3.3 程序性能检测及优化

::: details 学习目标

- 检测程序的效率
- 一些能够提高程序效率的技巧
- 函数
    - `cv2.getTickCount()`
    - `cv2.getTickFrequency()`
- 扩展
    - IPython
    - 模块 `time` 和 `timeit`

:::

### 3.3.1 使用 OpenCV 检测程序效率

可用函数：
- `cv2.getTickCount()` 函数返回从参考点到这个函数被执行的时钟数。所以当你在一个函数执行前后都调用它的话，你就会得到这个函数的执行时间
- `cv2.getTickFrequency()` 返回时钟频率，或者说每秒钟的时钟数。所以你可以按照下面的方式得到一个函数运行了多少秒

```python
import cv2
import numpy as np
e1 = cv2.getTickCount()

## 此处是代码
e2 = cv2.getTickCount()
time = (e2 - e1) / cv2.getTickFrequency()
```

例如，测试代码：

```python
import cv2
import numpy as np

img1 = cv2.imread('t1.jpg')

e1 = cv2.getTickCount()
for i in range(5, 49, 2):
    img1 = cv2.medianBlur(img1, i)
e2 = cv2.getTickCount()
t = (e2 - e1) / cv2.getTickFrequency()
print(t)
```

### 3.3.2 OpenCV 中的默认优化

OpenCV 中的很多函数都被优化过（使用 SSE2，AVX 等）。你可以使用函数 `cv2.useOptimized()` 来查看优化是否被开启了，使用函数 `cv2.setUseOptimized()` 来开启优化。

使用 IPython 的魔术命令 `%timeit` 测试代码的：

```python
%timeit res = cv2.medianBlur(img, 49)
```

中值滤波是被 SIMD 优化的，如果关闭优化进行测试，发现结果不同。

### 3.3.3 在 IPython 中检测程序效率

当你想知道哪些程序执行得更快，例如：

```python
x = 5; y = x ∗ ∗2
x = 5; y = x ∗ x
x = np.uint([5]); y = x ∗ x
y = np.squre(x)
```

你可以使用上面的方法，`%timeit`。

例如，可以比较 `cv2.countNonZero()` 和 `np.count_nonzero()` 的效率差别

```python
%timeit z = cv2.countNonZero(img)
%timeit z = np.count_nonzero(img)
```

::: info OpenCV 与 NumPy

一般情况下 OpenCV 的函数要比 NumPy 函数快。所以对于相同的操作最好使用 OpenCV 的函数。当然也有例外，尤其是当使用 NumPy 对视图（而非复制）进行操作时。

:::

### 3.3.4 更多 IPython 的魔法命令

还有几个魔法命令可以用来检测程序的效率、内存使用等。

常见魔法命令：
- 使用 `%magic` 可以直接调出有关于魔法命令的详细说明文档，这就等同于 `vim` 编辑器中的 `:help` 一样
- 你只是想列出哪些魔法命令我们可以使用，那么可以直接调用 `%lsmagic` 进行输出
- 如果只是查看具体某个魔法命令的用法，那么可以直接在魔法命令之后接上一个 `?` 问号，类似于这样 `%run?`

### 3.3.5 效率优化技术

算法优化原则：
1. 算法中尽量使用向量操作，因为 NumPy 和 OpenCV 都对向量操作进行了优化
2. 利用高速缓存一致性

::: info 有用的资料

Python 速度优化技巧 [官方文档](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)。

NumPy 优化技巧 [scipy-lectures.org: NumPy 进阶](http://scipy-lectures.org/advanced/advanced_numpy/index.html#advanced-numpy)。

:::

## 3.4 OpenCV 中的数学工具

***TODO***

其余的内容将整理到其他地方。
