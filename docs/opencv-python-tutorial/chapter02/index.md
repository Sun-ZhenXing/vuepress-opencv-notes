# 2. OpenCV 中的 GUI 特性

[[TOC]]

## 2.1 图片

::: details 学习目标

- 读取、显示和保存图像
- 函数
    - `cv2.imread()` 读取图像
    - `cv2.imshow()` 显示图像
    - `cv2.imwrite()` 保存图像
    - `cv2.waitKey()` 等待读入一个键盘值
    - `cv2.destroyAllWindows()` 关闭所有窗体
    - `cv2.namedWindow()` 创建命名窗体
- 如何使用 Matplotlib 显示图片，并解决颜色错乱的问题

:::

### 2.1.1 读取一张图片

```python
import numpy as np
import cv2

img = cv2.imread('messi5.jpg', 0)
```

::: warning 读取错误

即使图像的路径是错的，OpenCV 也不会提醒你的，结果是 `None`。

:::

| 参数                        | 含义                                                 |
| --------------------------- | ---------------------------------------------------- |
| `cv2.IMREAD_COLOR = 1`      | 读入一副彩色图像。图像的透明度会被忽略，这是默认参数 |
| `cv2.IMREAD_UNCHANGED = -1` | 读入一幅图像，并且包括图像的 Alpha 通道              |
| `cv2.IMREAD_GRAYSCALE = 0`  | 以灰度模式读入图像                                   |

### 2.1.2 显示图像

```python
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

::: tip 建议

你也可以先创建一个窗口，之后再加载图像。这种情况下，你可以决定窗口是否可以调整大小。使用到的函数是 `cv2.namedWindow()`，此时默认值为 `cv2.WINDOW_AUTOSIZE = 1`，可以设置为 `cv2.WINDOW_NORMAL = 0`。

:::

命名窗口的使用方式如下：

```python
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
```

### 2.1.3 保存图像

```python
cv2.imwrite('messigray.png', img)
```

### 2.1.4 综合示例

以灰度模式打开图像，按 `s` 保存，`ESC` 退出则不保存。

```python
import cv2

img = cv2.imread('messi5.jpg', 0)
cv2.imshow('image', img)
k = cv2.waitKey(0) & 0xff
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('messigray.png', img)
    cv2.destroyAllWindows()
```

### 2.1.5 使用 Matplotlib

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('cv2test.png', 0)
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([])
plt.yticks([])
plt.show()
```

::: warning 不正确的颜色空间

彩色图像使用 OpenCV 加载时是 BGR 模式。但是 Matplotib 是 RGB 模式。所以彩色图像如果已经被 OpenCV 读取，那它将不会被 Matplotib 正确显示。

颜色转换参考 [Stack Overflow](https://stackoverflow.com/questions/15072736/extracting-a-region-from-an-image-using-slicing-in-python-opencv/15074748#15074748)，也可以使用 `img = img[..., ::-1]` 转换图像颜色空间。

:::

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('cv2test.png')
b, g, r = cv2.split(img)
img2 = cv2.merge([r, g, b])
plt.subplot(121)

# 错误的颜色
plt.imshow(img)
plt.subplot(122)

# 正确的颜色
plt.imshow(img2)
plt.show()

# 正确的颜色
cv2.imshow('bgr image', img)

# 错误的颜色
cv2.imshow('rgb image', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 2.2 视频

::: details 学习目标

- 读取、显示和保存视频
- 从摄像头读取并显示视频
- 函数
    - `cv2.VideoCapture()` 读取摄像头
    - `cv2.VideoWrite()` 写入视频
    - `cv2.cvtColor()` 转换颜色空间
    - `cv2.flip()` 翻转图像

:::

### 2.2.1 捕获摄像头

`cv2.VideoCapture(0)` 捕获第一个摄像头，`cap.read()` 读取图像。

```python
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

如果可能打开失败时，使用 `cap.isOpened()` 判断是否打开成功。

你可以使用函数 `cap.get(propId)` 来获得视频的一些参数信息。这里 `propId` 每一个数代表视频的一个属性，参见 [OpenCV 官方文档：`cv::VideoCaptureProperties`](https://docs.opencv.org/4.7.0/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d)。

其中的一些值可以使用 `cap.set(propId, value)` 来修改，`value` 就是你想要设置成的新值。例如，可以使用 `cap.get(3)` 和 `cap.get(4)` 来查看每一帧的宽和高。

### 2.2.2 从文件中播放视频

输入文件名，然后播放视频文件的内容：

```python
import cv2

video_path = input('video path:')
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 可以翻转过来播放
    # frame = cv2.flip(frame, 0)
    cv2.imshow('frame', frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

::: warning 依赖安装

默认情况下，OpenCV 安装时就包含了合适的依赖，如 FFmpeg。但是可能和你的电脑驱动不匹配，导致无法打开摄像头、无法保存视频等问题。这时候你需要选择合适的版本或更新你电脑中的驱动。

:::

### 2.2.3 保存捕获的视频

使用 `cv2.cv.FOURCC(*'MJPG')` 保存，`cv2.VideoWriter()` 用于写入视频。

```python
import cv2

cap = cv2.VideoCapture(0)

# 定义 codec 并创建 VideoWriter 对象
fourcc = cv2.cv.FOURCC(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame, 0)

        # 写入裁剪的帧
        out.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# 释放内存
cap.release()
out.release()
cv2.destroyAllWindows()
```

编码指南：
- `X264` 适合小尺寸视频
- `MJPG` 适合大尺寸视频
- `XVID` 适合自适应
- `DIVX` 适合 Windows 系统
- `WMV1` 微软开发的一种视频格式，属于 WMV
- `WMV2` 微软开发的一种视频格式，属于 WMV

::: info 编码

如果想查找编码的详细信息，参考 [fourcc.org](https://www.fourcc.org/codecs.php)。

:::

## 2.3 绘图函数

::: details 学习目标

- 绘制不同的几何图形
- 函数
    - `cv2.line()` 画线
    - `cv2.circle()` 画圈
    - `cv2.rectangle()` 矩形
    - `cv2.ellipse()` 椭圆形
    - `cv2.putText()` 绘制文本

:::

### 2.3.1 画线

```python
import numpy as np
import cv2

# 黑色的背景图
img = np.zeros((512, 512, 3), np.uint8)

# 蓝色的线，5px
cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)

winname = 'example'
cv2.namedWindow(winname)
cv2.imshow(winname, img)
cv2.waitKey(0)
cv2.destroyWindow(winname)
```

::: info 擦除图像

可以使用 `img = cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)` 来获得新的图像。

:::

### 2.3.2 画矩形

```python
cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
```

### 2.3.3 画圆

```python
cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)
```

### 2.3.4 画椭圆

```python
cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)
```

### 2.3.5 画多边形

```python
pts = np.array(
    [[10, 5],
     [20, 30],
     [70, 20],
     [50, 10]],
    np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(img, [pts], True, (255, 255, 255), 1)
```

如果 `cv2.polylines()` 第三个参数是 `False`，我们得到的多边形是不闭合的（首尾不相连）。

::: tip 绘制多条线

`cv2.polylines()` 可以被用来画很多条线。只需要把想要画的线放在一个列表中，将这个列表传给函数就可以了。每条线都会被独立绘制。这会比用 `cv2.line()` 一条一条的绘制要快一些。

:::

### 2.3.6 在图片上添加文字

设置参数：
- 你要绘制的文字
- 你要绘制的位置
- 字体类型（通过查看 `cv2.putText()` 的文档找到支持的字体）
- 字体的大小
- 文字的一般属性如颜色，粗细，线条的类型等。为了更好看一点推荐使用 `linetype = cv2.LINE_AA`

```python
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2)

winname = 'example'
cv2.namedWindow(winname)
cv2.imshow(winname, img)
cv2.waitKey(0)
cv2.destroyWindow(winname)
```

::: warning 绘制中文

OpenCV 目前并不支持在图片上绘制中文，不过 OpenCV Contrib 模块中的 `freetype` 模块提供了字体支持，用于代替 `cv2.putText()`。

但是，直接安装 `opencv-contrib-python` 可能并不包含 `freetype` 模块。官方构建发行包时默认不构建 `freetype` 模块，具体参考 [GitHub Issue](https://github.com/opencv/opencv-python/issues/243)，如果有需要可以下载源码进行构建，CMake 构建参数为 `-DWITH_FREETYPE=ON`。

如果希望获取构建时信息，使用

```python
print(cv2.getBuildInformation())
```

如果已经有 `freetype` 模块，可以使用：

```python
import cv2
import numpy as np

img = np.zeros((300, 512, 3), dtype=np.uint8)

ft = cv2.freetype.createFreeType2()
ft.loadFontData(fontFileName='/path/to/font.ttf', id=0)
ft.putText(
    img=img,
    text='这是一段中文',
    org=(15, 70),
    fontHeight=30,
    color=(255, 255, 255),
    thickness=-1,
    line_type=cv2.LINE_AA,
    bottomLeftOrigin=True
)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.imwrite('image.png', img)
```

:::

### 2.3.7 绘图函数总结

参数列表：
- `img`：你想要绘制图形的那幅图像
- `color`：形状的颜色。以 RGB 为例，需要传入一个元组，例如：`(255, 0, 0)` 代表蓝色。对于灰度图只需要传入灰度值
- `thickness`：线条的粗细。如果给一个闭合图形设置为 `-1`，那么这个图形就会被填充。默认值是 `1`
- `linetype`：线条的类型，8 连接，抗锯齿等。默认情况是 8 连接。`cv2.LINE_AA` 为抗锯齿，这样看起来会非常平滑

语法总结：
- `line(img, pt1, pt2, color, thickness=..., lineType=..., shift=...)`
- `rectangle(img, pt1, pt2, color, thickness=..., lineType=..., shift=...)`
- `circle(img, center, radius, color, thickness=..., lineType=..., shift=...)`
- `ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness=..., lineType=..., shift=...)`
- `polylines(img, pts, isClosed, color, thickness=..., lineType=..., shift=...)`

## 2.4 鼠标事件

::: details 学习目标

- 学习使用 OpenCV 处理鼠标事件
- 函数
    - `cv2.setMouseCallback()` 设置鼠标回调函数

:::

### 2.4.1 双击的地方绘制圆

查看有哪些事件受支持：

```python
import cv2

def print_const(prefix: str) -> None:
    prefix += '_'
    names = (
        '{:22} = {:5}'.format(key, val)
        for key, val in vars(cv2).items()
        if key.startswith(prefix)
    )
    print(*names, sep='\n')

print_const('EVENT')
```

`ESC` 退出，双击的地方绘制圆，不要按 `X` 关闭窗口，否则会无法绘制：

```python
import cv2
import numpy as np

# 鼠标回调函数
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 100, (255, 0, 0), -1)

# 创建图像与窗口并将窗口与回调函数绑定
img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while True:
    cv2.imshow('image', img)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()
```

拖动绘制绿色矩形，按 `m` 切换为绘制红色圆点：

```python
import cv2
import numpy as np

drawing = False
mode = True

ix, iy = -1, -1

def draw_circle(event: int, x: int, y: int, flags: int,
                param) -> None:
    global ix, iy, drawing, mode
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and\
            flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while True:
    cv2.imshow('image', img)
    key = cv2.waitKey(1) & 0xff
    if key == ord('m'):
        print('mode:', mode)
        mode = not mode
    elif key == 27:
        break

cv2.destroyAllWindows()
```

## 2.5 调色板

::: details 学习目标

- 把滑动条绑定到 OpenCV 窗口上
- 函数
    - `cv2.creatTrackbar()` 创建拖动条
    - `cv2.getTrackbarPos()` 获取拖动条位置

:::

```python
import cv2
import numpy as np

def nothing(x: object) -> None:
    ...

img = np.zeros((300, 512, 3), np.uint8)

cv2.namedWindow('image')
cv2.createTrackbar('R', 'image', 0, 255, nothing)
cv2.createTrackbar('G', 'image', 0, 255, nothing)
cv2.createTrackbar('B', 'image', 0, 255, nothing)

switch = '0:OFF\n1:ON'
cv2.createTrackbar(switch, 'image', 0, 1, nothing)

while True:
    cv2.imshow('image', img)
    key = cv2.waitKey(1) & 0xff
    if key == 27:
        break

    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')
    s = cv2.getTrackbarPos(switch, 'image')

    if s == 0:
        img[:] = 0
    else:
        img[:] = b, g, r

cv2.destroyAllWindows()
```

结合上一节的知识，创建一个画板，可以自选各种颜色的画笔绘画各种图形：

```python
import cv2
import numpy as np

def nothing(x: object) -> None:
    pass

drawing = False
mode = True
ix, iy = -1, -1

def draw_circle(event: int, x: int, y: int, flags: int,
                param) -> None:
    global ix, iy, drawing, mode
    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')
    color = (b, g, r)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and\
            flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img, (ix, iy), (x, y),
                              color, -1)
            else:
                cv2.circle(img, (x, y), 3, color, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

img = np.zeros((300, 512, 3), np.uint8)

cv2.namedWindow('image')
cv2.createTrackbar('R', 'image', 0, 255, nothing)
cv2.createTrackbar('G', 'image', 0, 255, nothing)
cv2.createTrackbar('B', 'image', 0, 255, nothing)
cv2.setMouseCallback('image', draw_circle)

while True:
    cv2.imshow('image', img)
    key = cv2.waitKey(1) & 0xff
    if key == 27:
        break
    elif key == ord('m'):
        mode = not mode

cv2.destroyAllWindows()
```
