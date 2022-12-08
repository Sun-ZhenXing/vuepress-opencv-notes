---
title: 第 2 章：OpenCV 基础知识导论
description: 使用 CMake 构建项目，并了解 OpenCV 的基础数据结构和操作
---

# 2. OpenCV 基础知识导论

本章介绍以下主题：
- [x] 使用 CMake 配置项目
- [x] 读取 / 写入图像
- [x] 读取视频和访问相机
- [x] 主要的图像结构
- [x] 其他重要的结构
- [x] 基本矩阵运算简介
- [x] 使用 XML / YAML 存储 OpenCV API 进行文件存储操作

## 2.1 技术要求

- [x] 熟悉 C++ 语言
- [x] [本章代码](https://github.com/PacktPublishing/Learn-OpenCV-4-By-Building-Projects-Second-Edition/tree/master/Chapter_02)
- [x] 现代操作系统，例如 Ubuntu 20+ 或者 Windows 10+

## 2.2 基本 CMake 配置文件

CMake 可以使用 `CMakeLists.txt` 配置编译过程，其文件内容类似于：

```cmake
# 指定 CMake 最小版本
cmake_minimum_required(VERSION 3.10)

# 项目名，此名称保存在 PROJECT_NAME
project(cmake-test)

# 添加编译目标：编译 main.cpp 为可执行文件
add_executable(${PROJECT_NAME} main.cpp)
```

`project(cmake-test)` 后，这个名称可以通过 `PROJECT_NAME` 来访问。`${}` 表达式能够访问上下文环境中定义的变量，上面的例子即使用项目名作为可执行文件名称。

## 2.3 创建一个库

```cmake
# 创建动态链接库
add_library(Hello hello.cpp)

# 创建可执行文件
add_executable(main main.cpp)

# 将目标链接到指定库
target_link_libraries(main Hello)
```

::: tip 链接行为

链接库的时候指定 `SHARED` 或者 `STATIC` 能指定生成的库是静态库（`.a` / `.lib`）还是共享库（`.so` / `.dll`）。

CMake 的链接是静态优先的，但查找库的时候默认查找 `.so` 文件，可以配合几个参数来个性化 CMake 设置：
- `set(CMAKE_FIND_LIBRARY_SUFFIXES .a)` 设置查找库名后缀
- `find_library()` 查找指定库

创建库的时候也可以指定可见属性：
- 如果源文件（`.cpp` / `.cc`）中包含第三方头文件，但是头文件（例如 `.hpp`）中不包含该第三方文件头，采用 `PRIVATE`
- 如果源文件和头文件中都包含该第三方文件头，采用 `PUBLIC`
- 如果头文件中包含该第三方文件头，但是源文件中不包含，采用 `INTERFACE`

:::

## 2.4 管理依赖项

CMake 具有搜索依赖项和外部库的能力，这使得我们能够根据项目中的复杂组件构建复杂的项目。

我们下面将 OpenCV 添加到项目中：

```cmake
cmake_minimum_required(VERSION 3.10)
project(chapter2)

find_package(OpenCV REQUIRED)
message("OpenCV version: ${OpenCV_VERSION}")

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
set(SRC main.cpp)
add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
```

我们可以使用 `cmake_policy()` 来设置当前的策略，来避免 CMake 版本过高或者过低的问题，语法：

```cmake
cmake_policy(VERSION <min>[...<max>])
# 或者
cmake_policy(SET CMP[NNNN] <variable>)
```

例如：

```cmake
cmake_policy(SET CMP0012 NEW)
```

`CMP0012` 规则为 `if()` 能够识别数字和布尔常量。

如果我们只需要 OpenCV 的某一个子模块，也可以使用：

```cmake
find_package(OpenCV REQUIRED core)
```

这样我们只会引入 OpenCV 的 `core` 模块。

我们还可以在一个变量里面添加更多的值，例如：

```cmake
set(SRC main.cpp
        utils.cpp
        color.cpp)
```

## 2.5 让脚本更复杂

下面我们创建一个更复杂的例子，包括子文件夹，库和可执行文件。使用一个 `CMakeLists.txt` 就可以构建，更常见的方式是为子项目使用不同的 `CMakeLists.txt`，可以使其更加灵活便捷。

下面是目录结构：

```txt
CMakeLists.txt
main.cpp
utils/
    CMakeLists.txt
    computeTime.cpp
    computeTime.h
    logger.cpp
    logger.h
    plotting.cpp
    plotting.h
```

项目根目录的 `CMakeLists.txt` 的内容是：

```cmake
cmake_minimum_required(VERSION 3.10)
project(chapter2)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

add_subdirectory(utils)

option(WITH_LOG "Build with output logs and images int tmp" OFF)
if(WITH_LOG)
    add_definitions(-DLOG)
endif()

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} Utils)
```

`add_subdirectory()` 告诉 CMake 分析所需子文件夹的 `CMakeLists.txt`。

下面是 `utils/` 文件夹下面的 `CMakeLists.txt`：

```cmake
set(UTILS_LIB_SRC
    computeTime.cpp
    logger.cpp
    plotting.cpp
)
add_library(Utils ${UTILS_LIB_SRC})

target_include_directories(Utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
```

`option()` 可用于创建变量，并包含描述。变量被定义后，可以通过预编译指令来使用：

```cpp
#ifdef LOG
    logi("Number of iteration %d", i);
#endif
```

现在我们已经基本入门了 CMake 了，可以在不同的操作系统中构建我们的项目。

## 2.6 图像与矩阵

任何图像都可以表示为包含一系列数字的矩阵，一般这些数字用于表示光的波长或波长范围的光强度的测量结果。图像中的每个点被称为像素，每个像素可以存储一个或多个值。

这些储存值的不同决定了图像的不同类别
- 只有一个比特的二进制图像
- 灰度图
- 三通道彩色图像

一般使用一个字节来保存（例如 RGB888），其范围是 $0-255$，也有例外，例如 HDR 或热成像通常使用浮点数。

OpenCV 使用 `Mat` 类来储存图像，灰度图为一个矩阵。而 RGB 彩色图像使用 $w \times h \times c$ 的矩阵来表示（分别是宽度、高度和通道数）。

OpenCV 的图像保存格式为 **行 -> 列 -> BGR**，读取一个像素的指针：

```cpp
val = row_i * num_cols * channels + col_i * channels;
```

::: info 使用指针访问

OpenCV 函数非常适合用于随机访问，但是有时直接访问内存更有效（使用指针运算），例如当我们必须在循环中访问所有像素的时候。

:::

## 2.7 读 / 写图像

我们直接看代码：

::: warning 代码风格

本文的代码风格和书本略有不同，如果不说明，本文都不使用 `using namespace` 来引入命名空间，这样做是为了防止潜在的命名冲突。

:::

```cpp
#include <iostream>
#include <sstream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char* argv[]) {
    cv::Mat color = cv::imread("lena.jpg");
    cv::Mat gray = cv::imread("lena.jpg", cv::IMREAD_GRAYSCALE);

    // 如果图片为空
    if (color.empty() || gray.empty()) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    // 写入文件
    cv::imwrite("lena_gray.png", gray);

    int myRow = color.cols - 1;
    int myCol = color.rows - 1;

    // 读取指定位置像素值
    cv::Vec3b myPixel = color.at<cv::Vec3b>(myRow, myCol);
    std::cout << "Pixel at (" << myRow << ", " << myCol << "): ("
              << (int)myPixel[0] << ", " << (int)myPixel[1] << ", "
              << (int)myPixel[2] << ")" << std::endl;

    // 显示图片
    cv::imshow("Color", color);
    cv::imshow("Gray", gray);
    // 持续等待直到任意按键被按下
    cv::waitKey(0);
    return 0;
}
```

`<opencv2/core.hpp>` 包含了基本的图像数据处理功能，包括基本的类（例如矩阵）。

`<opencv2/highgui.hpp>` 包含了读写函数和 GUI 相关功能。

`imread()` 函数是读取图像的主函数。打开一个图像，并使用矩阵储存它。它接收两个参数，第一个参数是图像路径的字符串。第二个参数是可选的，常用选项为：

| 名称                   | 实际值           | 功能                 |
| ---------------------- | ---------------- | -------------------- |
| `cv::IMREAD_UNCHANGED` | `enum -1`        | 如果有深度则保留深度 |
| `cv::IMREAD_COLOR`     | `enum 1`（默认） | 转换为三通道图像     |
| `cv::IMREAD_GRAYSCALE` | `enum 0`         | 转换为灰度图         |

我们创建以下示例 `CMakeLists.txt`，并编译代码：

```cmake
cmake_minimum_required(VERSION 3.10)
project(chapter2)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
```

Linux 下编译：

```bash
mkdir -p build
cd build
cmake ..
make
```

::: tip 构建指南

关于其他系统下的编译，或者需要编写各种系统兼容的程序，可以阅读 [OpenCV 使用 CMake 构建跨平台应用](../../opencv-development-practice/use-cmake-build-project/)。

:::

## 2.8 读取视频和摄像头

首先，我们介绍一个十分有用的类 `CommandLineParser`，它用于解析命令行的参数，在 OpenCV 3.0 中被引入。

允许的参数以字符串的形式给出，可以分割为多行，其中每一行都是如下格式：

```cpp
"{name | default_value | description}"
```

使用类似 `@name` 格式定义的参数将会作为默认输入。下面是一些解析器的实用方法：
- `parser.about(msg)`：程序的关于信息
- `parser.has(name)`：是否存在指定参数
- `parser.get<cls>(index)`：获取指定位置的参数
- `parser.printMessage()`：打印信息

```cpp
#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

const char* keys = {
    "{help h usage ? | | print this message}"
    "{@video | | Video file, if not defined try to use webcamera}"
};

int main(int argc, char* argv[]) {
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("This program shows how to read video from a file or camera.");
    if (parser.has("help")) {
        parser.printMessage();
        return -1;
    }
    std::string arg = parser.get<std::string>(0);
    cv::VideoCapture cap(0, cv::CAP_MSMF);
    if (arg.empty()) {
        cap.open(0);
    } else {
        cap.open(arg);
    }
    if (!cap.isOpened()) {
        parser.printMessage();
        return -1;
    }
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (!frame.empty()) {
            cv::imshow("Video", frame);
        } else {
            std::cout << "Can't read frame." << std::endl;
            break;
        }
        if (cv::waitKey(10) == 27) {
            break;  // stop capturing by pressing ESC
        }
    }
    cap.release();
    return 0;
}
```

::: warning Windows 潜在错误

Windows 调试摄像头可能出现莫名其妙的错误，查看 [附录](/learn-opencv-by-building-projects/appendix/windows-errors.md) 尝试解决。

:::

## 2.9 其他基本对象类型

我们已经了解了 `Mat` 和 `Vec3b` 类，还有很多类需要学习，常见的是：
- `Vec`
- `Scalar`
- `Point`
- `Size`
- `Rect`
- `RotatedRect`

### 2.9.1 `Vec` 对象类型

`Vec` 是一个数值向量模板，可以定义向量的类型和组件的数量：

```cpp
Vec<double, 19> myVector;
```

这里有很多预定义类型：

```cpp
typedef Vec<uchar, 2> Vec2b;
typedef Vec<uchar, 3> Vec3b;
typedef Vec<uchar, 4> Vec4b;

typedef Vec<short, 2> Vec2s;
typedef Vec<short, 3> Vec3s;
typedef Vec<short, 4> Vec4s;

typedef Vec<int, 2> Vec2i;
typedef Vec<int, 3> Vec3i;
typedef Vec<int, 4> Vec4i;

typedef Vec<float, 2> Vec2f;
typedef Vec<float, 3> Vec3f;
typedef Vec<float, 4> Vec4f;
typedef Vec<float, 6> Vec6f;

typedef Vec<double, 2> Vec2d;
typedef Vec<double, 3> Vec3d;
typedef Vec<double, 4> Vec4d;
typedef Vec<double, 6> Vec6d;
```

向量还支持：

```cpp
v1 = v2 + v3;
v1 = v2 - v3;
v1 = scale * v2;
v1 = -v2;
v1 += v2;

v1 == v2;
v1 != v2;
norm(v1);
```

`norm(v)` 为计算 **欧几里德范数**（Euclidean norm），即

$$
\left\Vert v \right\Vert _2 = \sqrt{\sum_{x \in v} x^2}
$$

### 2.9.2 `Scalar` 对象类型

`Scalar` 对象类型是从 `Vec` 派生的模板类，有四个元素。`Scalar` 类型主要用于传递和读取像素值。

可以使用 `[]` 运算符访问和读取下标位置的值，可以使用不同的方式初始化：

```cpp
cv::Scalar s0(0);
cv::Scalar s1(0.0, 1.0, 2.0, 3.0);
cv::Scalar s2(s1);
```

### 2.9.3 `Point` 对象类型

另一个非常常见的类模板是 `Point`，该类定义一个由其坐标 `x` 和 `y` 指定的 2D 点。

::: tip Point3

和 `Point` 一样，`Point3` 模板类用于定义指定的 3D 点。

:::

和 `Vec` 类一样，OpenCV 为方便定义了以下别名：

```cpp
typedef Point_<int> Point2i;
typedef Point2i Point;
typedef Point_<float> Point2f;
typedef Point_<double> Point2d;
```

OpenCV 为 `Point` 定义了以下运算符：

```cpp
p1 = p2 + p3;
p1 = p2 - p3;
p1 = a * p2;
p2 = p2 / a;
p1 += p2;
p1 -= p2;
p1 *= a;
p1 /= a;
double value = norm(p1);
p1 == p2;
p1 != p2;
```

其中 `norm()` 仍然为计算欧几里得范数。

### 2.9.4 `Size` 对象类型

`Size` 在 OpenCV 中被广泛使用，用于指定图像或者矩形的大小。这个类有两个重要的成员 `width` 和 `height`，以及 `area()` 方法。

示例：

```cpp
cv::Size s(100, 100);
cv::Mat img = cv::Mat::zeros(s, CV_8UC1);
s.width = 200;
int area = s.area();
```

### 2.9.5 `Rect` 对象类型

`Rect` 也是一个非常重要的模板类，用于定义以下参数的 2D 矩形：
- 左上角是坐标
- 矩形的宽度和高度

`Rect` 用于定义图像的 **感兴趣区域**（ROI，Region of Interest），例如：

```cpp
cv::Mat img = cv::imread("lena.jpg");
cv::Rect rect_roi(0, 0, 100, 100);
cv::Mat img_roi = img(rect_roi);
```

### 2.9.6 `RotatedRect` 对象类型

`RotatedRect` 用于定义一个旋转矩形，参数由中心点、宽度、高度和旋转角度（单位：°）指定。其声明如下：

```cpp
RotatedRect(const Point2f& center, const Size2f& size, float angle);
```

这个类的一个有趣的方法是 `boundingBox()`，该函数返回一个包含旋转矩形的 `Rect`。

## 2.10 基本矩阵运算

创建一个 `Mat`：

```cpp
cv::Mat a = cv::Mat(cv::Size(5, 5), cv::CV_32F);
```

::: tip 从缓冲区创建

使用 `Mat(size, type, point_to_buffer)` 可以使用来自第三方库的存储缓冲区创建一个新的矩阵，无需复制数据。

:::

常见的通道类型：
- `CV_8UC1`
- `CV_8UC3`
- `CV_8UC4`
- `CV_32FC1`
- `CV_32FC3`
- `CV_32FC4`

::: tip 任何类型的矩阵

使用 `CV_number_typeC(n)` 创建任何类型的矩阵，其中 `number_type` 是 8 位无符号数（`8U`）到 64 位浮点数（`64F`），其中 `(n)` 是通道数，允许的范围是 `[1, CV_CN_MAX]`。

:::

C++ OpenCV 使用 `Mat` 类操作图像，其结构大致如下：

```cpp
class CV_EXPORTS Mat {
public:
    // 一系列函数
    //  ...
    /* flag 参数中包含许多关于矩阵的信息，如：
        - Mat 的标识
        - 数据是否连续
        - 深度
        - 通道数目
     */
    int flags;
    // 矩阵的维数，取值应该大于或等于 2
    int dims;
    // 矩阵的行数和列数，如果矩阵超过 2 维，这两个变量的值都为 -1
    int rows, cols;
    // 指向数据的指针
    uchar* data;
    // 指向引用计数的指针
    // 如果数据是由用户分配的，则为 NULL
    int* refcount;
    // 其他成员变量和成员函数
    // ...
};
```

`Mat` 对象支持所有的矩阵运算，包括（`+` / `-`）一个相同大小的矩阵：

```cpp
Mat a = Mat::eye(Size(3, 2), CV_32F);
Mat b = Mat::ones(Size(3, 2), CV_32F);
Mat c = a + b;
Mat d = a - b;
```

如果加减运算对象是数字，那么将自动进行 **广播** 操作，相当于矩阵的每个元素都和这个数运算。

乘法有两种，一种是线性代数所定义的乘法，还有一种是元素积（即对应位置的元素相乘，要求操作数大小相同，相当于 MATLAB 中的 `.*`）。

OpenCV 支持元素积，需要使用 `.mul()` 方法，同样也支持乘一个数。

其他常见操作：
- 转置 `.t()`
- 求逆 `.inv()`

还有一些实用的数学函数：
- `int countNonZero(src)` 计算非零元素数量
- `void meanStdDev(src, mean, srddev)` 计算平均值和标准差
- `void minMaxLoc(src, minVal, maxVal, minLoc, maxLoc)` 检测矩阵的最小值、最大值并且包括最值的位置

::: tip 核心功能

可以从官方文档查看 [各种模块](https://docs.opencv.org/4.6.0/modules.html) 的 API，从 Core 模块查看核心功能。

:::

## 2.11 基本数据存储

OpenCV 支持使用 XML/YAML 来存储和读取数据。

### 写入文件

要把一些 OpenCV 或其他数值写入文件，可以使用 `FileStorage` 类，同时要使用流运算符 `<<` 来操作 STL 流：

```cpp
#include <iostream>
#include <opencv2/core.hpp>

int main(int argc, char* argv[]) {
    cv::FileStorage fs("test.yml", cv::FileStorage::WRITE);
    int fps = 5;
    fs << "fps" << fps;
    cv::Mat m1 = cv::Mat::eye(2, 3, CV_32F);
    cv::Mat m2 = cv::Mat::ones(3, 2, CV_32F);
    cv::Mat result = (m1 + 1).mul(m1 + 3);
    fs << "Result" << result;
    fs.release();

    cv::FileStorage fs2("test.yml", cv::FileStorage::READ);
    cv::Mat r;
    fs2["Result"] >> r;
    std::cout << r << std::endl;
    fs2.release();
    return 0;
}
```

完成后会创建一个 `test.yml` 文件，内容为：

```yml
%YAML:1.0
---
fps: 5
Result: !!opencv-matrix
   rows: 2
   cols: 3
   dt: f
   data: [ 8., 3., 3., 3., 8., 3. ]
```

### 读取文件

```cpp
#include <iostream>
#include <opencv2/core.hpp>

int main(int argc, char* argv[]) {
    cv::FileStorage fs2("test.yml", cv::FileStorage::READ);
    cv::Mat r;
    fs2["Result"] >> r;
    std::cout << r << std::endl;
    fs2.release();
    return 0;
}
```

## 2.12 总结

本章我们学习了 OpenCV 最重要的类型和操作，了解了矩阵的结构和基本运算，并且还有一些其他类、向量等。我们还探讨了保存数据文件的方法。
