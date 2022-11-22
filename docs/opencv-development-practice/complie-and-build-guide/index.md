---
lang: zh-CN
title: OpenCV C++ 编译和构建指南
description: 使用 C++ 版本的 OpenCV 的编译指南
---

# OpenCV C++ 编译和构建指南

## 1. 基本编译指南

需要预先安装编译依赖。

CMake 配置：

```bash
mkdir build
cmake ..
```

编译和安装：

```bash
make -j $(nproc)

sudo make install
```

## 2. 构建 Qt 支持

如果需要编译基于 Qt 的图形界面，启用 `-D WITH_QT=ON` 来编译，例如：

```bash
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local/opencv460 \
    -D INSTALL_C_EXAMPLE=OFF \
    -D BUILD_EXAMPLE=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.6.0/modules \
    -D WITH_V4L=ON \
    -D WITH_TBB=ON \
    -D WITH_VTK=ON \
    -D WITH_GTK=ON \
    -D WITH_OPENMP=ON \
    -D WITH_OPENGL=ON \
    -D WITH_QT=ON \
    -D BUILD_opencv_python3=OFF \
    -D BUILD_TIFF=ON ..
```
