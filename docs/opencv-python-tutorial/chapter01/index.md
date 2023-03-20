# 1. 安装 OpenCV

[[TOC]]

## 1.1 OpenCV 的 Python 默认发行版

OpenCV 提供 PyPI 发行的 Python 包，使用 Python 的二进制扩展并且是预编译的。

::: info 下载预编译包

PyPI 目前只提供 CPU 版本的预编译包，如果你需要 CUDA 支持或其他架构支持的 OpenCV 可以查阅其他包管理工具或自行编译。

:::

如果你的 Python 包管理器内没有安装 OpenCV，可以使用下面的命令直接安装：

```bash
pip install opencv-python
```

也可以指定版本：

```bash
pip install opencv-python==4.7.0.68
```

如果你使用 Anaconda，那么 OpenCV 已经默认安装，如果你想更新可以使用：

```bash
pip install -U opencv-python
# 或者
conda update opencv-python
```

## 1.2 安装扩展包和无 GUI 版本的 OpenCV

如果需要 OpenCV Contrib 模块中包含的算法，需要安装 `opencv-contrib-python`，安装命令如下：

```bash
pip install opencv-contrib-python
```

如果你在 `libGL` 支持不完备的系统（通常是无桌面的系统）上安装或使用 `opencv-python`，可能出现错误，可以安装无 GUI 支持的 OpenCV Headless 版本：

```bash
pip install opencv-python-headless
```

通常 Headless 版本用于服务器上使用。

同样，`opencv-contrib-python` 也提供 Headless 版本 `opencv-contrib-python-headless`：

```bash
pip install opencv-contrib-python-headless
```

::: warning 版本一致

`opencv-python` 和 `opencv-python-headless` 不能一起安装，否则导入包时产生冲突，另一个包无法被加载。安装 Contrib 版本也要和默认的 OpenCV 版本一致，否则会出现不兼容的问题。

:::

## 1.3 使用其他包管理器

在 Debian/Ubuntu 上使用，可以使用系统的包管理器安装：

```bash
sudo apt-get install python3-opencv
```

在 Termux 上安装时，默认的 `pip` 安装命令可能失败，使用 `pkg` 包管理器安装

```bash
pkg install opencv-python
```
