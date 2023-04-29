# OpenCV C++ 编译和构建指南

## 1. 基本编译指南

下面都以 OpenCV 4.7.0 的编译为例。下载源代码 [`opencv-4.7.0.zip`](https://github.com/opencv/opencv/releases) 和 [`opencv_contrib-4.7.0.zip`](https://github.com/opencv/opencv_contrib/tags)，解压到同一个文件夹。

::: code-tabs

@tab Linux

```bash
curl -Lj -o opencv-4.7.0.zip https://github.com/opencv/opencv/archive/refs/tags/4.7.0.zip
curl -Lj -o opencv_contrib-4.7.0.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.7.0.zip
unzip opencv-4.7.0.zip -d .
unzip opencv_contrib-4.7.0.zip -d .
cd opencv-4.7.0
mkdir -p build
```

@tab Windows

```bash
curl -Lj -o opencv-4.7.0.zip https://github.com/opencv/opencv/archive/refs/tags/4.7.0.zip
curl -Lj -o opencv_contrib-4.7.0.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.7.0.zip
7z x opencv-4.7.0.zip
7z x opencv_contrib-4.7.0.zip
cd opencv-4.7.0
md build
```

:::

需要预先安装编译依赖，Windows 系统见下文。例如 Ubuntu 的基本依赖为：

```bash
sudo apt install build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-22-dev libopenexr-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
```

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
    -D CMAKE_INSTALL_PREFIX=/usr/local/opencv470 \
    -D INSTALL_C_EXAMPLE=OFF \
    -D BUILD_EXAMPLE=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.7.0/modules \
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

## 3. Windows 编译指南

Windows 编译要求：
- Visual Studio 2019，并安装 C++ 桌面开发基本组件
- CMake GUI 版本，也可以完全使用命令行

在配置过程中将会自动下载依赖。

建议配置项：
- CUDA 支持，如果安装了 CUDA 可选择支持，不过需要选择你的显卡对应的算力
- `BUILD_opencv_world` 编译包含所有库的动态链接库
- Test 选项可去除
- Python/Java/JS 选项去除

::: info 使用代理

使用大陆的网络可能无法下载依赖，需要使用代理。在 Environment 中增加 `http_proxy` 和 `https_proxy` 变量，并将其指定为一个可用的代理地址。例如：

```yml
http_proxy: 'http://127.0.0.1:10809'
https_proxy: 'socks5://127.0.0.1:10808'
```

在 Windows 上，Python 可能对 SOCKS 支持不完备，如果使用 SOCKS 协议时出现无法握手或者无法使用 DNS，将 `https_proxy` 代替为 `http_proxy` 的值即可：

```yml
http_proxy: 'http://127.0.0.1:10809'
https_proxy: 'http://127.0.0.1:10809'
```

:::

::: details 命令行编译指南

```bash
cmake -G "Visual Studio 16 2019" -T host=x64 -A x64 ^
    -DBUILD_DOCS=OFF ^
    -DBUILD_SHARED_LIBS=ON ^
    -DBUILD_FAT_JAVA_LIB=OFF ^
    -DBUILD_TESTS=OFF ^
    -DBUILD_TIFF=ON ^
    -DBUILD_JASPER=ON ^
    -DBUILD_JPEG=ON ^
    -DBUILD_PNG=ON ^
    -DBUILD_ZLIB=ON ^
    -DBUILD_OPENEXR=OFF ^
    -DBUILD_opencv_apps=OFF ^
    -DBUILD_opencv_calib3d=OFF ^
    -DBUILD_opencv_contrib=OFF ^
    -DBUILD_opencv_features2d=OFF ^
    -DBUILD_opencv_flann=OFF ^
    -DBUILD_opencv_gpu=OFF ^
    -DBUILD_opencv_java=OFF ^
    -DBUILD_opencv_legacy=OFF ^
    -DBUILD_opencv_ml=OFF ^
    -DBUILD_opencv_nonfree=OFF ^
    -DBUILD_opencv_objdetect=OFF ^
    -DBUILD_opencv_ocl=OFF ^
    -DBUILD_opencv_photo=OFF ^
    -DBUILD_opencv_python=OFF ^
    -DBUILD_opencv_stitching=OFF ^
    -DBUILD_opencv_superres=OFF ^
    -DBUILD_opencv_ts=OFF ^
    -DBUILD_opencv_video=OFF ^
    -DBUILD_opencv_videostab=OFF ^
    -DBUILD_opencv_world=ON ^
    -DBUILD_opencv_lengcy=OFF ^
    -DBUILD_opencv_lengcy=OFF ^
    -DWITH_1394=OFF ^
    -DWITH_EIGEN=OFF ^
    -DWITH_FFMPEG=OFF ^
    -DWITH_GIGEAPI=OFF ^
    -DWITH_GSTREAMER=OFF ^
    -DWITH_GTK=OFF ^
    -DWITH_PVAPI=OFF ^
    -DWITH_V4L=OFF ^
    -DWITH_LIBV4L=OFF ^
    -DWITH_CUDA=OFF ^
    -DWITH_CUFFT=OFF ^
    -DWITH_OPENCL=OFF ^
    -DWITH_OPENCLAMDBLAS=OFF ^
    -DWITH_OPENCLAMDFFT=OFF ..
cmake --build . --config Release --target ALL_BUILD -j 20 --
```

:::

## 4. OpenCV CUDA 编译指南

下面进行一份完整的 CUDA 支持的 OpenCV 编译过程，以 Windows 和 OpenCV 4.7.0 为例。

下面我的机器配置，请根据条件选择合适的配置项目：

| 配置项 | 值                            |
| ------ | ----------------------------- |
| CPU    | Intel(R) Core(TM) i9-12900H   |
| GPU    | NVIDIA RTX 3070 Ti Laptop GPU |
| 显存   | 8 GB                          |
| 内存   | 16 GB                         |
| CUDA   | 12.1                          |
| VS     | Visual Studio 2019 Community  |

打开 CMake-GUI，然后选择源码和编译目录。

配置 Environment，设置 `http_proxy` 和 `https_proxy` 为代理地址。

```yml
http_proxy: 'http://127.0.0.1:10809'
https_proxy: 'http://127.0.0.1:10809'
```

点击 Configure，选择 Visual Studio 2019，然后点击 Finish。

第一次可选配置项：

| 配置项                                 | 值                                                         | 说明                                     |
| -------------------------------------- | ---------------------------------------------------------- | ---------------------------------------- |
| `OPENCV_GENERATE_SETUPVARS`            | `OFF`                                                      | 为了防止 `OpenCVGenSetupVars` 警告       |
| `BUILD_CUDA_STUBS`                     | `OFF`                                                      | CUDA 桩库                                |
| `WITH_CUDA`                            | `ON`                                                       | CUDA                                     |
| `OPENCV_DNN_CUDA`                      | `ON`                                                       | CUDA DNN                                 |
| `ENABLE_FAST_MATH`                     | `ON`                                                       | 快速数学库                               |
| `BUILD_JAVA`                           | `OFF`                                                      | Java 绑定                                |
| `BUILD_opencv_java_bindings_generator` | `OFF`                                                      | Java 绑定生成器                          |
| `OPENCV_EXTRA_MODULES_PATH`            | `D:/workspace/repo/opencv4.7/opencv_contrib-4.7.0/modules` | 额外模块路径，选择自己的下载位置         |
| `OPENCV_ENABLE_NONFREE`                | `ON`                                                       | 非自由组件                               |
| `BUILD_opencv_js`                      | `OFF`                                                      | JS 绑定                                  |
| `BUILD_opencv_js_bindings_generator`   | `OFF`                                                      | JS 绑定生成器                            |
| `BUILD_opencv_objc_bindings_generator` | `OFF`                                                      | Objective-C 绑定生成器                   |
| `BUILD_opencv_world`                   | `ON`                                                       | 编译包含所有库的动态链接库               |
| `BUILD_SHARED_LIBS`                    | `ON`                                                       | 编译动态链接库，如果需要静态库请不要勾选 |

根据需要选择即可，点击 Configure。

查看显卡算力可以在 [NVIDIA 开发者官网](https://developer.nvidia.com/cuda-gpus#compute) 的算力表，在 **CUDA-Enabled GeForce and TITAN Products** 一栏可以看到 GeForce 系列的算力，例如 RTX 3070 Ti Laptop GPU 的算力为 8.6。

第二次可选配置项：

| 配置项           | 值        | 说明                                |
| ---------------- | --------- | ----------------------------------- |
| `CUDA_FAST_MATH` | `ON`      | 快速数学库                          |
| `CUDA_ARCH_BIN`  | `7.5;8.6` | 显卡算力，此值兼容 20/30 系多数显卡 |
| `WITH_FREETYPE`  | `OFF`     | FreeType 字体库                     |

点击 Configure，然后点击 Generate。此时将生成 Visual Studio 2019 的解决方案。

打开 `OpenCV.sln`，然后选择 Release 模式，右键 `ALL_BUILD`，然后选择生成。

生成完成后，右键 `INSTALL`，然后选择生成。
