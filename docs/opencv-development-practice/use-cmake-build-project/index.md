---
title: OpenCV 使用 CMake 构建跨平台应用
description: 
---

# OpenCV 使用 CMake 构建跨平台应用

## 1. 定义标准化的工具

首先，我们习惯上定义一个公共头文件 `common.h`，包含一些和系统有关的函数和宏。

考虑到 Windows 臭名昭著的乱码和字符集问题，我们使用 `wchar_t` 类型来解决，其中：
- 宏 `DLL_EXPORT` 表示这个函数被导出
- 宏 `S` 表示将字符串常量转换为当前系统上的特定类型字符串
- 类型 `path_t` 表示当前系统上的字符指针类型
- 函数 `toStr()` 则表示将当前系统上的特定类型字符串转换为 `std::string`

下面是头文件和实现函数：

::: code-tabs

@tab h

```cpp
// common.h
#ifndef COMMON_H
#define COMMON_H

#include <string>

#if _WIN32
#define DLL_EXPORT extern "C" __declspec(dllexport)
#define S(t) L##t
#include <windows.h>

typedef wchar_t* path_t;

std::string toStr(LPCWSTR pwszSrc);

#else

#define DLL_EXPORT extern "C"
#define S(t) t

typedef char* path_t;

std::string toStr(const path_t path);

#endif  // _WIN32

#endif  // COMMON_H
```

@tab cpp

```cpp
// common.cpp
#include "common.h"

#if _WIN32

std::string toStr(LPCWSTR pwszSrc) {
    int nLen = WideCharToMultiByte(CP_ACP, 0, pwszSrc, -1, NULL, 0, NULL, NULL);
    if (nLen <= 0)
        return std::string("");
    char* pszDst = new char[nLen];
    if (NULL == pszDst)
        return std::string("");
    WideCharToMultiByte(CP_ACP, 0, pwszSrc, -1, pszDst, nLen, NULL, NULL);
    pszDst[nLen - 1] = 0;
    std::string strTemp(pszDst);
    delete[] pszDst;
    return strTemp;
}

#else

std::string toStr(const path_t path) {
    std::string s = path;
    return s;
}

#endif  // _WIN32
```

:::

## 2. CMake 定义

这里我们以 YOLO 举例，假设我的项目里有三个编译目标（`yolov3`、`yolov5` 和 `yolov7`）：

```cmake
cmake_minimum_required(VERSION 3.10)
project(yolo-collections)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIBRARY_DIRS})

macro(create_target name)
    add_executable(${name} "src/${name}.cpp" "src/common.cpp")
    target_link_libraries(
        ${name}
        ${OpenCV_LIBS}
    )
endmacro(create_target name)

create_target(yolov3)
create_target(yolov5)
create_target(yolov7)
```

## 3. 构建项目

### 3.1 Windows 下编译构建

Windows 使用 CMake 可以采用几种不同的方式：
1. 创建 Visual Studio 项目，在 VS 中构建，详情略
2. 使用 CMake-GUI 编译，下面是步骤
3. 使用 CMake 命令行工具

下面是使用 CMake-GUI 的步骤：
1. 配置 `source` 文件夹和 `build` 文件夹
2. 点击 **配置**（Configure）
3. 点击 **生成**（Generate）

::: warning 使用 MinGW

在使用 CMake 推荐使用 Visual Studio 作为后端，MinGW 在 Windows 上构建项目并不被官方支持，因此有很多项目无法成功构建。在本站下面的项目都不会使用 MinGW 构建，如果需要请自行尝试，遇到问题也无法提供解答。

:::

下面举例使用 VS 2019 作为后端，用于生成 64 位程序的命令：

```bash
cmake -B ./build -G "Visual Studio 16 2019" -T host=x64 -A x64 .
cmake --build ./build --config Release --target ALL_BUILD -j 4 --
```

### 3.2 Linux / Mac 系统编译构建

直接使用 CMake 即可：

```bash
mkdir -p build
cd build
cmake ../src
make -j $(nproc)
```

如果不想每次都创建 `build/` 文件夹，可以在 CMake 中指定编译目标输出路径：

```cmake
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
```
