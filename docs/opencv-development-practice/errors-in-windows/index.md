---
title: Windows 错误合集
description: OpenCV 调试过程中在 Windows 平台上出现的错误
---

## 调用摄像头时出现错误

### VSFilter.dll 非法地址访问

| 错误调用       | 源             | 错误类型     |
| -------------- | -------------- | ------------ |
| `cap >> frame` | `VSFilter.dll` | 非法地址访问 |

Windows 上默认提供的 `VSFilter.dll` 没有符号表，版本也较低。可能和 OpenCV 4.6.0 已经不兼容，需要安装新的 VSFilter。

从 [GitHub: xy-VSFilter/releases](https://github.com/pinterf/xy-VSFilter/releases) 中下载最新的一个稳定版本，解压后得到 32/64 位的动态链接库，将 64 位的 DLL 覆盖到 `C:\Windows\System32`，32 位的 DLL 覆盖到 `C:\Windows\SysWOW64\`，重新运行后没有问题。
