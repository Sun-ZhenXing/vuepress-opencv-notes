# 使用 VS Code 开发 C++ OpenCV

需要使用 CMake 和 C/C++ 语言扩展，并配置 `.vscode/c_cpp_properties.json`，样例如下：

```json
{
    "configurations": [
        {
            "name": "Win32",
            "includePath": [
                "${default}",
                "${workspaceFolder}/**",
                "D:/Program/opencv4.7/opencv/build/include"
            ],
            "defines": [
                "_DEBUG",
                "UNICODE",
                "_UNICODE"
            ],
            "windowsSdkVersion": "10.0.19044.0",
            "cStandard": "c17",
            "cppStandard": "c++17",
            "intelliSenseMode": "windows-msvc-x64",
            "configurationProvider": "ms-vscode.cmake-tools",
            "compilerPath": "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.35.32215/bin/Hostx64/x64/cl.exe"
        }
    ],
    "version": 4
}
```

这份配置依赖于已经安装的 Visual Studio，其中需要更改的配置：

- `includePath` 需要改为本机的 OpenCV Include 路径，具体取决于安装时的路径
- `windowsSdkVersion` 需要改为本机已经安装的 Windows SDK
- `compilerPath` 修改为本机安装的 MSVC 编译器路径

其他常见更改：

- C/C++ 版本
- 在 `env` 中定义用户定义变量
- 定义标志 `defines`
