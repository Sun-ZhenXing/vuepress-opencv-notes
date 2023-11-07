# VuePress OpenCV 笔记

![actions status](https://img.shields.io/github/actions/workflow/status/Sun-ZhenXing/vuepress-opencv-notes/deploy-docs.yml?branch=main)

🚀 [在线阅读（中文）](https://blog.alexsun.top/vuepress-opencv-notes/).

## 📝 介绍

这是一个学习 OpenCV 的笔记。

## 🛠️ 构建文档

使用 `pnpm` 构建， 内容将生成在 `docs/.vuepress/dist` 下:

```bash
pnpm i
pnpm build
```

使用 Docker 构建:

```bash
docker build -t vuepress-opencv-notes .
docker run -itd -p 80:80 vuepress-opencv-notes
```

现在你可以访问 <http://localhost> 来查看你的文档。

## 📜 开源协议

- 除了文档外的其他文件（除了 `/docs/` 文件夹），使用 [MIT 协议](https://mit-license.org/) 开源
- 文档（即 `/docs/` 下的文件）使用 [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 共享

详情请阅读 [此系列文档开源协议](https://github.com/Sun-ZhenXing/Sun-ZhenXing.github.io#%E5%BC%80%E6%BA%90%E5%8D%8F%E8%AE%AE)。
