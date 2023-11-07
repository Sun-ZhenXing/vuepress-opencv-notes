# VuePress OpenCV Notes

![actions status](https://img.shields.io/github/actions/workflow/status/Sun-ZhenXing/vuepress-opencv-notes/deploy-docs.yml?branch=main)

🌐 [中文文档](./README.zh.md) | 🚀 [Online Reading (Chinese)](https://blog.alexsun.top/vuepress-opencv-notes/)

## 📝 Introduction

This is a note for learning OpenCV.

## 🛠️ Build Docs

Build with `pnpm`, the content will be generated in `docs/.vuepress/dist`:

```bash
pnpm i
pnpm build
```

Build with Docker:

```bash
docker build -t vuepress-opencv-notes .
docker run -itd -p 80:80 vuepress-opencv-notes
```

Now you can visit <http://localhost> to see your docs.

## 📜 License

- All files except `/docs/*` are licensed under [MIT License](https://mit-license.org/)
- `/docs/*` are licensed under [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

For more details, please read [this series of documents license](https://github.com/Sun-ZhenXing/Sun-ZhenXing.github.io#%E5%BC%80%E6%BA%90%E5%8D%8F%E8%AE%AE).
