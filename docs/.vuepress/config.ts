import { defineUserConfig, defaultTheme } from 'vuepress'
import { mdEnhancePlugin } from 'vuepress-plugin-md-enhance'
import { searchPlugin } from '@vuepress/plugin-search'

export default defineUserConfig({
  lang: 'zh-CN',
  title: 'OpenCV 笔记合集',
  description: '使用 C++ 和 Python 编写现代 OpenCV 应用',
  locales: {
    '/': {
      lang: 'zh-CN',
      title: 'OpenCV 笔记合集',
      description: '使用 C++ 和 Python 编写现代 OpenCV 应用',
    }
  },
  head: [
    ['link', { rel: 'icon', href: '/vuepress-opencv-notes/favicon.svg' }]
  ],
  base: '/vuepress-opencv-notes/',
  theme: defaultTheme({
    logo: '/favicon.svg',
    repo: 'Sun-ZhenXing/vuepress-opencv-notes',
    navbar: [
      {
        text: 'OpenCV4 计算机视觉项目实战笔记',
        children: [
          '/learn-opencv-by-building-projects/chapter01/',
          '/learn-opencv-by-building-projects/chapter02/',
          '/learn-opencv-by-building-projects/chapter03/',
          '/learn-opencv-by-building-projects/chapter04/',
          '/learn-opencv-by-building-projects/chapter05/',
          '/learn-opencv-by-building-projects/chapter06/',
          '/learn-opencv-by-building-projects/chapter07/',
          '/learn-opencv-by-building-projects/chapter08/',
          '/learn-opencv-by-building-projects/chapter09/',
          '/learn-opencv-by-building-projects/chapter10/',
          '/learn-opencv-by-building-projects/chapter11/',
          '/learn-opencv-by-building-projects/chapter12/',
        ]
      }
    ]
  }),
  plugins: [
    mdEnhancePlugin({
      gfm: true,
      container: true,
      linkCheck: true,
      vPre: true,
      tabs: true,
      codetabs: true,
      align: true,
      attrs: true,
      sub: true,
      sup: true,
      footnote: true,
      mark: true,
      imageLazyload: true,
      tasklist: true,
      katex: true,
      mermaid: true,
      delay: 200,
    }),
    searchPlugin({
    })
  ]
})
