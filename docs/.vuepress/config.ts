import { defineUserConfig, defaultTheme } from 'vuepress'
import { mdEnhancePlugin } from 'vuepress-plugin-md-enhance'
import { searchPlugin } from '@vuepress/plugin-search'
import { copyCodePlugin } from 'vuepress-plugin-copy-code2'

const USER_NAME = 'Sun-ZhenXing'
const BASE_PATH = '/vuepress-opencv-notes/'

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
    ['link', { rel: 'icon', href: `${BASE_PATH}favicon.svg` }]
  ],
  base: BASE_PATH,
  theme: defaultTheme({
    logo: '/favicon.svg',
    repo: `${USER_NAME}${BASE_PATH}`,
    editLinkText: '在 GitHub 上编辑此页',
    contributorsText: '贡献者',
    lastUpdatedText: '上次更新',
    navbar: [
      {
        text: '合集',
        children: [
          {
            text: 'OpenCV4 计算机视觉项目实战笔记',
            link: '/learn-opencv-by-building-projects/'
          },
          {
            text: 'OpenCV 开发实践总结',
            link: '/opencv-development-practice/'
          },
        ]
      }
    ],
    sidebar: {
      '/learn-opencv-by-building-projects/': [
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
            {
              text: '附录',
              children: [
                '/learn-opencv-by-building-projects/appendix/',
                '/learn-opencv-by-building-projects/appendix/windows-errors.md',
              ]
            },
          ]
        }
      ],
      '/opencv-development-practice/': [
        {
          text: 'OpenCV 开发实践总结',
          children: [
            '/opencv-development-practice/dnn-deploy-yolov7/',
          ]
        }
      ]
    }
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
    }),
    copyCodePlugin({
      showInMobile: true,
    }),
  ]
})
