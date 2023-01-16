import { defineUserConfig, defaultTheme } from 'vuepress'
import { mdEnhancePlugin } from 'vuepress-plugin-md-enhance'
import { searchProPlugin } from 'vuepress-plugin-search-pro'
import { copyCodePlugin } from 'vuepress-plugin-copy-code2'
import { autoCatalogPlugin } from 'vuepress-plugin-auto-catalog'

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
  markdown: {
    code: {
      lineNumbers: 10
    }
  },
  theme: defaultTheme({
    logo: '/favicon.svg',
    repo: `${USER_NAME}${BASE_PATH}`,
    docsDir: 'docs',
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
          {
            text: 'Python OpenCV 教程',
            link: '/opencv-python-tutorial/'
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
          ]
        }
      ],
      '/opencv-development-practice/': [
        {
          text: 'OpenCV 开发实践总结',
          children: [
            '/opencv-development-practice/basic-image-transform/',
            '/opencv-development-practice/common-geo-transform/',
            '/opencv-development-practice/complie-and-build-guide/',
            '/opencv-development-practice/dnn-deploy-yolov7/',
            '/opencv-development-practice/errors-in-windows/',
            '/opencv-development-practice/ps-operator-alternative-guide/',
            '/opencv-development-practice/skills-in-use/',
            '/opencv-development-practice/use-cmake-build-project/',
          ]
        }
      ],
      '/opencv-python-tutorial/': [
        {
          text: 'Python OpenCV 教程',
          children: [
            '/opencv-python-tutorial/chapter01/',
            '/opencv-python-tutorial/chapter02/',
            '/opencv-python-tutorial/chapter03/',
            '/opencv-python-tutorial/chapter04/',
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
      imgLazyload: true,
      tasklist: true,
      katex: true,
      mermaid: true,
      delay: 200,
      stylize: [
        {
          matcher: '@def',
          replacer: ({ tag }) => {
            if (tag === 'em') return {
              tag: 'Badge',
              attrs: { type: 'tip', vertical: 'middle' },
              content: '定义'
            }
          }
        }
      ]
    }),
    searchProPlugin({}),
    autoCatalogPlugin({}),
    copyCodePlugin({
      showInMobile: true
    })
  ]
})
