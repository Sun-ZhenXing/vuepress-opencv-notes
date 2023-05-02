import process from 'node:process'
import { defineUserConfig, defaultTheme } from 'vuepress'
import { getDirname, path } from '@vuepress/utils'
import { mdEnhancePlugin } from 'vuepress-plugin-md-enhance'
import { docsearchPlugin } from '@vuepress/plugin-docsearch'
import { copyCodePlugin } from 'vuepress-plugin-copy-code2'
import { autoCatalogPlugin } from 'vuepress-plugin-auto-catalog'

const __dirname = getDirname(import.meta.url)
const isProd = process.env.NODE_ENV === 'production'
const ROOT_PATH = path.resolve(__dirname, '../..')
const CURRENT_PATH = path.resolve(__dirname, '.')
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
    },
    importCode: {
      handleImportPath: str => str
        .replace(/^\//, ROOT_PATH.replace(/(?:|\\|\/)$/, '/'))
        .replace(/^@/, CURRENT_PATH),
    },
  },
  theme: defaultTheme({
    logo: '/favicon.svg',
    repo: `${USER_NAME}${BASE_PATH}`,
    docsDir: 'docs',
    selectLanguageName: '简体中文',
    selectLanguageText: '选择语言',
    selectLanguageAriaLabel: '选择语言',
    editLinkText: '在 GitHub 上编辑此页',
    contributorsText: '贡献者',
    lastUpdatedText: '上次更新',
    openInNewWindow: '在新窗口打开',
    toggleColorMode: '切换颜色模式',
    toggleSidebar: '切换侧边栏',
    tip: '提示',
    warning: '注意',
    danger: '警告',
    notFound: [
      '这里什么都没有',
      '我们怎么到这来了？',
      '这是一个 404 页面',
      '看起来我们进入了错误的链接',
    ],
    backToHome: '返回首页',
    navbar: [
    ],
    sidebar: {
      '/learn-opencv-by-building-projects/': [
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
    },
    themePlugins: {
      git: isProd,
    },
  }),
  plugins: [
    mdEnhancePlugin({
      gfm: true,
      container: true,
      vPre: true,
      tabs: true,
      card: true,
      codetabs: true,
      include: {
        resolvePath: file => {
          if (file.startsWith('@'))
            return file.replace('@', CURRENT_PATH)
          return file
        },
      },
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
              attrs: { type: 'tip' },
              content: '定义'
            }
          }
        },
        {
          matcher: /@note:.+/,
          replacer: ({ tag, content }) => {
            if (tag === 'em') return {
              tag: 'Badge',
              attrs: { type: 'warning' },
              content: content.substring(6)
            }
          },
        },
      ],
    }, false),
    docsearchPlugin({
      appId: 'DF0MWQNCKW',
      apiKey: '3b6438cb1895eff367c5c84c8fa50383',
      indexName: 'alexsun_blog',
      placeholder: '搜索文档',
      searchParameters: {
        facetFilters: ['tags:opencv'],
      },
      locales: {
        '/': {
          placeholder: '搜索文档',
          translations: {
            button: {
              buttonText: '搜索文档',
              buttonAriaLabel: '搜索文档',
            },
            modal: {
              searchBox: {
                resetButtonTitle: '清除查询条件',
                resetButtonAriaLabel: '清除查询条件',
                cancelButtonText: '取消',
                cancelButtonAriaLabel: '取消',
              },
              startScreen: {
                recentSearchesTitle: '搜索历史',
                noRecentSearchesText: '没有搜索历史',
                saveRecentSearchButtonTitle: '保存至搜索历史',
                removeRecentSearchButtonTitle: '从搜索历史中移除',
                favoriteSearchesTitle: '收藏',
                removeFavoriteSearchButtonTitle: '从收藏中移除',
              },
              errorScreen: {
                titleText: '无法获取结果',
                helpText: '你可能需要检查你的网络连接',
              },
              footer: {
                selectText: '选择',
                navigateText: '切换',
                closeText: '关闭',
                searchByText: '搜索提供者',
              },
              noResultsScreen: {
                noResultsText: '无法找到相关结果',
                suggestedQueryText: '你可以尝试查询',
                reportMissingResultsText: '你认为该查询应该有结果？',
                reportMissingResultsLinkText: '点击反馈',
              },
            },
          },
        },
      },
    }),
    autoCatalogPlugin({
      orderGetter: ({ title, routeMeta }) => {
        if (routeMeta.order) return routeMeta.order as number
        const prefix = title.match(/^\d+. /)
        if (prefix) return parseInt(prefix[0])
        const suffix = title.match(/\d+$/)
        if (suffix) return parseInt(suffix[0])
        return 0
      }
    }),
    copyCodePlugin({
      showInMobile: true,
    }),
  ],
  alias: {
    '@': CURRENT_PATH,
  },
})
