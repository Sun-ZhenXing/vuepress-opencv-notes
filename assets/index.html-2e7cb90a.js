import{_ as t,M as e,p,q as c,R as s,t as o,N as l,a1 as n}from"./framework-8980b429.js";const i="/vuepress-opencv-notes/assets/2022-11-11-10-04-49-7333ffb0.webp",u={},k=n('<h1 id="_4-深入研究直方图和滤波器" tabindex="-1"><a class="header-anchor" href="#_4-深入研究直方图和滤波器" aria-hidden="true">#</a> 4. 深入研究直方图和滤波器</h1><p>本章介绍以下主题：</p><ul class="task-list-container"><li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" id="task-item-0" checked="checked" disabled="disabled"><label class="task-list-item-label" for="task-item-0"> 直方图和直方图均衡</label></li><li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" id="task-item-1" checked="checked" disabled="disabled"><label class="task-list-item-label" for="task-item-1"> 查找表</label></li><li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" id="task-item-2" checked="checked" disabled="disabled"><label class="task-list-item-label" for="task-item-2"> 模糊和中位数模糊</label></li><li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" id="task-item-3" checked="checked" disabled="disabled"><label class="task-list-item-label" for="task-item-3"> Canny 过滤器</label></li><li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" id="task-item-4" checked="checked" disabled="disabled"><label class="task-list-item-label" for="task-item-4"> 图像 - 颜色均衡</label></li><li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" id="task-item-5" checked="checked" disabled="disabled"><label class="task-list-item-label" for="task-item-5"> 图像类型之间的转换</label></li></ul><p>我们还会创建一个完整的应用程序，因此本章还将涵盖以下主题：</p><ul class="task-list-container"><li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" id="task-item-6" checked="checked" disabled="disabled"><label class="task-list-item-label" for="task-item-6"> 生成 CMake 脚本文件</label></li><li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" id="task-item-7" checked="checked" disabled="disabled"><label class="task-list-item-label" for="task-item-7"> 创建图像用户界面</label></li><li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" id="task-item-8" checked="checked" disabled="disabled"><label class="task-list-item-label" for="task-item-8"> 计算和绘制直方图</label></li><li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" id="task-item-9" checked="checked" disabled="disabled"><label class="task-list-item-label" for="task-item-9"> 直方图均衡</label></li><li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" id="task-item-10" checked="checked" disabled="disabled"><label class="task-list-item-label" for="task-item-10"> Lomography 相机效果</label></li><li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" id="task-item-11" checked="checked" disabled="disabled"><label class="task-list-item-label" for="task-item-11"> 卡通化效果</label></li></ul><p>该应用程序将帮助我们了解如何从头开始创建整个项目，并理解直方图的概念。</p><h2 id="_4-1-技术要求" tabindex="-1"><a class="header-anchor" href="#_4-1-技术要求" aria-hidden="true">#</a> 4.1 技术要求</h2>',7),d={class:"task-list-container"},r=s("li",{class:"task-list-item"},[s("input",{type:"checkbox",class:"task-list-item-checkbox",id:"task-item-12",checked:"checked",disabled:"disabled"}),s("label",{class:"task-list-item-label",for:"task-item-12"}," 熟悉 C++ 语言")],-1),b={class:"task-list-item"},m=s("input",{type:"checkbox",class:"task-list-item-checkbox",id:"task-item-13",checked:"checked",disabled:"disabled"},null,-1),v={class:"task-list-item-label",for:"task-item-13"},h={href:"https://github.com/PacktPublishing/Learn-OpenCV-4-By-Building-Projects-Second-Edition/tree/master/Chapter_04",target:"_blank",rel:"noopener noreferrer"},g=s("li",{class:"task-list-item"},[s("input",{type:"checkbox",class:"task-list-item-checkbox",id:"task-item-14",checked:"checked",disabled:"disabled"}),s("label",{class:"task-list-item-label",for:"task-item-14"}," 现代操作系统，例如 Ubuntu 20+ 或者 Windows 10+")],-1),y=n(`<h2 id="_4-2-生成-cmake-脚本文件" tabindex="-1"><a class="header-anchor" href="#_4-2-生成-cmake-脚本文件" aria-hidden="true">#</a> 4.2 生成 CMake 脚本文件</h2><p>基础的 <code>CMakeLists.txt</code> 文件：</p><div class="language-cmake line-numbers-mode" data-ext="cmake"><pre class="language-cmake"><code><span class="token keyword">cmake_minimum_required</span><span class="token punctuation">(</span><span class="token property">VERSION</span> <span class="token number">3.0</span><span class="token punctuation">)</span>

<span class="token keyword">project</span><span class="token punctuation">(</span>chapter_4_phototool<span class="token punctuation">)</span>

<span class="token keyword">set</span><span class="token punctuation">(</span><span class="token variable">CMAKE_CXX_STANDARD</span> <span class="token number">11</span><span class="token punctuation">)</span>

<span class="token keyword">find_package</span><span class="token punctuation">(</span>OpenCV REQUIRED<span class="token punctuation">)</span>
<span class="token keyword">message</span><span class="token punctuation">(</span><span class="token string">&quot;OpenCV version: <span class="token interpolation"><span class="token punctuation">\${</span><span class="token variable">OpenCV_VERSION</span><span class="token punctuation">}</span></span>&quot;</span><span class="token punctuation">)</span>

<span class="token keyword">include_directories</span><span class="token punctuation">(</span><span class="token punctuation">\${</span>OpenCV_INCLUDE_DIRS<span class="token punctuation">}</span><span class="token punctuation">)</span>
<span class="token keyword">link_directories</span><span class="token punctuation">(</span><span class="token punctuation">\${</span>OpenCV_LIBRARY_DIRS<span class="token punctuation">}</span><span class="token punctuation">)</span>

<span class="token keyword">add_executable</span><span class="token punctuation">(</span><span class="token punctuation">\${</span><span class="token variable">PROJECT_NAME</span><span class="token punctuation">}</span> main.cpp<span class="token punctuation">)</span>
<span class="token keyword">target_link_libraries</span><span class="token punctuation">(</span><span class="token punctuation">\${</span><span class="token variable">PROJECT_NAME</span><span class="token punctuation">}</span> <span class="token punctuation">\${</span>OpenCV_LIBS<span class="token punctuation">}</span><span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>本文的示例使用 C++ 11 标准。</p><h2 id="_4-3-创建图形用户界面" tabindex="-1"><a class="header-anchor" href="#_4-3-创建图形用户界面" aria-hidden="true">#</a> 4.3 创建图形用户界面</h2><p>我们将使用基于 Qt 的 GUI，应用程序接收一个输入参数来加载要处理的图像。此外我们还有四个按钮：</p><ul><li>Show histogram（展示直方图）</li><li>Equalize histogram（直方图均衡）</li><li>Lomography effect（LOMO 效果）</li><li>Cartoonize effect（卡通效果）</li></ul><p><img src="`+i+`" alt="" loading="lazy"></p><p>OpenCV 3.0 开始就包含了一个新的命令行解析器 <code>CommandLineParser</code>，首先我们先编写一个命令行解析图片路径的程序：</p><div class="language-cpp line-numbers-mode" data-ext="cpp"><pre class="language-cpp"><code><span class="token macro property"><span class="token directive-hash">#</span><span class="token directive keyword">include</span> <span class="token string">&lt;opencv2/core/utility.hpp&gt;</span></span>
<span class="token macro property"><span class="token directive-hash">#</span><span class="token directive keyword">include</span> <span class="token string">&lt;opencv2/highgui.hpp&gt;</span></span>
<span class="token macro property"><span class="token directive-hash">#</span><span class="token directive keyword">include</span> <span class="token string">&lt;opencv2/imgproc.hpp&gt;</span></span>

<span class="token macro property"><span class="token directive-hash">#</span><span class="token directive keyword">include</span> <span class="token string">&lt;iostream&gt;</span></span>
<span class="token macro property"><span class="token directive-hash">#</span><span class="token directive keyword">include</span> <span class="token string">&lt;string&gt;</span></span>
<span class="token macro property"><span class="token directive-hash">#</span><span class="token directive keyword">include</span> <span class="token string">&lt;vector&gt;</span></span>

<span class="token keyword">const</span> <span class="token keyword">char</span><span class="token operator">*</span> keys <span class="token operator">=</span> <span class="token punctuation">{</span>
    <span class="token string">&quot;{help h usage ? |      | print this message   }&quot;</span>
    <span class="token string">&quot;{@image         |&lt;none&gt;| image to process     }&quot;</span><span class="token punctuation">}</span><span class="token punctuation">;</span>

cv<span class="token double-colon punctuation">::</span>Mat img<span class="token punctuation">;</span>

<span class="token keyword">void</span> <span class="token function">showHistoCallback</span><span class="token punctuation">(</span><span class="token keyword">int</span> state<span class="token punctuation">,</span> <span class="token keyword">void</span><span class="token operator">*</span> userData<span class="token punctuation">)</span> <span class="token punctuation">{</span>
<span class="token punctuation">}</span>

<span class="token keyword">void</span> <span class="token function">equalizeCallback</span><span class="token punctuation">(</span><span class="token keyword">int</span> state<span class="token punctuation">,</span> <span class="token keyword">void</span><span class="token operator">*</span> userData<span class="token punctuation">)</span> <span class="token punctuation">{</span>
<span class="token punctuation">}</span>

<span class="token keyword">void</span> <span class="token function">lomoCallback</span><span class="token punctuation">(</span><span class="token keyword">int</span> state<span class="token punctuation">,</span> <span class="token keyword">void</span><span class="token operator">*</span> userData<span class="token punctuation">)</span> <span class="token punctuation">{</span>
<span class="token punctuation">}</span>

<span class="token keyword">void</span> <span class="token function">cartoonCallback</span><span class="token punctuation">(</span><span class="token keyword">int</span> state<span class="token punctuation">,</span> <span class="token keyword">void</span><span class="token operator">*</span> userData<span class="token punctuation">)</span> <span class="token punctuation">{</span>
<span class="token punctuation">}</span>

<span class="token keyword">int</span> <span class="token function">main</span><span class="token punctuation">(</span><span class="token keyword">int</span> argc<span class="token punctuation">,</span> <span class="token keyword">char</span><span class="token operator">*</span><span class="token operator">*</span> argv<span class="token punctuation">)</span> <span class="token punctuation">{</span>
    cv<span class="token double-colon punctuation">::</span>CommandLineParser <span class="token function">parser</span><span class="token punctuation">(</span>argc<span class="token punctuation">,</span> argv<span class="token punctuation">,</span> keys<span class="token punctuation">)</span><span class="token punctuation">;</span>
    parser<span class="token punctuation">.</span><span class="token function">about</span><span class="token punctuation">(</span><span class="token string">&quot;This program shows how to read an image from a file.&quot;</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
    <span class="token keyword">if</span> <span class="token punctuation">(</span>parser<span class="token punctuation">.</span><span class="token function">has</span><span class="token punctuation">(</span><span class="token string">&quot;help&quot;</span><span class="token punctuation">)</span><span class="token punctuation">)</span> <span class="token punctuation">{</span>
        parser<span class="token punctuation">.</span><span class="token function">printMessage</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
        <span class="token keyword">return</span> <span class="token number">0</span><span class="token punctuation">;</span>
    <span class="token punctuation">}</span>
    std<span class="token double-colon punctuation">::</span>string image_path <span class="token operator">=</span> parser<span class="token punctuation">.</span><span class="token generic-function"><span class="token function">get</span><span class="token generic class-name"><span class="token operator">&lt;</span>std<span class="token double-colon punctuation">::</span>string<span class="token operator">&gt;</span></span></span><span class="token punctuation">(</span><span class="token number">0</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
    <span class="token keyword">if</span> <span class="token punctuation">(</span><span class="token operator">!</span>parser<span class="token punctuation">.</span><span class="token function">check</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span> <span class="token punctuation">{</span>
        parser<span class="token punctuation">.</span><span class="token function">printErrors</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
        <span class="token keyword">return</span> <span class="token number">0</span><span class="token punctuation">;</span>
    <span class="token punctuation">}</span>
    img <span class="token operator">=</span> cv<span class="token double-colon punctuation">::</span><span class="token function">imread</span><span class="token punctuation">(</span>image_path<span class="token punctuation">,</span> cv<span class="token double-colon punctuation">::</span>IMREAD_COLOR<span class="token punctuation">)</span><span class="token punctuation">;</span>
    <span class="token keyword">if</span> <span class="token punctuation">(</span>img<span class="token punctuation">.</span><span class="token function">empty</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span> <span class="token punctuation">{</span>
        std<span class="token double-colon punctuation">::</span>cout <span class="token operator">&lt;&lt;</span> <span class="token string">&quot;Could not read the image: &quot;</span> <span class="token operator">&lt;&lt;</span> image_path <span class="token operator">&lt;&lt;</span> std<span class="token double-colon punctuation">::</span>endl<span class="token punctuation">;</span>
        <span class="token keyword">return</span> <span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">;</span>
    <span class="token punctuation">}</span>
    cv<span class="token double-colon punctuation">::</span><span class="token function">namedWindow</span><span class="token punctuation">(</span><span class="token string">&quot;Input&quot;</span><span class="token punctuation">,</span> cv<span class="token double-colon punctuation">::</span>WINDOW_AUTOSIZE<span class="token punctuation">)</span><span class="token punctuation">;</span>

    cv<span class="token double-colon punctuation">::</span><span class="token function">createButton</span><span class="token punctuation">(</span><span class="token string">&quot;Show histogram&quot;</span><span class="token punctuation">,</span> showHistoCallback<span class="token punctuation">,</span> <span class="token keyword">nullptr</span><span class="token punctuation">,</span> cv<span class="token double-colon punctuation">::</span>QT_PUSH_BUTTON<span class="token punctuation">,</span> <span class="token boolean">false</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
    cv<span class="token double-colon punctuation">::</span><span class="token function">createButton</span><span class="token punctuation">(</span><span class="token string">&quot;Equalize histogram&quot;</span><span class="token punctuation">,</span> equalizeCallback<span class="token punctuation">,</span> <span class="token keyword">nullptr</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
    cv<span class="token double-colon punctuation">::</span><span class="token function">createButton</span><span class="token punctuation">(</span><span class="token string">&quot;Lomography effect&quot;</span><span class="token punctuation">,</span> lomoCallback<span class="token punctuation">,</span> <span class="token keyword">nullptr</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
    cv<span class="token double-colon punctuation">::</span><span class="token function">createButton</span><span class="token punctuation">(</span><span class="token string">&quot;Cartoonize effect&quot;</span><span class="token punctuation">,</span> cartoonCallback<span class="token punctuation">,</span> <span class="token keyword">nullptr</span><span class="token punctuation">)</span><span class="token punctuation">;</span>

    cv<span class="token double-colon punctuation">::</span><span class="token function">imshow</span><span class="token punctuation">(</span><span class="token string">&quot;Input&quot;</span><span class="token punctuation">,</span> img<span class="token punctuation">)</span><span class="token punctuation">;</span>
    cv<span class="token double-colon punctuation">::</span><span class="token function">waitKey</span><span class="token punctuation">(</span><span class="token number">0</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
    <span class="token keyword">return</span> <span class="token number">0</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>我们有四个未完成的函数，下面我们将实现这四个函数。</p><h2 id="_4-4-绘制直方图" tabindex="-1"><a class="header-anchor" href="#_4-4-绘制直方图" aria-hidden="true">#</a> 4.4 绘制直方图</h2><div class="language-cpp" data-ext="cpp"><pre class="language-cpp"><code>
</code></pre></div>`,13);function _(f,w){const a=e("ExternalLinkIcon");return p(),c("div",null,[k,s("ul",d,[r,s("li",b,[m,s("label",v,[s("a",h,[o("本章代码"),l(a)])])]),g]),y])}const C=t(u,[["render",_],["__file","index.html.vue"]]);export{C as default};