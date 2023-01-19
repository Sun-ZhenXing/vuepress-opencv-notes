import{_ as a,p as s,q as n,a1 as e}from"./framework-d3922052.js";const p={},t=e(`<h1 id="opencv-c-编译和构建指南" tabindex="-1"><a class="header-anchor" href="#opencv-c-编译和构建指南" aria-hidden="true">#</a> OpenCV C++ 编译和构建指南</h1><h2 id="_1-基本编译指南" tabindex="-1"><a class="header-anchor" href="#_1-基本编译指南" aria-hidden="true">#</a> 1. 基本编译指南</h2><p>需要预先安装编译依赖。</p><p>CMake 配置：</p><div class="language-bash" data-ext="sh"><pre class="language-bash"><code><span class="token function">mkdir</span> build
cmake <span class="token punctuation">..</span>
</code></pre></div><p>编译和安装：</p><div class="language-bash" data-ext="sh"><pre class="language-bash"><code><span class="token function">make</span> <span class="token parameter variable">-j</span> <span class="token variable"><span class="token variable">$(</span>nproc<span class="token variable">)</span></span>

<span class="token function">sudo</span> <span class="token function">make</span> <span class="token function">install</span>
</code></pre></div><h2 id="_2-构建-qt-支持" tabindex="-1"><a class="header-anchor" href="#_2-构建-qt-支持" aria-hidden="true">#</a> 2. 构建 Qt 支持</h2><p>如果需要编译基于 Qt 的图形界面，启用 <code>-D WITH_QT=ON</code> 来编译，例如：</p><div class="language-bash line-numbers-mode" data-ext="sh"><pre class="language-bash"><code>cmake <span class="token parameter variable">-D</span> <span class="token assign-left variable">CMAKE_BUILD_TYPE</span><span class="token operator">=</span>RELEASE <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">CMAKE_INSTALL_PREFIX</span><span class="token operator">=</span>/usr/local/opencv460 <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">INSTALL_C_EXAMPLE</span><span class="token operator">=</span>OFF <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">BUILD_EXAMPLE</span><span class="token operator">=</span>OFF <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">OPENCV_EXTRA_MODULES_PATH</span><span class="token operator">=</span><span class="token punctuation">..</span>/opencv_contrib-4.6.0/modules <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">WITH_V4L</span><span class="token operator">=</span>ON <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">WITH_TBB</span><span class="token operator">=</span>ON <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">WITH_VTK</span><span class="token operator">=</span>ON <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">WITH_GTK</span><span class="token operator">=</span>ON <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">WITH_OPENMP</span><span class="token operator">=</span>ON <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">WITH_OPENGL</span><span class="token operator">=</span>ON <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">WITH_QT</span><span class="token operator">=</span>ON <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">BUILD_opencv_python3</span><span class="token operator">=</span>OFF <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">BUILD_TIFF</span><span class="token operator">=</span>ON <span class="token punctuation">..</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>`,10),l=[t];function o(c,i){return s(),n("div",null,l)}const d=a(p,[["render",o],["__file","index.html.vue"]]);export{d as default};