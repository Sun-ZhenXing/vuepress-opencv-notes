import{_ as c,M as p,p as v,q as d,R as a,t as s,N as n,V as t,a1 as b}from"./framework-546207d5.js";const k={},u=a("h1",{id:"opencv-c-编译和构建指南",tabindex:"-1"},[a("a",{class:"header-anchor",href:"#opencv-c-编译和构建指南","aria-hidden":"true"},"#"),s(" OpenCV C++ 编译和构建指南")],-1),m=a("h2",{id:"_1-基本编译指南",tabindex:"-1"},[a("a",{class:"header-anchor",href:"#_1-基本编译指南","aria-hidden":"true"},"#"),s(" 1. 基本编译指南")],-1),_={href:"https://github.com/opencv/opencv/releases",target:"_blank",rel:"noopener noreferrer"},D=a("code",null,"opencv-4.7.0.zip",-1),h={href:"https://github.com/opencv/opencv_contrib/tags",target:"_blank",rel:"noopener noreferrer"},F=a("code",null,"opencv_contrib-4.7.0.zip",-1),g=a("div",{class:"language-bash","data-ext":"sh"},[a("pre",{class:"language-bash"},[a("code",null,[a("span",{class:"token function"},"curl"),s(),a("span",{class:"token parameter variable"},"-Lj"),s(),a("span",{class:"token parameter variable"},"-o"),s(` opencv-4.7.0.zip https://github.com/opencv/opencv/archive/refs/tags/4.7.0.zip
`),a("span",{class:"token function"},"curl"),s(),a("span",{class:"token parameter variable"},"-Lj"),s(),a("span",{class:"token parameter variable"},"-o"),s(` opencv_contrib-4.7.0.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.7.0.zip
`),a("span",{class:"token function"},"unzip"),s(" opencv-4.7.0.zip "),a("span",{class:"token parameter variable"},"-d"),s(),a("span",{class:"token builtin class-name"},"."),s(`
`),a("span",{class:"token function"},"unzip"),s(" opencv_contrib-4.7.0.zip "),a("span",{class:"token parameter variable"},"-d"),s(),a("span",{class:"token builtin class-name"},"."),s(`
`),a("span",{class:"token builtin class-name"},"cd"),s(` opencv-4.7.0
`),a("span",{class:"token function"},"mkdir"),s(),a("span",{class:"token parameter variable"},"-p"),s(` build
`)])])],-1),I=a("div",{class:"language-bash","data-ext":"sh"},[a("pre",{class:"language-bash"},[a("code",null,[a("span",{class:"token function"},"curl"),s(),a("span",{class:"token parameter variable"},"-Lj"),s(),a("span",{class:"token parameter variable"},"-o"),s(` opencv-4.7.0.zip https://github.com/opencv/opencv/archive/refs/tags/4.7.0.zip
`),a("span",{class:"token function"},"curl"),s(),a("span",{class:"token parameter variable"},"-Lj"),s(),a("span",{class:"token parameter variable"},"-o"),s(` opencv_contrib-4.7.0.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.7.0.zip
7z x opencv-4.7.0.zip
7z x opencv_contrib-4.7.0.zip
`),a("span",{class:"token builtin class-name"},"cd"),s(` opencv-4.7.0
md build
`)])])],-1),O=b(`<p>需要预先安装编译依赖，Windows 系统见下文。例如 Ubuntu 的基本依赖为：</p><div class="language-bash" data-ext="sh"><pre class="language-bash"><code><span class="token function">sudo</span> <span class="token function">apt</span> <span class="token function">install</span> build-essential cmake <span class="token function">git</span> pkg-config libgtk-3-dev <span class="token punctuation">\\</span>
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev <span class="token punctuation">\\</span>
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev <span class="token punctuation">\\</span>
    gfortran openexr libatlas-base-dev python3-dev python3-numpy <span class="token punctuation">\\</span>
    libtbb2 libtbb-dev libdc1394-22-dev libopenexr-dev <span class="token punctuation">\\</span>
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
</code></pre></div><p>CMake 配置：</p><div class="language-bash" data-ext="sh"><pre class="language-bash"><code><span class="token function">mkdir</span> build
cmake <span class="token punctuation">..</span>
</code></pre></div><p>编译和安装：</p><div class="language-bash" data-ext="sh"><pre class="language-bash"><code><span class="token function">make</span> <span class="token parameter variable">-j</span> <span class="token variable"><span class="token variable">$(</span>nproc<span class="token variable">)</span></span>

<span class="token function">sudo</span> <span class="token function">make</span> <span class="token function">install</span>
</code></pre></div><h2 id="_2-构建-qt-支持" tabindex="-1"><a class="header-anchor" href="#_2-构建-qt-支持" aria-hidden="true">#</a> 2. 构建 Qt 支持</h2><p>如果需要编译基于 Qt 的图形界面，启用 <code>-D WITH_QT=ON</code> 来编译，例如：</p><div class="language-bash line-numbers-mode" data-ext="sh"><pre class="language-bash"><code>cmake <span class="token parameter variable">-D</span> <span class="token assign-left variable">CMAKE_BUILD_TYPE</span><span class="token operator">=</span>RELEASE <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">CMAKE_INSTALL_PREFIX</span><span class="token operator">=</span>/usr/local/opencv470 <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">INSTALL_C_EXAMPLE</span><span class="token operator">=</span>OFF <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">BUILD_EXAMPLE</span><span class="token operator">=</span>OFF <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">OPENCV_EXTRA_MODULES_PATH</span><span class="token operator">=</span><span class="token punctuation">..</span>/opencv_contrib-4.7.0/modules <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">WITH_V4L</span><span class="token operator">=</span>ON <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">WITH_TBB</span><span class="token operator">=</span>ON <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">WITH_VTK</span><span class="token operator">=</span>ON <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">WITH_GTK</span><span class="token operator">=</span>ON <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">WITH_OPENMP</span><span class="token operator">=</span>ON <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">WITH_OPENGL</span><span class="token operator">=</span>ON <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">WITH_QT</span><span class="token operator">=</span>ON <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">BUILD_opencv_python3</span><span class="token operator">=</span>OFF <span class="token punctuation">\\</span>
    <span class="token parameter variable">-D</span> <span class="token assign-left variable">BUILD_TIFF</span><span class="token operator">=</span>ON <span class="token punctuation">..</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h2 id="_3-windows-编译指南" tabindex="-1"><a class="header-anchor" href="#_3-windows-编译指南" aria-hidden="true">#</a> 3. Windows 编译指南</h2><p>Windows 编译要求：</p><ul><li>Visual Studio 2019，并安装 C++ 桌面开发基本组件</li><li>CMake GUI 版本，也可以完全使用命令行</li></ul><p>在配置过程中将会自动下载依赖。</p><p>建议配置项：</p><ul><li>CUDA 支持，如果安装了 CUDA 可选择支持，不过需要选择你的显卡对应的算力</li><li><code>BUILD_opencv_world</code> 编译包含所有库的动态链接库</li><li>Test 选项可去除</li><li>Python/Java/JS 选项去除</li></ul><div class="hint-container info"><p class="hint-container-title">使用代理</p><p>使用大陆的网络可能无法下载依赖，需要使用代理。在 Environment 中增加 <code>http_proxy</code> 和 <code>https_proxy</code> 变量，并将其指定为一个可用的代理地址。例如：</p><div class="language-yaml" data-ext="yml"><pre class="language-yaml"><code><span class="token key atrule">http_proxy</span><span class="token punctuation">:</span> <span class="token string">&#39;http://127.0.0.1:10809&#39;</span>
<span class="token key atrule">https_proxy</span><span class="token punctuation">:</span> <span class="token string">&#39;socks5://127.0.0.1:10808&#39;</span>
</code></pre></div><p>在 Windows 上，Python 可能对 SOCKS 支持不完备，如果使用 SOCKS 协议时出现无法握手或者无法使用 DNS，将 <code>https_proxy</code> 代替为 <code>http_proxy</code> 的值即可：</p><div class="language-yaml" data-ext="yml"><pre class="language-yaml"><code><span class="token key atrule">http_proxy</span><span class="token punctuation">:</span> <span class="token string">&#39;http://127.0.0.1:10809&#39;</span>
<span class="token key atrule">https_proxy</span><span class="token punctuation">:</span> <span class="token string">&#39;http://127.0.0.1:10809&#39;</span>
</code></pre></div></div><details class="hint-container details"><summary>命令行编译指南</summary><div class="language-bash line-numbers-mode" data-ext="sh"><pre class="language-bash"><code>cmake <span class="token parameter variable">-G</span> <span class="token string">&quot;Visual Studio 16 2019&quot;</span> <span class="token parameter variable">-T</span> <span class="token assign-left variable">host</span><span class="token operator">=</span>x64 <span class="token parameter variable">-A</span> x64 ^
    <span class="token parameter variable">-DBUILD_DOCS</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DBUILD_SHARED_LIBS</span><span class="token operator">=</span>ON ^
    <span class="token parameter variable">-DBUILD_FAT_JAVA_LIB</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DBUILD_TESTS</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DBUILD_TIFF</span><span class="token operator">=</span>ON ^
    <span class="token parameter variable">-DBUILD_JASPER</span><span class="token operator">=</span>ON ^
    <span class="token parameter variable">-DBUILD_JPEG</span><span class="token operator">=</span>ON ^
    <span class="token parameter variable">-DBUILD_PNG</span><span class="token operator">=</span>ON ^
    <span class="token parameter variable">-DBUILD_ZLIB</span><span class="token operator">=</span>ON ^
    <span class="token parameter variable">-DBUILD_OPENEXR</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DBUILD_opencv_apps</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DBUILD_opencv_calib3d</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DBUILD_opencv_contrib</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DBUILD_opencv_features2d</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DBUILD_opencv_flann</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DBUILD_opencv_gpu</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DBUILD_opencv_java</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DBUILD_opencv_legacy</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DBUILD_opencv_ml</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DBUILD_opencv_nonfree</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DBUILD_opencv_objdetect</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DBUILD_opencv_ocl</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DBUILD_opencv_photo</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DBUILD_opencv_python</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DBUILD_opencv_stitching</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DBUILD_opencv_superres</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DBUILD_opencv_ts</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DBUILD_opencv_video</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DBUILD_opencv_videostab</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DBUILD_opencv_world</span><span class="token operator">=</span>ON ^
    <span class="token parameter variable">-DBUILD_opencv_lengcy</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DBUILD_opencv_lengcy</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DWITH_1394</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DWITH_EIGEN</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DWITH_FFMPEG</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DWITH_GIGEAPI</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DWITH_GSTREAMER</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DWITH_GTK</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DWITH_PVAPI</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DWITH_V4L</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DWITH_LIBV4L</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DWITH_CUDA</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DWITH_CUFFT</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DWITH_OPENCL</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DWITH_OPENCLAMDBLAS</span><span class="token operator">=</span>OFF ^
    <span class="token parameter variable">-DWITH_OPENCLAMDFFT</span><span class="token operator">=</span>OFF <span class="token punctuation">..</span>
cmake <span class="token parameter variable">--build</span> <span class="token builtin class-name">.</span> <span class="token parameter variable">--config</span> Release <span class="token parameter variable">--target</span> ALL_BUILD <span class="token parameter variable">-j</span> <span class="token number">20</span> --
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div></details>`,17);function L(f,B){const e=p("ExternalLinkIcon"),l=p("CodeTabs");return v(),d("div",null,[u,m,a("p",null,[s("下面都以 OpenCV 4.7.0 的编译为例。下载源代码 "),a("a",_,[D,n(e)]),s(" 和 "),a("a",h,[F,n(e)]),s("，解压到同一个文件夹。")]),n(l,{id:"9",data:[{title:"Linux"},{title:"Windows"}]},{tab0:t(({title:o,value:r,isActive:i})=>[g]),tab1:t(({title:o,value:r,isActive:i})=>[I]),_:1}),O])}const U=c(k,[["render",L],["__file","index.html.vue"]]);export{U as default};
