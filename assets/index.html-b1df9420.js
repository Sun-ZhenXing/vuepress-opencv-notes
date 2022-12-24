import{_ as t,M as r,p as c,q as o,R as a,t as e,N as i,a1 as s}from"./framework-35149b8f.js";const d={},l=s('<h1 id="_1-opencv-入门" tabindex="-1"><a class="header-anchor" href="#_1-opencv-入门" aria-hidden="true">#</a> 1. OpenCV 入门</h1><p>计算机视觉应用程序很有趣，而且很有用，但是其底层算法是计算密集型的。</p><p>OpenCV 库提供实时高效的计算机视觉算法，已经是该领域的事实标准。它可以在几乎所有平台上使用。</p><p>本节将主要关注下面几个主题：</p><ul class="task-list-container"><li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" id="task-item-0" checked="checked" disabled="disabled"><label class="task-list-item-label" for="task-item-0"> 人类如何处理视觉数据，如何理解图像内容</label></li><li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" id="task-item-1" checked="checked" disabled="disabled"><label class="task-list-item-label" for="task-item-1"> OpenCV 能做什么，OpenCV 中可以用于实现这些目标的各种模块是什么</label></li><li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" id="task-item-2" checked="checked" disabled="disabled"><label class="task-list-item-label" for="task-item-2"> 如何在 Windows、Linux 和 Mac OS 上安装 OpenCV</label></li></ul><h2 id="_1-1-了解人类视觉系统" tabindex="-1"><a class="header-anchor" href="#_1-1-了解人类视觉系统" aria-hidden="true">#</a> 1.1 了解人类视觉系统</h2><p>人眼可以捕获视野内所有信息，例如 <strong>颜色</strong>、<strong>形状</strong>、<strong>亮度</strong>。</p><p>一些事实：</p><ul><li>人眼的视觉系统对低频的内容更敏感，而高度纹理化的表面人眼不容易发现目标</li><li>人眼对于亮度的变化比对颜色更敏感</li><li>人眼对于运动的事物很敏感，即使没有被直接看到</li><li>人类擅长记忆对象的特征点，以便在下一次看见它的时候能够快速识别出来</li></ul><p><strong>人类视觉系统</strong>（HVS）模型的一些事实：</p>',10),p=a("ul",null,[a("li",null,[e("对颜色："),a("span",{class:"katex"},[a("span",{class:"katex-mathml"},[a("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[a("semantics",null,[a("mrow",null,[a("mn",null,"30"),a("mi",{mathvariant:"normal"},"°"),a("mo",null,"∼"),a("mn",null,"60"),a("mi",{mathvariant:"normal"},"°")]),a("annotation",{encoding:"application/x-tex"},"30\\degree \\sim 60\\degree")])])]),a("span",{class:"katex-html","aria-hidden":"true"},[a("span",{class:"base"},[a("span",{class:"strut",style:{height:"0.6944em"}}),a("span",{class:"mord"},"30°"),a("span",{class:"mspace",style:{"margin-right":"0.2778em"}}),a("span",{class:"mrel"},"∼"),a("span",{class:"mspace",style:{"margin-right":"0.2778em"}})]),a("span",{class:"base"},[a("span",{class:"strut",style:{height:"0.6944em"}}),a("span",{class:"mord"},"60°")])])])]),a("li",null,[e("对形状："),a("span",{class:"katex"},[a("span",{class:"katex-mathml"},[a("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[a("semantics",null,[a("mrow",null,[a("mn",null,"5"),a("mi",{mathvariant:"normal"},"°"),a("mo",null,"∼"),a("mn",null,"30"),a("mi",{mathvariant:"normal"},"°")]),a("annotation",{encoding:"application/x-tex"},"5\\degree \\sim 30\\degree")])])]),a("span",{class:"katex-html","aria-hidden":"true"},[a("span",{class:"base"},[a("span",{class:"strut",style:{height:"0.6944em"}}),a("span",{class:"mord"},"5°"),a("span",{class:"mspace",style:{"margin-right":"0.2778em"}}),a("span",{class:"mrel"},"∼"),a("span",{class:"mspace",style:{"margin-right":"0.2778em"}})]),a("span",{class:"base"},[a("span",{class:"strut",style:{height:"0.6944em"}}),a("span",{class:"mord"},"30°")])])])]),a("li",null,[e("对文本："),a("span",{class:"katex"},[a("span",{class:"katex-mathml"},[a("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[a("semantics",null,[a("mrow",null,[a("mn",null,"5"),a("mi",{mathvariant:"normal"},"°"),a("mo",null,"∼"),a("mn",null,"10"),a("mi",{mathvariant:"normal"},"°")]),a("annotation",{encoding:"application/x-tex"},"5\\degree \\sim 10\\degree")])])]),a("span",{class:"katex-html","aria-hidden":"true"},[a("span",{class:"base"},[a("span",{class:"strut",style:{height:"0.6944em"}}),a("span",{class:"mord"},"5°"),a("span",{class:"mspace",style:{"margin-right":"0.2778em"}}),a("span",{class:"mrel"},"∼"),a("span",{class:"mspace",style:{"margin-right":"0.2778em"}})]),a("span",{class:"base"},[a("span",{class:"strut",style:{height:"0.6944em"}}),a("span",{class:"mord"},"10°")])])])])],-1),h=s('<h2 id="_1-2-人类如何理解图像内容" tabindex="-1"><a class="header-anchor" href="#_1-2-人类如何理解图像内容" aria-hidden="true">#</a> 1.2 人类如何理解图像内容</h2><p>人在看见一个椅子，立刻就能识别出它是椅子。而计算机执行这项任务缺异常困难，研究人员还在不断研究为什么计算机在这方面做的没有我们好。</p><p>人的视觉数据处理发生在腹侧视觉流中，它基本上是大脑中的一个区域层次结构，帮助我们识别对象。</p><p>人类在识别对象时产生了某种不变性。大脑会自动提取对象的特征并以某种方式储存起来。这种方式不受视角、方向、大小和光照等因素影响。</p><p>腹侧流更进一步是更复杂的细胞，它们被训练为识别更加复杂的对象。</p><h3 id="为什么机器难以理解图像内容" tabindex="-1"><a class="header-anchor" href="#为什么机器难以理解图像内容" aria-hidden="true">#</a> 为什么机器难以理解图像内容</h3><p>机器并不容易做到这一点，如果换一个角度拍摄图像，图片将产生很大的不同。图片有很多变化，如形状、大小、视角、角度、照明、遮挡。计算机不可能记住对象的所有可能的特征，我们需要一些实用算法来处理。</p><p>当对象的一部分被遮挡时，计算机仍然无法识别它，计算机可能认为这是一个新对象。因此，我们要想构建底层功能模块，就需要组合不同的处理算法来形成更复杂的算法。</p><p>OpenCV 提供了很多这样的功能，它们经过高度优化，一旦了解了 OpenCV 的功能，就可以构建更加复杂的应用程序。</p><h2 id="_1-3-你能用-opencv-做什么" tabindex="-1"><a class="header-anchor" href="#_1-3-你能用-opencv-做什么" aria-hidden="true">#</a> 1.3 你能用 OpenCV 做什么</h2><h3 id="_1-3-1-内置数据结构和输入-输出" tabindex="-1"><a class="header-anchor" href="#_1-3-1-内置数据结构和输入-输出" aria-hidden="true">#</a> 1.3.1 内置数据结构和输入/输出</h3><p><code>imgcodecs</code> 模块包含了图像的读取和写入。</p><p><code>videoio</code> 模块处理视频的输入输出操作，可以从文件或者摄像头读取，甚至获取或设置帧率、帧大小等信息。</p><h3 id="_1-3-2-图像处理操作" tabindex="-1"><a class="header-anchor" href="#_1-3-2-图像处理操作" aria-hidden="true">#</a> 1.3.2 图像处理操作</h3><p><code>imgproc</code> 模块提供了图像处理操作，例如 <strong>图像过滤</strong>、<strong>形态学操作</strong>、<strong>几何变换</strong>、<strong>颜色转换</strong>、<strong>图像绘制</strong>、<strong>直方图</strong>、<strong>形状分析</strong>、<strong>运动分析</strong>、<strong>特征检测</strong> 等。</p><p><code>ximgproc</code> 模块包含高级图像处理算法，可以用于如 <strong>结构化森林的边缘检测</strong>、<strong>域变换滤波器</strong>、<strong>自适应流形滤波器</strong> 等处理。</p><h3 id="_1-3-3-gui" tabindex="-1"><a class="header-anchor" href="#_1-3-3-gui" aria-hidden="true">#</a> 1.3.3 GUI</h3><p><code>highgui</code> 模块，用于处理 GUI。</p><h3 id="_1-3-4-视频分析" tabindex="-1"><a class="header-anchor" href="#_1-3-4-视频分析" aria-hidden="true">#</a> 1.3.4 视频分析</h3><p><code>video</code> 模块可以 <strong>分析连续帧之间的运动</strong>，<strong>目标追踪</strong>，创建 <strong>视频监控模型</strong> 等任务。</p><p><code>videostab</code> 模块用于处理视频稳定性问题。</p><h3 id="_1-3-5-3d-重建" tabindex="-1"><a class="header-anchor" href="#_1-3-5-3d-重建" aria-hidden="true">#</a> 1.3.5 3D 重建</h3><p><code>calib3d</code> 计算 2D 图像中各种对象之间的关系，计算其 3D 位置，该模块也可用于摄像机校准。</p><h3 id="_1-3-6-特征提取" tabindex="-1"><a class="header-anchor" href="#_1-3-6-特征提取" aria-hidden="true">#</a> 1.3.6 特征提取</h3><p><code>features2d</code> 的 OpenCV 模块提供了特征检测和提取功能，例如 <strong>尺度不变特征转换</strong>（Scale Invariant Feature Transform, SIFT）算法、<strong>加速鲁棒特征</strong>（Speeded Up Robust Features, SURF）和 <strong>加速分段测试特征</strong>（Features From Accelerated Segment Test, FAST）等算法。</p><p><code>xfeatures2d</code> 提供了更多特征提取器，其中一部分仍然处于实验阶段。</p><p><code>bioinspired</code> 模块提供了受到生物学启发的计算机视觉模型算法。</p><h3 id="_1-3-7-对象检测" tabindex="-1"><a class="header-anchor" href="#_1-3-7-对象检测" aria-hidden="true">#</a> 1.3.7 对象检测</h3><p>对象检测是检测图像中对象的位置，与对象类型无关。</p><p><code>objdetect</code> 和 <code>xobjdetect</code> 提供了对象检测器框架。</p><h3 id="_1-3-8-机器学习" tabindex="-1"><a class="header-anchor" href="#_1-3-8-机器学习" aria-hidden="true">#</a> 1.3.8 机器学习</h3><p><code>ml</code> 模块提供了很多机器学习方法，例如 <strong>贝叶斯分类器</strong>（Bayes Classifier）、<strong>K 邻近</strong>（KNN）、支持向量机（SVM）、决策树（Decision Tree）、<strong>神经网络</strong>（Neural Network）等。</p><p>此外，它还包含了 <strong>快速近似最临近搜索库</strong>，即 <code>flann</code>，用于在大型数据集中进行快速最近近搜索的算法。</p><h3 id="_1-3-9-计算摄影" tabindex="-1"><a class="header-anchor" href="#_1-3-9-计算摄影" aria-hidden="true">#</a> 1.3.9 计算摄影</h3><p><code>photo</code> 和 <code>xphoto</code> 提供了计算摄影学有关的算法，包括 HDR 成像、图像补光、光场相机等算法。</p><p><code>stitching</code> 模块提供创建全景图像的算法。</p><h3 id="_1-3-10-形状分析" tabindex="-1"><a class="header-anchor" href="#_1-3-10-形状分析" aria-hidden="true">#</a> 1.3.10 形状分析</h3><p><code>shape</code> 模块提供了提取形状、测量相似性、转换对象形状等操作相关的算法。</p><h3 id="_1-3-11-光流算法" tabindex="-1"><a class="header-anchor" href="#_1-3-11-光流算法" aria-hidden="true">#</a> 1.3.11 光流算法</h3><p>光流算法用于在视频中跟踪连续帧中的特征。</p><p><code>optflow</code> 模块用于执行光流操作，<code>tracking</code> 模块包含跟踪特征的更多算法。</p><h3 id="_1-3-12-人脸和对象识别" tabindex="-1"><a class="header-anchor" href="#_1-3-12-人脸和对象识别" aria-hidden="true">#</a> 1.3.12 人脸和对象识别</h3><p><code>face</code> 用于识别人脸的位置。</p><p><code>saliency</code> 模块用于检测静态图像和视频中的显著区域。</p><h3 id="_1-3-13-表面匹配" tabindex="-1"><a class="header-anchor" href="#_1-3-13-表面匹配" aria-hidden="true">#</a> 1.3.13 表面匹配</h3><p><code>surface_matching</code> 用于 3D 对象的识别方法，以及 3D 特征姿势估计算法。</p><h3 id="_1-3-14-文本检测和识别" tabindex="-1"><a class="header-anchor" href="#_1-3-14-文本检测和识别" aria-hidden="true">#</a> 1.3.14 文本检测和识别</h3><p><code>text</code> 模块包含了含有文字的检测和识别的算法。</p><h3 id="_1-3-15-深度学习" tabindex="-1"><a class="header-anchor" href="#_1-3-15-深度学习" aria-hidden="true">#</a> 1.3.15 深度学习</h3><p><code>dnn</code> 模块包含了深度学习相关的内容，包括 TensorFlow、Caffe、ONNX 等模型的导入器和部分网络的推理。</p><h2 id="_1-4-安装-opencv" tabindex="-1"><a class="header-anchor" href="#_1-4-安装-opencv" aria-hidden="true">#</a> 1.4 安装 OpenCV</h2>',51),g={href:"https://opencv.org/releases/",target:"_blank",rel:"noopener noreferrer"},m=s(`<h3 id="_1-4-1-windows" tabindex="-1"><a class="header-anchor" href="#_1-4-1-windows" aria-hidden="true">#</a> 1.4.1 Windows</h3><div class="custom-container tip"><p class="custom-container-title">Windows 预编译版本</p><p>目前 OpenCV 提供 64 位不同版本的 OpenCV，这些不同版本的 OpenCV 预编译包是在不同版本的 Visual Studio 中编译的。如果你需要 32 位的 OpenCV 或者兼容更高版本的 VS，则需要自己编译源代码。</p></div><p>例如，你将 OpenCV 安装到 <code>D:\\OpenCV4.6\\</code> 文件夹下。</p><p>可以使用下面的命令创建变量：</p><div class="language-bat" data-ext="bat"><pre class="language-bat"><code>setx -m OPENCV_DIR D:\\OpenCV4.6\\opencv\\build\\x64\\vc15
</code></pre></div><p>安装后，在路径（PATH）上加入以下路径 <code>%OPENCV_DIR%/bin</code>。</p><p>也可以将完整 <code>bin/</code> 路径加入 PATH：</p><div class="language-bat" data-ext="bat"><pre class="language-bat"><code>D:\\OpenCV4.6\\opencv\\build\\x64\\vc15\\bin
</code></pre></div><h3 id="_1-4-2-mac-os" tabindex="-1"><a class="header-anchor" href="#_1-4-2-mac-os" aria-hidden="true">#</a> 1.4.2 Mac OS</h3><p>需要安装 CMake，可以从官网查找发行版，如果不满足可以从源码开始编译，过程大致和 Linux 类似。</p><h3 id="_1-4-3-linux" tabindex="-1"><a class="header-anchor" href="#_1-4-3-linux" aria-hidden="true">#</a> 1.4.3 Linux</h3><p>可以编译安装，可以先安装编译依赖项，然后下载构建即可。</p><p>在编译前需要预先安装一些依赖库，请检查最新版本要求。</p><div class="language-bash" data-ext="sh"><pre class="language-bash"><code><span class="token comment"># 去官网或者 GitHub 查找下载最新版</span>
<span class="token function">wget</span> <span class="token string">&quot;https://github.com/opencv/opencv/&lt;最新版&gt;.tar.gz&quot;</span> <span class="token parameter variable">-O</span> opencv.tar.gz
<span class="token function">wget</span> <span class="token string">&quot;https://github.com/opencv/opencv_contrib/&lt;最新版&gt;.tar.gz&quot;</span> <span class="token parameter variable">-O</span> opencv_contrib.tar.gz
<span class="token function">tar</span> <span class="token parameter variable">-zxvf</span> opencv.tar.gz
<span class="token function">tar</span> <span class="token parameter variable">-xzvf</span> opencv_contrib.tar.gz
</code></pre></div><p>进入各自目录，然后选择合适的参数，使用 CMake 编译：</p><div class="language-bash" data-ext="sh"><pre class="language-bash"><code>cmake -D<span class="token operator">&lt;</span>各种配置<span class="token operator">&gt;</span> <span class="token punctuation">..</span>/
<span class="token function">make</span> <span class="token parameter variable">-j</span> <span class="token number">4</span>
<span class="token function">sudo</span> <span class="token function">make</span> <span class="token function">install</span>
</code></pre></div><p>然后复制配置文件：</p><div class="language-bash" data-ext="sh"><pre class="language-bash"><code><span class="token function">cp</span> <span class="token operator">&lt;</span>路径<span class="token operator">&gt;</span>/build/lib/pkgconfig/opencv.pc /usr/local/lib/pkgconfig.opencv4.pc
</code></pre></div><h2 id="_1-5-总结" tabindex="-1"><a class="header-anchor" href="#_1-5-总结" aria-hidden="true">#</a> 1.5 总结</h2><p>本章讨论了计算机视觉系统，以及人类如何处理视觉数据。解释了为什么机器难以做到这一点，以及在设计计算机视觉库时需要考虑的因素。</p><p>我们学习了 OpenCV 可以完成的工作，以及可用于完成这些任务的各种模块。最后学习了如何在各种操作系统中安装 OpenCV。</p>`,21);function u(b,_){const n=r("ExternalLinkIcon");return c(),o("div",null,[l,p,h,a("p",null,[e("OpenCV 发行版下载："),a("a",g,[e("OpenCV Releases"),i(n)]),e("。")]),m])}const k=t(d,[["render",u],["__file","index.html.vue"]]);export{k as default};
