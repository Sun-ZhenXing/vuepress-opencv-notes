import{_ as o,r as l,o as c,c as t,b as s,a as e,w as p,d as n,e as r}from"./app-e045e1fd.js";const d={},i=s("h1",{id:"_1-安装-opencv",tabindex:"-1"},[s("a",{class:"header-anchor",href:"#_1-安装-opencv","aria-hidden":"true"},"#"),n(" 1. 安装 OpenCV")],-1),h={class:"table-of-contents"},y=r(`<h2 id="_1-1-opencv-的-python-默认发行版" tabindex="-1"><a class="header-anchor" href="#_1-1-opencv-的-python-默认发行版" aria-hidden="true">#</a> 1.1 OpenCV 的 Python 默认发行版</h2><p>OpenCV 提供 PyPI 发行的 Python 包，使用 Python 的二进制扩展并且是预编译的。</p><div class="hint-container info"><p class="hint-container-title">下载预编译包</p><p>PyPI 目前只提供 CPU 版本的预编译包，如果你需要 CUDA 支持或其他架构支持的 OpenCV 可以查阅其他包管理工具或自行编译。</p></div><p>如果你的 Python 包管理器内没有安装 OpenCV，可以使用下面的命令直接安装：</p><div class="language-bash" data-ext="sh"><pre class="shiki dark-plus" style="background-color:#1E1E1E;" tabindex="0"><code><span class="line"><span style="color:#DCDCAA;">pip</span><span style="color:#D4D4D4;"> </span><span style="color:#CE9178;">install</span><span style="color:#D4D4D4;"> </span><span style="color:#CE9178;">opencv-python</span></span>
<span class="line"></span></code></pre></div><p>也可以指定版本：</p><div class="language-bash" data-ext="sh"><pre class="shiki dark-plus" style="background-color:#1E1E1E;" tabindex="0"><code><span class="line"><span style="color:#DCDCAA;">pip</span><span style="color:#D4D4D4;"> </span><span style="color:#CE9178;">install</span><span style="color:#D4D4D4;"> </span><span style="color:#CE9178;">opencv-python==</span><span style="color:#B5CEA8;">4.7</span><span style="color:#CE9178;">.0.68</span></span>
<span class="line"></span></code></pre></div><p>如果你使用 Anaconda，那么 OpenCV 已经默认安装，如果你想更新可以使用：</p><div class="language-bash" data-ext="sh"><pre class="shiki dark-plus" style="background-color:#1E1E1E;" tabindex="0"><code><span class="line"><span style="color:#DCDCAA;">pip</span><span style="color:#D4D4D4;"> </span><span style="color:#CE9178;">install</span><span style="color:#D4D4D4;"> </span><span style="color:#569CD6;">-U</span><span style="color:#D4D4D4;"> </span><span style="color:#CE9178;">opencv-python</span></span>
<span class="line"><span style="color:#6A9955;"># 或者</span></span>
<span class="line"><span style="color:#DCDCAA;">conda</span><span style="color:#D4D4D4;"> </span><span style="color:#CE9178;">update</span><span style="color:#D4D4D4;"> </span><span style="color:#CE9178;">opencv-python</span></span>
<span class="line"></span></code></pre></div><h2 id="_1-2-安装扩展包和无-gui-版本的-opencv" tabindex="-1"><a class="header-anchor" href="#_1-2-安装扩展包和无-gui-版本的-opencv" aria-hidden="true">#</a> 1.2 安装扩展包和无 GUI 版本的 OpenCV</h2><p>如果需要 OpenCV Contrib 模块中包含的算法，需要安装 <code>opencv-contrib-python</code>，安装命令如下：</p><div class="language-bash" data-ext="sh"><pre class="shiki dark-plus" style="background-color:#1E1E1E;" tabindex="0"><code><span class="line"><span style="color:#DCDCAA;">pip</span><span style="color:#D4D4D4;"> </span><span style="color:#CE9178;">install</span><span style="color:#D4D4D4;"> </span><span style="color:#CE9178;">opencv-contrib-python</span></span>
<span class="line"></span></code></pre></div><p>如果你在 <code>libGL</code> 支持不完备的系统（通常是无桌面的系统）上安装或使用 <code>opencv-python</code>，可能出现错误，可以安装无 GUI 支持的 OpenCV Headless 版本：</p><div class="language-bash" data-ext="sh"><pre class="shiki dark-plus" style="background-color:#1E1E1E;" tabindex="0"><code><span class="line"><span style="color:#DCDCAA;">pip</span><span style="color:#D4D4D4;"> </span><span style="color:#CE9178;">install</span><span style="color:#D4D4D4;"> </span><span style="color:#CE9178;">opencv-python-headless</span></span>
<span class="line"></span></code></pre></div><p>通常 Headless 版本用于服务器上使用。</p><p>同样，<code>opencv-contrib-python</code> 也提供 Headless 版本 <code>opencv-contrib-python-headless</code>：</p><div class="language-bash" data-ext="sh"><pre class="shiki dark-plus" style="background-color:#1E1E1E;" tabindex="0"><code><span class="line"><span style="color:#DCDCAA;">pip</span><span style="color:#D4D4D4;"> </span><span style="color:#CE9178;">install</span><span style="color:#D4D4D4;"> </span><span style="color:#CE9178;">opencv-contrib-python-headless</span></span>
<span class="line"></span></code></pre></div><div class="hint-container warning"><p class="hint-container-title">版本一致</p><p><code>opencv-python</code> 和 <code>opencv-python-headless</code> 不能一起安装，否则导入包时产生冲突，另一个包无法被加载。安装 Contrib 版本也要和默认的 OpenCV 版本一致，否则会出现不兼容的问题。</p></div><h2 id="_1-3-使用其他包管理器" tabindex="-1"><a class="header-anchor" href="#_1-3-使用其他包管理器" aria-hidden="true">#</a> 1.3 使用其他包管理器</h2><p>在 Debian/Ubuntu 上使用，可以使用系统的包管理器安装：</p><div class="language-bash" data-ext="sh"><pre class="shiki dark-plus" style="background-color:#1E1E1E;" tabindex="0"><code><span class="line"><span style="color:#DCDCAA;">sudo</span><span style="color:#D4D4D4;"> </span><span style="color:#CE9178;">apt-get</span><span style="color:#D4D4D4;"> </span><span style="color:#CE9178;">install</span><span style="color:#D4D4D4;"> </span><span style="color:#CE9178;">python3-opencv</span></span>
<span class="line"></span></code></pre></div><p>在 Termux 上安装时，默认的 <code>pip</code> 安装命令可能失败，使用 <code>pkg</code> 包管理器安装</p><div class="language-bash" data-ext="sh"><pre class="shiki dark-plus" style="background-color:#1E1E1E;" tabindex="0"><code><span class="line"><span style="color:#DCDCAA;">pkg</span><span style="color:#D4D4D4;"> </span><span style="color:#CE9178;">install</span><span style="color:#D4D4D4;"> </span><span style="color:#CE9178;">opencv-python</span></span>
<span class="line"></span></code></pre></div>`,23);function D(C,u){const a=l("router-link");return c(),t("div",null,[i,s("nav",h,[s("ul",null,[s("li",null,[e(a,{to:"#_1-1-opencv-的-python-默认发行版"},{default:p(()=>[n("1.1 OpenCV 的 Python 默认发行版")]),_:1})]),s("li",null,[e(a,{to:"#_1-2-安装扩展包和无-gui-版本的-opencv"},{default:p(()=>[n("1.2 安装扩展包和无 GUI 版本的 OpenCV")]),_:1})]),s("li",null,[e(a,{to:"#_1-3-使用其他包管理器"},{default:p(()=>[n("1.3 使用其他包管理器")]),_:1})])])]),y])}const E=o(d,[["render",D],["__file","index.html.vue"]]);export{E as default};
