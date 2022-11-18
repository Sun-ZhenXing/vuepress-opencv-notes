import{_ as o,r as d,o as a,c as n,a as t,b as e,d as s,e as l}from"./app.6a5adde4.js";const c={},i=l('<h2 id="调用摄像头时出现错误" tabindex="-1"><a class="header-anchor" href="#调用摄像头时出现错误" aria-hidden="true">#</a> 调用摄像头时出现错误</h2><h3 id="vsfilter-dll-非法地址访问" tabindex="-1"><a class="header-anchor" href="#vsfilter-dll-非法地址访问" aria-hidden="true">#</a> VSFilter.dll 非法地址访问</h3><table><thead><tr><th>错误调用</th><th>源</th><th>错误类型</th></tr></thead><tbody><tr><td><code>cap &gt;&gt; frame</code></td><td><code>VSFilter.dll</code></td><td>非法地址访问</td></tr></tbody></table><p>Windows 上默认提供的 <code>VSFilter.dll</code> 没有符号表，版本也较低。可能和 OpenCV 4.6.0 已经不兼容，需要安装新的 VSFilter 。</p>',4),h={href:"https://github.com/pinterf/xy-VSFilter/releases",target:"_blank",rel:"noopener noreferrer"},_=t("code",null,"C:\\Windows\\System32",-1),f=t("code",null,"C:\\Windows\\SysWOW64\\",-1);function p(u,m){const r=d("ExternalLinkIcon");return a(),n("div",null,[i,t("p",null,[e("从 "),t("a",h,[e("GitHub：xy-VSFilter/releases"),s(r)]),e(" 中下载最新的一个稳定版本，解压后得到 32/64 位的动态链接库，将 64 位的 DLL 覆盖到 "),_,e("，32 位的 DLL 覆盖到 "),f,e("，重新运行后没有问题。")])])}const b=o(c,[["render",p],["__file","windows-errors.html.vue"]]);export{b as default};
