import{_ as d,r as s,o,c as n,a as e,b as i,d as t,e as a}from"./app.3ca3e818.js";const k={},h=a('<h1 id="_5-自动光学检查、对象分割和检测" tabindex="-1"><a class="header-anchor" href="#_5-自动光学检查、对象分割和检测" aria-hidden="true">#</a> 5. 自动光学检查、对象分割和检测</h1><p>本章介绍以下主题：</p><ul class="task-list-container"><li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" id="task-item-0" checked="checked" disabled="disabled"><label class="task-list-item-label" for="task-item-0"> 噪声消除</label></li><li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" id="task-item-1" checked="checked" disabled="disabled"><label class="task-list-item-label" for="task-item-1"> 光 / 背景去除知识</label></li><li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" id="task-item-2" checked="checked" disabled="disabled"><label class="task-list-item-label" for="task-item-2"> 阈值</label></li><li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" id="task-item-3" checked="checked" disabled="disabled"><label class="task-list-item-label" for="task-item-3"> 用于对象分割的连通组件</label></li><li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" id="task-item-4" checked="checked" disabled="disabled"><label class="task-list-item-label" for="task-item-4"> 对象分割中的轮廓查找</label></li></ul><p>许多行业都会使用复杂的计算机视觉系统和硬件，计算机视觉技术可以用于检测问题并最大限度地减少生产过程中产生的错误，从而提高最终产品的质量。</p><p>在这个领域中，计算机视觉任务是 <strong>自动光学检查</strong>（AOI）。如今，使用不同的相机的光学检查技术以及复杂的算法正在成千上万的行业中应用，例如用于缺陷检测、分类等。</p><h2 id="_5-1-技术要求" tabindex="-1"><a class="header-anchor" href="#_5-1-技术要求" aria-hidden="true">#</a> 5.1 技术要求</h2>',6),r={class:"task-list-container"},b=e("li",{class:"task-list-item"},[e("input",{type:"checkbox",class:"task-list-item-checkbox",id:"task-item-5",checked:"checked",disabled:"disabled"}),e("label",{class:"task-list-item-label",for:"task-item-5"}," 熟悉 C++ 语言")],-1),m={class:"task-list-item"},p=e("input",{type:"checkbox",class:"task-list-item-checkbox",id:"task-item-6",checked:"checked",disabled:"disabled"},null,-1),_={class:"task-list-item-label",for:"task-item-6"},u={href:"https://github.com/PacktPublishing/Learn-OpenCV-4-By-Building-Projects-Second-Edition/tree/master/Chapter_05",target:"_blank",rel:"noopener noreferrer"},x=e("li",{class:"task-list-item"},[e("input",{type:"checkbox",class:"task-list-item-checkbox",id:"task-item-7",checked:"checked",disabled:"disabled"}),e("label",{class:"task-list-item-label",for:"task-item-7"}," 现代操作系统，例如 Ubuntu 20+ 或者 Windows 10+")],-1),f=e("h2",{id:"_5-2-隔离场景中的对象",tabindex:"-1"},[e("a",{class:"header-anchor",href:"#_5-2-隔离场景中的对象","aria-hidden":"true"},"#"),i(" 5.2 隔离场景中的对象")],-1),g=a(`<h2 id="_5-3-为-aoi-创建应用程序" tabindex="-1"><a class="header-anchor" href="#_5-3-为-aoi-创建应用程序" aria-hidden="true">#</a> 5.3 为 AOI 创建应用程序</h2><p>本节我们创建一个命令行程序，其参数构成如下：</p><ol><li>要被处理的图像</li><li>光图像模式 <ul><li>0，代表减法操作，即求差异</li><li>1，代表除法运算</li></ul></li><li>分割 <ul><li>1，采用用于分割的连通组件方法</li><li>2，采用具有统计区域的连通组件方法</li><li>3，将查找轮廓的方法应用于分割</li></ul></li></ol><p>我们的命令行解析器如下：</p><div class="language-cpp line-numbers-mode" data-ext="cpp"><pre class="language-cpp"><code>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div>`,5);function y(v,B){const l=s("ExternalLinkIcon"),c=s("Mermaid");return o(),n("div",null,[h,e("ul",r,[b,e("li",m,[p,e("label",_,[e("a",u,[i("本章代码"),t(l)])])]),x]),f,t(c,{id:"mermaid-62",code:"eNpLL0osyFAIceFSAAJHjWill4tani5peT6hTSlWU0FX1w4s4QSUeNrR9rRzE7KoM1D0eefOp/san/VPeNo/DVnOBSj3bM6upzNXPF277MnOBUDdzzfuRlbhCjJzQt+zOfPh9nEBAJoTOAo="}),g])}const O=d(k,[["render",y],["__file","index.html.vue"]]);export{O as default};