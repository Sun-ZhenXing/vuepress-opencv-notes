import{_ as p,r as o,o as c,c as r,b as s,a as l,w as e,d as n,e as i}from"./app-e045e1fd.js";const t={},D=s("h1",{id:"python-opencv-支持-unicode",tabindex:"-1"},[s("a",{class:"header-anchor",href:"#python-opencv-支持-unicode","aria-hidden":"true"},"#"),n(" Python OpenCV 支持 Unicode")],-1),d={class:"table-of-contents"},y=i(`<p>如果在 <code>cv2</code> 中使用 Unicode 字符串，在某些系统上会无法读取。</p><p>C++ OpenCV 可以通过对应平台的编码来解决这个问题，但是 Python OpenCV 没有这个功能。</p><h2 id="_1-使用-numpy-读取" tabindex="-1"><a class="header-anchor" href="#_1-使用-numpy-读取" aria-hidden="true">#</a> 1. 使用 NumPy 读取</h2><div class="language-python line-numbers-mode" data-ext="py"><pre class="shiki dark-plus" style="background-color:#1E1E1E;" tabindex="0"><code><span class="line"><span style="color:#C586C0;">import</span><span style="color:#D4D4D4;"> cv2</span></span>
<span class="line"><span style="color:#C586C0;">import</span><span style="color:#D4D4D4;"> numpy </span><span style="color:#C586C0;">as</span><span style="color:#D4D4D4;"> np</span></span>
<span class="line"><span style="color:#D4D4D4;"> </span></span>
<span class="line"><span style="color:#569CD6;">def</span><span style="color:#D4D4D4;"> </span><span style="color:#DCDCAA;">cv_imread</span><span style="color:#D4D4D4;">(</span><span style="color:#9CDCFE;">path</span><span style="color:#D4D4D4;">: </span><span style="color:#4EC9B0;">str</span><span style="color:#D4D4D4;">, </span><span style="color:#9CDCFE;">flags</span><span style="color:#D4D4D4;">=cv2.IMREAD_COLOR):</span></span>
<span class="line"><span style="color:#D4D4D4;">    cv_img = cv2.imdecode(np.fromfile(path, </span><span style="color:#9CDCFE;">dtype</span><span style="color:#D4D4D4;">=np.uint8), flags)</span></span>
<span class="line"><span style="color:#D4D4D4;">    </span><span style="color:#C586C0;">return</span><span style="color:#D4D4D4;"> cv_img</span></span>
<span class="line"></span>
<span class="line"><span style="color:#569CD6;">def</span><span style="color:#D4D4D4;"> </span><span style="color:#DCDCAA;">cv_imwrite</span><span style="color:#D4D4D4;">(</span><span style="color:#9CDCFE;">path</span><span style="color:#D4D4D4;">: </span><span style="color:#4EC9B0;">str</span><span style="color:#D4D4D4;">, </span><span style="color:#9CDCFE;">img</span><span style="color:#D4D4D4;">: np.ndarray):</span></span>
<span class="line"><span style="color:#D4D4D4;">    buffer: np.ndarray = cv2.imencode(</span><span style="color:#CE9178;">&#39;.jpg&#39;</span><span style="color:#D4D4D4;">, img)[</span><span style="color:#B5CEA8;">1</span><span style="color:#D4D4D4;">]</span></span>
<span class="line"><span style="color:#D4D4D4;">    buffer.tofile(path)</span></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h2 id="_2-使用-pil-读取" tabindex="-1"><a class="header-anchor" href="#_2-使用-pil-读取" aria-hidden="true">#</a> 2. 使用 PIL 读取</h2><div class="language-python line-numbers-mode" data-ext="py"><pre class="shiki dark-plus" style="background-color:#1E1E1E;" tabindex="0"><code><span class="line"><span style="color:#C586C0;">import</span><span style="color:#D4D4D4;"> cv2</span></span>
<span class="line"><span style="color:#C586C0;">from</span><span style="color:#D4D4D4;"> PIL </span><span style="color:#C586C0;">import</span><span style="color:#D4D4D4;"> Image</span></span>
<span class="line"></span>
<span class="line"><span style="color:#569CD6;">def</span><span style="color:#D4D4D4;"> </span><span style="color:#DCDCAA;">pli_to_cv2</span><span style="color:#D4D4D4;">(</span><span style="color:#9CDCFE;">img</span><span style="color:#D4D4D4;">: Image.Image):</span></span>
<span class="line"><span style="color:#D4D4D4;">    </span><span style="color:#C586C0;">return</span><span style="color:#D4D4D4;"> cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)</span></span>
<span class="line"></span>
<span class="line"><span style="color:#569CD6;">def</span><span style="color:#D4D4D4;"> </span><span style="color:#DCDCAA;">cv2_to_pli</span><span style="color:#D4D4D4;">(</span><span style="color:#9CDCFE;">img</span><span style="color:#D4D4D4;">: np.ndarray):</span></span>
<span class="line"><span style="color:#D4D4D4;">    </span><span style="color:#C586C0;">return</span><span style="color:#D4D4D4;"> Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))</span></span>
<span class="line"></span>
<span class="line"><span style="color:#569CD6;">def</span><span style="color:#D4D4D4;"> </span><span style="color:#DCDCAA;">cv_imread</span><span style="color:#D4D4D4;">(</span><span style="color:#9CDCFE;">path</span><span style="color:#D4D4D4;">: </span><span style="color:#4EC9B0;">str</span><span style="color:#D4D4D4;">, **</span><span style="color:#9CDCFE;">kwargs</span><span style="color:#D4D4D4;">):</span></span>
<span class="line"><span style="color:#D4D4D4;">    </span><span style="color:#C586C0;">return</span><span style="color:#D4D4D4;"> pli_to_cv2(Image.open(path, **kwargs))</span></span>
<span class="line"></span>
<span class="line"><span style="color:#569CD6;">def</span><span style="color:#D4D4D4;"> </span><span style="color:#DCDCAA;">cv_imwrite</span><span style="color:#D4D4D4;">(</span><span style="color:#9CDCFE;">path</span><span style="color:#D4D4D4;">: </span><span style="color:#4EC9B0;">str</span><span style="color:#D4D4D4;">, </span><span style="color:#9CDCFE;">img</span><span style="color:#D4D4D4;">: np.ndarray):</span></span>
<span class="line"><span style="color:#D4D4D4;">    cv2_to_pli(img).save(path)</span></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>`,6);function v(C,m){const a=o("router-link");return c(),r("div",null,[D,s("nav",d,[s("ul",null,[s("li",null,[l(a,{to:"#_1-使用-numpy-读取"},{default:e(()=>[n("1. 使用 NumPy 读取")]),_:1})]),s("li",null,[l(a,{to:"#_2-使用-pil-读取"},{default:e(()=>[n("2. 使用 PIL 读取")]),_:1})])])]),y])}const _=p(t,[["render",v],["__file","cv2-support-unicode.html.vue"]]);export{_ as default};
