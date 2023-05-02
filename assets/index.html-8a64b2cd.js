import{_ as n,p as s,q as a,a1 as t}from"./framework-a77b8537.js";const p={},e=t(`<h1 id="python-opencv-目标分割" tabindex="-1"><a class="header-anchor" href="#python-opencv-目标分割" aria-hidden="true">#</a> Python OpenCV 目标分割</h1><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token keyword">import</span> cv2
<span class="token keyword">import</span> numpy <span class="token keyword">as</span> np


<span class="token keyword">def</span> <span class="token function">cluster</span><span class="token punctuation">(</span>img<span class="token punctuation">:</span> np<span class="token punctuation">.</span>ndarray<span class="token punctuation">,</span> k<span class="token punctuation">:</span> <span class="token builtin">int</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token triple-quoted-string string">&quot;&quot;&quot;聚类实现目标分割
    @param \`img\`: 图像
    @param \`k\`: 聚类数
    &quot;&quot;&quot;</span>
    <span class="token comment"># 图像尺寸</span>
    h<span class="token punctuation">,</span> w <span class="token operator">=</span> img<span class="token punctuation">.</span>shape<span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token number">2</span><span class="token punctuation">]</span>
    <span class="token comment"># 将图像转换为二维矩阵</span>
    data <span class="token operator">=</span> img<span class="token punctuation">.</span>reshape<span class="token punctuation">(</span><span class="token punctuation">(</span>h <span class="token operator">*</span> w<span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    <span class="token comment"># 转换为浮点型</span>
    data <span class="token operator">=</span> np<span class="token punctuation">.</span>float32<span class="token punctuation">(</span>data<span class="token punctuation">)</span>
    <span class="token comment"># 定义停止条件</span>
    criteria <span class="token operator">=</span> cv2<span class="token punctuation">.</span>TERM_CRITERIA_EPS <span class="token operator">+</span> cv2<span class="token punctuation">.</span>TERM_CRITERIA_MAX_ITER<span class="token punctuation">,</span> <span class="token number">10</span><span class="token punctuation">,</span> <span class="token number">1.0</span>
    <span class="token comment"># 聚类</span>
    ret<span class="token punctuation">,</span> label<span class="token punctuation">,</span> center <span class="token operator">=</span> cv2<span class="token punctuation">.</span>kmeans<span class="token punctuation">(</span>
        data<span class="token punctuation">,</span> k<span class="token punctuation">,</span> <span class="token boolean">None</span><span class="token punctuation">,</span> criteria<span class="token punctuation">,</span> <span class="token number">10</span><span class="token punctuation">,</span> cv2<span class="token punctuation">.</span>KMEANS_RANDOM_CENTERS
    <span class="token punctuation">)</span>
    <span class="token comment"># 转换回uint8</span>
    center <span class="token operator">=</span> np<span class="token punctuation">.</span>uint8<span class="token punctuation">(</span>center<span class="token punctuation">)</span>
    <span class="token comment"># 分割图像</span>
    res <span class="token operator">=</span> center<span class="token punctuation">[</span>label<span class="token punctuation">.</span>flatten<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">]</span>
    <span class="token keyword">return</span> res<span class="token punctuation">.</span>reshape<span class="token punctuation">(</span><span class="token punctuation">(</span>img<span class="token punctuation">.</span>shape<span class="token punctuation">)</span><span class="token punctuation">)</span>


<span class="token keyword">def</span> <span class="token function">threshold</span><span class="token punctuation">(</span>img<span class="token punctuation">:</span> np<span class="token punctuation">.</span>ndarray<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token triple-quoted-string string">&quot;&quot;&quot;自适应阈值分割实现目标分割
    @param \`img\`: 图像
    &quot;&quot;&quot;</span>
    <span class="token comment"># 图像灰度化</span>
    gray <span class="token operator">=</span> cv2<span class="token punctuation">.</span>cvtColor<span class="token punctuation">(</span>img<span class="token punctuation">,</span> cv2<span class="token punctuation">.</span>COLOR_BGR2GRAY<span class="token punctuation">)</span>
    <span class="token comment"># 阈值分割</span>
    ret<span class="token punctuation">,</span> binary <span class="token operator">=</span> cv2<span class="token punctuation">.</span>threshold<span class="token punctuation">(</span>
        gray<span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">,</span> <span class="token number">255</span><span class="token punctuation">,</span> cv2<span class="token punctuation">.</span>THRESH_BINARY <span class="token operator">|</span> cv2<span class="token punctuation">.</span>THRESH_OTSU
    <span class="token punctuation">)</span>
    <span class="token keyword">return</span> binary


<span class="token keyword">def</span> <span class="token function">main</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
    img <span class="token operator">=</span> cv2<span class="token punctuation">.</span>imread<span class="token punctuation">(</span><span class="token string">&#39;test.png&#39;</span><span class="token punctuation">)</span>
    cv2<span class="token punctuation">.</span>imshow<span class="token punctuation">(</span><span class="token string">&#39;img&#39;</span><span class="token punctuation">,</span> img<span class="token punctuation">)</span>
    <span class="token comment"># 聚类分割</span>
    res_cluster <span class="token operator">=</span> cluster<span class="token punctuation">(</span>img<span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">)</span>
    <span class="token comment"># 显示图像</span>
    cv2<span class="token punctuation">.</span>imshow<span class="token punctuation">(</span><span class="token string">&#39;res&#39;</span><span class="token punctuation">,</span> res_cluster<span class="token punctuation">)</span>
    <span class="token comment"># 阈值分割</span>
    res_threshold <span class="token operator">=</span> threshold<span class="token punctuation">(</span>img<span class="token punctuation">)</span>
    cv2<span class="token punctuation">.</span>imshow<span class="token punctuation">(</span><span class="token string">&#39;res2&#39;</span><span class="token punctuation">,</span> res_threshold<span class="token punctuation">)</span>
    <span class="token comment"># 等待显示</span>
    cv2<span class="token punctuation">.</span>waitKey<span class="token punctuation">(</span><span class="token punctuation">)</span>
    cv2<span class="token punctuation">.</span>destroyAllWindows<span class="token punctuation">(</span><span class="token punctuation">)</span>


<span class="token keyword">if</span> __name__ <span class="token operator">==</span> <span class="token string">&#39;__main__&#39;</span><span class="token punctuation">:</span>
    main<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>`,2),c=[e];function o(i,l){return s(),a("div",null,c)}const r=n(p,[["render",o],["__file","index.html.vue"]]);export{r as default};
