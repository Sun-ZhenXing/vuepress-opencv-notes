import{_ as s,o as n,c as a,e as l}from"./app-e9203006.js";const e={},p=l(`<h1 id="python-opencv-目标分割" tabindex="-1"><a class="header-anchor" href="#python-opencv-目标分割" aria-hidden="true">#</a> Python OpenCV 目标分割</h1><div class="language-python line-numbers-mode" data-ext="py"><pre class="shiki dark-plus" style="background-color:#1E1E1E;" tabindex="0"><code><span class="line"><span style="color:#C586C0;">import</span><span style="color:#D4D4D4;"> cv2</span></span>
<span class="line"><span style="color:#C586C0;">import</span><span style="color:#D4D4D4;"> numpy </span><span style="color:#C586C0;">as</span><span style="color:#D4D4D4;"> np</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="color:#569CD6;">def</span><span style="color:#D4D4D4;"> </span><span style="color:#DCDCAA;">cluster</span><span style="color:#D4D4D4;">(</span><span style="color:#9CDCFE;">img</span><span style="color:#D4D4D4;">: np.ndarray, </span><span style="color:#9CDCFE;">k</span><span style="color:#D4D4D4;">: </span><span style="color:#4EC9B0;">int</span><span style="color:#D4D4D4;">):</span></span>
<span class="line"><span style="color:#D4D4D4;">    </span><span style="color:#CE9178;">&quot;&quot;&quot;聚类实现目标分割</span></span>
<span class="line"><span style="color:#CE9178;">    @param \`img\`: 图像</span></span>
<span class="line"><span style="color:#CE9178;">    @param \`k\`: 聚类数</span></span>
<span class="line"><span style="color:#CE9178;">    &quot;&quot;&quot;</span></span>
<span class="line"><span style="color:#D4D4D4;">    </span><span style="color:#6A9955;"># 图像尺寸</span></span>
<span class="line"><span style="color:#D4D4D4;">    h, w = img.shape[:</span><span style="color:#B5CEA8;">2</span><span style="color:#D4D4D4;">]</span></span>
<span class="line"><span style="color:#D4D4D4;">    </span><span style="color:#6A9955;"># 将图像转换为二维矩阵</span></span>
<span class="line"><span style="color:#D4D4D4;">    data = img.reshape((h * w, </span><span style="color:#B5CEA8;">3</span><span style="color:#D4D4D4;">))</span></span>
<span class="line"><span style="color:#D4D4D4;">    </span><span style="color:#6A9955;"># 转换为浮点型</span></span>
<span class="line"><span style="color:#D4D4D4;">    data = np.float32(data)</span></span>
<span class="line"><span style="color:#D4D4D4;">    </span><span style="color:#6A9955;"># 定义停止条件</span></span>
<span class="line"><span style="color:#D4D4D4;">    criteria = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, </span><span style="color:#B5CEA8;">10</span><span style="color:#D4D4D4;">, </span><span style="color:#B5CEA8;">1.0</span></span>
<span class="line"><span style="color:#D4D4D4;">    </span><span style="color:#6A9955;"># 聚类</span></span>
<span class="line"><span style="color:#D4D4D4;">    ret, label, center = cv2.kmeans(</span></span>
<span class="line"><span style="color:#D4D4D4;">        data, k, </span><span style="color:#569CD6;">None</span><span style="color:#D4D4D4;">, criteria, </span><span style="color:#B5CEA8;">10</span><span style="color:#D4D4D4;">, cv2.KMEANS_RANDOM_CENTERS</span></span>
<span class="line"><span style="color:#D4D4D4;">    )</span></span>
<span class="line"><span style="color:#D4D4D4;">    </span><span style="color:#6A9955;"># 转换回uint8</span></span>
<span class="line"><span style="color:#D4D4D4;">    center = np.uint8(center)</span></span>
<span class="line"><span style="color:#D4D4D4;">    </span><span style="color:#6A9955;"># 分割图像</span></span>
<span class="line"><span style="color:#D4D4D4;">    res = center[label.flatten()]</span></span>
<span class="line"><span style="color:#D4D4D4;">    </span><span style="color:#C586C0;">return</span><span style="color:#D4D4D4;"> res.reshape((img.shape))</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="color:#569CD6;">def</span><span style="color:#D4D4D4;"> </span><span style="color:#DCDCAA;">threshold</span><span style="color:#D4D4D4;">(</span><span style="color:#9CDCFE;">img</span><span style="color:#D4D4D4;">: np.ndarray):</span></span>
<span class="line"><span style="color:#D4D4D4;">    </span><span style="color:#CE9178;">&quot;&quot;&quot;自适应阈值分割实现目标分割</span></span>
<span class="line"><span style="color:#CE9178;">    @param \`img\`: 图像</span></span>
<span class="line"><span style="color:#CE9178;">    &quot;&quot;&quot;</span></span>
<span class="line"><span style="color:#D4D4D4;">    </span><span style="color:#6A9955;"># 图像灰度化</span></span>
<span class="line"><span style="color:#D4D4D4;">    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)</span></span>
<span class="line"><span style="color:#D4D4D4;">    </span><span style="color:#6A9955;"># 阈值分割</span></span>
<span class="line"><span style="color:#D4D4D4;">    ret, binary = cv2.threshold(gray, </span><span style="color:#B5CEA8;">0</span><span style="color:#D4D4D4;">, </span><span style="color:#B5CEA8;">255</span><span style="color:#D4D4D4;">, cv2.THRESH_BINARY | cv2.THRESH_OTSU)</span></span>
<span class="line"><span style="color:#D4D4D4;">    </span><span style="color:#C586C0;">return</span><span style="color:#D4D4D4;"> binary</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="color:#569CD6;">def</span><span style="color:#D4D4D4;"> </span><span style="color:#DCDCAA;">main</span><span style="color:#D4D4D4;">():</span></span>
<span class="line"><span style="color:#D4D4D4;">    img = cv2.imread(</span><span style="color:#CE9178;">&quot;test.png&quot;</span><span style="color:#D4D4D4;">)</span></span>
<span class="line"><span style="color:#D4D4D4;">    cv2.imshow(</span><span style="color:#CE9178;">&quot;img&quot;</span><span style="color:#D4D4D4;">, img)</span></span>
<span class="line"><span style="color:#D4D4D4;">    </span><span style="color:#6A9955;"># 聚类分割</span></span>
<span class="line"><span style="color:#D4D4D4;">    res_cluster = cluster(img, </span><span style="color:#B5CEA8;">3</span><span style="color:#D4D4D4;">)</span></span>
<span class="line"><span style="color:#D4D4D4;">    </span><span style="color:#6A9955;"># 显示图像</span></span>
<span class="line"><span style="color:#D4D4D4;">    cv2.imshow(</span><span style="color:#CE9178;">&quot;res&quot;</span><span style="color:#D4D4D4;">, res_cluster)</span></span>
<span class="line"><span style="color:#D4D4D4;">    </span><span style="color:#6A9955;"># 阈值分割</span></span>
<span class="line"><span style="color:#D4D4D4;">    res_threshold = threshold(img)</span></span>
<span class="line"><span style="color:#D4D4D4;">    cv2.imshow(</span><span style="color:#CE9178;">&quot;res2&quot;</span><span style="color:#D4D4D4;">, res_threshold)</span></span>
<span class="line"><span style="color:#D4D4D4;">    </span><span style="color:#6A9955;"># 等待显示</span></span>
<span class="line"><span style="color:#D4D4D4;">    cv2.waitKey()</span></span>
<span class="line"><span style="color:#D4D4D4;">    cv2.destroyAllWindows()</span></span>
<span class="line"></span>
<span class="line"></span>
<span class="line"><span style="color:#C586C0;">if</span><span style="color:#D4D4D4;"> </span><span style="color:#9CDCFE;">__name__</span><span style="color:#D4D4D4;"> == </span><span style="color:#CE9178;">&quot;__main__&quot;</span><span style="color:#D4D4D4;">:</span></span>
<span class="line"><span style="color:#D4D4D4;">    main()</span></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>`,2),o=[p];function c(i,r){return n(),a("div",null,o)}const t=s(e,[["render",c],["__file","index.html.vue"]]);export{t as default};
