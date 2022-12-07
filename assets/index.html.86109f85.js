import{_ as n,o as a,c as s,e}from"./app.884e9959.js";const t={},c=e(`<h1 id="使用技巧总结" tabindex="-1"><a class="header-anchor" href="#使用技巧总结" aria-hidden="true">#</a> 使用技巧总结</h1><h2 id="_1-内存引用" tabindex="-1"><a class="header-anchor" href="#_1-内存引用" aria-hidden="true">#</a> 1. 内存引用</h2><h3 id="_1-1-内存释放" tabindex="-1"><a class="header-anchor" href="#_1-1-内存释放" aria-hidden="true">#</a> 1.1 内存释放</h3><p>图像是及其占用内存的操作，所以需要在必要的时候释放：</p><div class="language-cpp" data-ext="cpp"><pre class="language-cpp"><code>Mat m<span class="token punctuation">;</span>
<span class="token comment">// ...</span>
<span class="token comment">// 释放空间</span>
m<span class="token punctuation">.</span><span class="token function">release</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
</code></pre></div><p>如果使用容器来储存图像，清除容器则意味着容器存储的对象也会被释放：</p><div class="language-cpp" data-ext="cpp"><pre class="language-cpp"><code>std<span class="token double-colon punctuation">::</span>vector<span class="token operator">&lt;</span>cv<span class="token double-colon punctuation">::</span>Mat<span class="token operator">&gt;</span> my_vector<span class="token punctuation">;</span>
<span class="token comment">// ...</span>
<span class="token comment">// 释放空间</span>
my_vector<span class="token punctuation">.</span><span class="token function">clear</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
</code></pre></div><p>注意，当还有图像的引用未被释放时，图像所申请的空间也不会被释放。</p>`,8),p=[c];function o(l,i){return a(),s("div",null,p)}const u=n(t,[["render",o],["__file","index.html.vue"]]);export{u as default};
