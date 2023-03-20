import{_ as n,p as s,q as a,a1 as o}from"./framework-546207d5.js";const t={},e=o(`<h1 id="使用-vs-code-开发-c-opencv" tabindex="-1"><a class="header-anchor" href="#使用-vs-code-开发-c-opencv" aria-hidden="true">#</a> 使用 VS Code 开发 C++ OpenCV</h1><p>需要使用 CMake 和 C/C++ 语言扩展，并配置 <code>.vscode/c_cpp_properties.json</code>，样例如下：</p><div class="language-json line-numbers-mode" data-ext="json"><pre class="language-json"><code><span class="token punctuation">{</span>
    <span class="token property">&quot;configurations&quot;</span><span class="token operator">:</span> <span class="token punctuation">[</span>
        <span class="token punctuation">{</span>
            <span class="token property">&quot;name&quot;</span><span class="token operator">:</span> <span class="token string">&quot;Win32&quot;</span><span class="token punctuation">,</span>
            <span class="token property">&quot;includePath&quot;</span><span class="token operator">:</span> <span class="token punctuation">[</span>
                <span class="token string">&quot;\${default}&quot;</span><span class="token punctuation">,</span>
                <span class="token string">&quot;\${workspaceFolder}/**&quot;</span><span class="token punctuation">,</span>
                <span class="token string">&quot;D:/Program/opencv4.7/opencv/build/include&quot;</span>
            <span class="token punctuation">]</span><span class="token punctuation">,</span>
            <span class="token property">&quot;defines&quot;</span><span class="token operator">:</span> <span class="token punctuation">[</span>
                <span class="token string">&quot;_DEBUG&quot;</span><span class="token punctuation">,</span>
                <span class="token string">&quot;UNICODE&quot;</span><span class="token punctuation">,</span>
                <span class="token string">&quot;_UNICODE&quot;</span>
            <span class="token punctuation">]</span><span class="token punctuation">,</span>
            <span class="token property">&quot;windowsSdkVersion&quot;</span><span class="token operator">:</span> <span class="token string">&quot;10.0.19044.0&quot;</span><span class="token punctuation">,</span>
            <span class="token property">&quot;cStandard&quot;</span><span class="token operator">:</span> <span class="token string">&quot;c17&quot;</span><span class="token punctuation">,</span>
            <span class="token property">&quot;cppStandard&quot;</span><span class="token operator">:</span> <span class="token string">&quot;c++17&quot;</span><span class="token punctuation">,</span>
            <span class="token property">&quot;intelliSenseMode&quot;</span><span class="token operator">:</span> <span class="token string">&quot;windows-msvc-x64&quot;</span><span class="token punctuation">,</span>
            <span class="token property">&quot;configurationProvider&quot;</span><span class="token operator">:</span> <span class="token string">&quot;ms-vscode.cmake-tools&quot;</span><span class="token punctuation">,</span>
            <span class="token property">&quot;compilerPath&quot;</span><span class="token operator">:</span> <span class="token string">&quot;C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.35.32215/bin/Hostx64/x64/cl.exe&quot;</span>
        <span class="token punctuation">}</span>
    <span class="token punctuation">]</span><span class="token punctuation">,</span>
    <span class="token property">&quot;version&quot;</span><span class="token operator">:</span> <span class="token number">4</span>
<span class="token punctuation">}</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>这份配置依赖于已经安装的 Visual Studio，其中需要更改的配置：</p><ul><li><code>includePath</code> 需要改为本机的 OpenCV Include 路径，具体取决于安装时的路径</li><li><code>windowsSdkVersion</code> 需要改为本机已经安装的 Windows SDK</li><li><code>compilerPath</code> 修改为本机安装的 MSVC 编译器路径</li></ul><p>其他常见更改：</p><ul><li>C/C++ 版本</li><li>在 <code>env</code> 中定义用户定义变量</li><li>定义标志 <code>defines</code></li></ul>`,7),p=[e];function c(i,l){return s(),a("div",null,p)}const r=n(t,[["render",c],["__file","use-vscode.html.vue"]]);export{r as default};
