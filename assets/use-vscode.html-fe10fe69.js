import{_ as s,o as n,c as l,e as a}from"./app-e045e1fd.js";const o={},e=a(`<h1 id="使用-vs-code-开发-c-opencv" tabindex="-1"><a class="header-anchor" href="#使用-vs-code-开发-c-opencv" aria-hidden="true">#</a> 使用 VS Code 开发 C++ OpenCV</h1><p>需要使用 CMake 和 C/C++ 语言扩展，并配置 <code>.vscode/c_cpp_properties.json</code>，样例如下：</p><div class="language-json line-numbers-mode" data-ext="json"><pre class="shiki dark-plus" style="background-color:#1E1E1E;" tabindex="0"><code><span class="line"><span style="color:#D4D4D4;">{</span></span>
<span class="line"><span style="color:#D4D4D4;">    </span><span style="color:#9CDCFE;">&quot;configurations&quot;</span><span style="color:#D4D4D4;">: [</span></span>
<span class="line"><span style="color:#D4D4D4;">        {</span></span>
<span class="line"><span style="color:#D4D4D4;">            </span><span style="color:#9CDCFE;">&quot;name&quot;</span><span style="color:#D4D4D4;">: </span><span style="color:#CE9178;">&quot;Win32&quot;</span><span style="color:#D4D4D4;">,</span></span>
<span class="line"><span style="color:#D4D4D4;">            </span><span style="color:#9CDCFE;">&quot;includePath&quot;</span><span style="color:#D4D4D4;">: [</span></span>
<span class="line"><span style="color:#D4D4D4;">                </span><span style="color:#CE9178;">&quot;\${default}&quot;</span><span style="color:#D4D4D4;">,</span></span>
<span class="line"><span style="color:#D4D4D4;">                </span><span style="color:#CE9178;">&quot;\${workspaceFolder}/**&quot;</span><span style="color:#D4D4D4;">,</span></span>
<span class="line"><span style="color:#D4D4D4;">                </span><span style="color:#CE9178;">&quot;D:/Program/opencv4.7/opencv/build/include&quot;</span></span>
<span class="line"><span style="color:#D4D4D4;">            ],</span></span>
<span class="line"><span style="color:#D4D4D4;">            </span><span style="color:#9CDCFE;">&quot;defines&quot;</span><span style="color:#D4D4D4;">: [</span></span>
<span class="line"><span style="color:#D4D4D4;">                </span><span style="color:#CE9178;">&quot;_DEBUG&quot;</span><span style="color:#D4D4D4;">,</span></span>
<span class="line"><span style="color:#D4D4D4;">                </span><span style="color:#CE9178;">&quot;UNICODE&quot;</span><span style="color:#D4D4D4;">,</span></span>
<span class="line"><span style="color:#D4D4D4;">                </span><span style="color:#CE9178;">&quot;_UNICODE&quot;</span></span>
<span class="line"><span style="color:#D4D4D4;">            ],</span></span>
<span class="line"><span style="color:#D4D4D4;">            </span><span style="color:#9CDCFE;">&quot;windowsSdkVersion&quot;</span><span style="color:#D4D4D4;">: </span><span style="color:#CE9178;">&quot;10.0.19044.0&quot;</span><span style="color:#D4D4D4;">,</span></span>
<span class="line"><span style="color:#D4D4D4;">            </span><span style="color:#9CDCFE;">&quot;cStandard&quot;</span><span style="color:#D4D4D4;">: </span><span style="color:#CE9178;">&quot;c17&quot;</span><span style="color:#D4D4D4;">,</span></span>
<span class="line"><span style="color:#D4D4D4;">            </span><span style="color:#9CDCFE;">&quot;cppStandard&quot;</span><span style="color:#D4D4D4;">: </span><span style="color:#CE9178;">&quot;c++17&quot;</span><span style="color:#D4D4D4;">,</span></span>
<span class="line"><span style="color:#D4D4D4;">            </span><span style="color:#9CDCFE;">&quot;intelliSenseMode&quot;</span><span style="color:#D4D4D4;">: </span><span style="color:#CE9178;">&quot;windows-msvc-x64&quot;</span><span style="color:#D4D4D4;">,</span></span>
<span class="line"><span style="color:#D4D4D4;">            </span><span style="color:#9CDCFE;">&quot;configurationProvider&quot;</span><span style="color:#D4D4D4;">: </span><span style="color:#CE9178;">&quot;ms-vscode.cmake-tools&quot;</span><span style="color:#D4D4D4;">,</span></span>
<span class="line"><span style="color:#D4D4D4;">            </span><span style="color:#9CDCFE;">&quot;compilerPath&quot;</span><span style="color:#D4D4D4;">: </span><span style="color:#CE9178;">&quot;C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.35.32215/bin/Hostx64/x64/cl.exe&quot;</span></span>
<span class="line"><span style="color:#D4D4D4;">        }</span></span>
<span class="line"><span style="color:#D4D4D4;">    ],</span></span>
<span class="line"><span style="color:#D4D4D4;">    </span><span style="color:#9CDCFE;">&quot;version&quot;</span><span style="color:#D4D4D4;">: </span><span style="color:#B5CEA8;">4</span></span>
<span class="line"><span style="color:#D4D4D4;">}</span></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>这份配置依赖于已经安装的 Visual Studio，其中需要更改的配置：</p><ul><li><code>includePath</code> 需要改为本机的 OpenCV Include 路径，具体取决于安装时的路径</li><li><code>windowsSdkVersion</code> 需要改为本机已经安装的 Windows SDK</li><li><code>compilerPath</code> 修改为本机安装的 MSVC 编译器路径</li></ul><p>其他常见更改：</p><ul><li>C/C++ 版本</li><li>在 <code>env</code> 中定义用户定义变量</li><li>定义标志 <code>defines</code></li></ul>`,7),p=[e];function c(i,t){return n(),l("div",null,p)}const r=s(o,[["render",c],["__file","use-vscode.html.vue"]]);export{r as default};