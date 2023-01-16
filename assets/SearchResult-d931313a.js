import{u as te,a as D,k as le,b as oe,P as b,B as ne,c as ae}from"./app-59ab8937.js";import{r as L,h as j,c as se,u as re,L as ie,aa as ue,o as ce,n as de,j as o,ab as q,z as ve,K as he,i as pe}from"./framework-d3922052.js";function fe(t){if(Array.isArray(t)){for(var e=0,l=Array(t.length);e<t.length;e++)l[e]=t[e];return l}else return Array.from(t)}var A=!1;if(typeof window<"u"){var G={get passive(){A=!0}};window.addEventListener("testPassive",null,G),window.removeEventListener("testPassive",null,G)}var J=typeof window<"u"&&window.navigator&&window.navigator.platform&&(/iP(ad|hone|od)/.test(window.navigator.platform)||window.navigator.platform==="MacIntel"&&window.navigator.maxTouchPoints>1),w=[],x=!1,B=-1,C=void 0,H=void 0,N=function(e){return w.some(function(l){return!!(l.options.allowTouchMove&&l.options.allowTouchMove(e))})},E=function(e){var l=e||window.event;return N(l.target)||l.touches.length>1?!0:(l.preventDefault&&l.preventDefault(),!1)},ye=function(e){if(H===void 0){var l=!!e&&e.reserveScrollBarGap===!0,n=window.innerWidth-document.documentElement.clientWidth;l&&n>0&&(H=document.body.style.paddingRight,document.body.style.paddingRight=n+"px")}C===void 0&&(C=document.body.style.overflow,document.body.style.overflow="hidden")},ge=function(){H!==void 0&&(document.body.style.paddingRight=H,H=void 0),C!==void 0&&(document.body.style.overflow=C,C=void 0)},me=function(e){return e?e.scrollHeight-e.scrollTop<=e.clientHeight:!1},we=function(e,l){var n=e.targetTouches[0].clientY-B;return N(e.target)?!1:l&&l.scrollTop===0&&n>0||me(l)&&n<0?E(e):(e.stopPropagation(),!0)},Le=function(e,l){if(!e){console.error("disableBodyScroll unsuccessful - targetElement must be provided when calling disableBodyScroll on IOS devices.");return}if(!w.some(function(a){return a.targetElement===e})){var n={targetElement:e,options:l||{}};w=[].concat(fe(w),[n]),J?(e.ontouchstart=function(a){a.targetTouches.length===1&&(B=a.targetTouches[0].clientY)},e.ontouchmove=function(a){a.targetTouches.length===1&&we(a,e)},x||(document.addEventListener("touchmove",E,A?{passive:!1}:void 0),x=!0)):ye(l)}},Ce=function(){J?(w.forEach(function(e){e.targetElement.ontouchstart=null,e.targetElement.ontouchmove=null}),x&&(document.removeEventListener("touchmove",E,A?{passive:!1}:void 0),x=!1),B=-1):ge(),w=[]};const He="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";const U=()=>o(b,{name:"close"},()=>o("path",{d:"m925.468 822.294-303.27-310.288L925.51 201.674c34.683-27.842 38.3-75.802 8.122-107.217-30.135-31.37-82.733-34.259-117.408-6.463L512.001 399.257 207.777 87.993C173.1 60.197 120.504 63.087 90.369 94.456c-30.179 31.415-26.561 79.376 8.122 107.217L401.8 512.005l-303.27 310.29c-34.724 27.82-38.34 75.846-8.117 107.194 30.135 31.437 82.729 34.327 117.408 6.486L512 624.756l304.177 311.22c34.68 27.84 87.272 24.95 117.408-6.487 30.223-31.348 26.56-79.375-8.118-107.195z"}));U.displayName="CloseIcon";const F=()=>o(b,{name:"heading"},()=>o("path",{d:"M250.4 704.6H64V595.4h202.4l26.2-166.6H94V319.6h214.4L352 64h127.8l-43.6 255.4h211.2L691 64h126.2l-43.6 255.4H960v109.2H756.2l-24.6 166.6H930v109.2H717L672 960H545.8l43.6-255.4H376.6L333 960H206.8l43.6-255.4zm168.4-276L394 595.4h211.2l24.6-166.6h-211z"}));F.displayName="HeadingIcon";const K=()=>o(b,{name:"heart"},()=>o("path",{d:"M1024 358.156C1024 195.698 892.3 64 729.844 64c-86.362 0-164.03 37.218-217.844 96.49C458.186 101.218 380.518 64 294.156 64 131.698 64 0 195.698 0 358.156 0 444.518 37.218 522.186 96.49 576H96l320 320c32 32 64 64 96 64s64-32 96-64l320-320h-.49c59.272-53.814 96.49-131.482 96.49-217.844zM841.468 481.232 517.49 805.49a2981.962 2981.962 0 0 1-5.49 5.48c-1.96-1.95-3.814-3.802-5.49-5.48L182.532 481.234C147.366 449.306 128 405.596 128 358.156 128 266.538 202.538 192 294.156 192c47.44 0 91.15 19.366 123.076 54.532L512 350.912l94.768-104.378C638.696 211.366 682.404 192 729.844 192 821.462 192 896 266.538 896 358.156c0 47.44-19.368 91.15-54.532 123.076z"}));K.displayName="HeartIcon";const T=()=>o(b,{name:"history"},()=>o("path",{d:"M512 1024a512 512 0 1 1 512-512 512 512 0 0 1-512 512zm0-896a384 384 0 1 0 384 384 384 384 0 0 0-384-384zm192 448H512a64 64 0 0 1-64-64V320a64 64 0 0 1 128 0v128h128a64 64 0 0 1 0 128z"}));T.displayName="HistoryIcon";const X=()=>o(b,{name:"title"},()=>o("path",{d:"M512 256c70.656 0 134.656 28.672 180.992 75.008A254.933 254.933 0 0 1 768 512c0 83.968-41.024 157.888-103.488 204.48C688.96 748.736 704 788.48 704 832c0 105.984-86.016 192-192 192-106.048 0-192-86.016-192-192h128a64 64 0 1 0 128 0 64 64 0 0 0-64-64 255.19 255.19 0 0 1-181.056-75.008A255.403 255.403 0 0 1 256 512c0-83.968 41.024-157.824 103.488-204.544C335.04 275.264 320 235.584 320 192A192 192 0 0 1 512 0c105.984 0 192 85.952 192 192H576a64.021 64.021 0 0 0-128 0c0 35.328 28.672 64 64 64zM384 512c0 70.656 57.344 128 128 128s128-57.344 128-128-57.344-128-128-128-128 57.344-128 128z"}));X.displayName="TitleIcon";const be={},Se=300,V=5,Ie={"/":{cancel:"取消",placeholder:"搜索",search:"搜索",select:"选择",navigate:"切换",exit:"关闭",history:"搜索历史",emptyHistory:"无搜索历史",emptyResult:"没有找到结果",loading:"正在加载搜索索引..."}},je="search-pro-history-results",g=te(je,[]),xe=()=>({history:g,addHistory:t=>{g.value.length<V?g.value=[t,...g.value]:g.value=[t,...g.value.slice(0,V-1)]},removeHistory:t=>{g.value=[...g.value.slice(0,t),...g.value.slice(t+1)]}}),Ee=L(He),Oe=j(()=>JSON.parse(ne(Ee.value))),I=(t,e)=>{const l=t.toLowerCase(),n=e.toLowerCase(),a=[];let r=0,v=0;const h=(s,p=!1)=>{let i="";v===0?i=s.length>20?`… ${s.slice(-20)}`:s:p?i=s.length+v>100?`${s.slice(0,100-v)}… `:s:i=s.length>20?`${s.slice(0,20)} … ${s.slice(-20)}`:s,i&&a.push(i),v+=i.length,p||(a.push(["strong",e]),v+=e.length,v>=100&&a.push(" …"))};let f=l.indexOf(n,r);if(f===-1)return null;for(;f>=0;){const s=f+n.length;if(h(t.slice(r,f)),r=s,v>100)break;f=l.indexOf(n,r)}return v<100&&h(t.slice(r),!0),a},Y=t=>t.reduce((e,{type:l})=>e+(l==="title"?50:l==="heading"?20:l==="custom"?10:1),0),ke=(t,e)=>{var l;const n={};for(const[a,r]of Object.entries(e)){const v=((l=e[a.replace(/\/[^\\]*$/,"")])==null?void 0:l.title)||"",h=`${v?`${v} > `:""}${r.title}`,f=I(r.title,t);f&&(n[h]=[...n[h]||[],{type:"title",path:a,display:f}]),r.customFields&&Object.entries(r.customFields).forEach(([s,p])=>{p.forEach(i=>{const u=I(i,t);u&&(n[h]=[...n[h]||[],{type:"custom",path:a,index:s,display:u}])})});for(const s of r.contents){const p=I(s.header,t);p&&(n[h]=[...n[h]||[],{type:"heading",path:a+(s.slug?`#${s.slug}`:""),display:p}]);for(const i of s.contents){const u=I(i,t);u&&(n[h]=[...n[h]||[],{type:"content",header:s.header,path:a+(s.slug?`#${s.slug}`:""),display:u}])}}}return Object.keys(n).sort((a,r)=>Y(n[a])-Y(n[r])).map(a=>({title:a,contents:n[a]}))},ze=t=>{const e=D(),l=L([]),n=j(()=>Oe.value[e.value]),a=ae(r=>{l.value=r?ke(r,n.value):[]},Se);return ve([t,e],()=>{a(t.value)}),l};var Pe=se({name:"SearchResult",props:{query:{type:String,required:!0}},emits:{close:()=>!0,updateQuery:t=>!0},setup(t,{emit:e}){const l=re(),n=ie(),a=D(),r=le(Ie),{history:v,addHistory:h,removeHistory:f}=xe(),s=ue(t,"query"),p=ze(s),i=L(0),u=L(0),P=L(),O=j(()=>p.value.length>0),k=j(()=>p.value[i.value]||null),W=()=>{i.value=i.value>0?i.value-1:p.value.length-1,u.value=k.value.contents.length-1},Z=()=>{i.value=i.value<p.value.length-1?i.value+1:0,u.value=0},$=()=>{u.value<k.value.contents.length-1?u.value=u.value+1:Z()},_=()=>{u.value>0?u.value=u.value-1:W()},Q=c=>c.map(d=>pe(d)?d:o(d[0],d[1])),R=c=>{if(c.type==="custom"){const d=be[c.index]||"$content",[m,S=""]=he(d)?d[a.value].split("$content"):d.split("$content");return Q([m,...c.display,S])}return Q(c.display)},z=()=>{i.value=0,u.value=0,e("updateQuery",""),e("close")};return ce(()=>{oe("keydown",c=>{if(O.value){if(c.key==="ArrowUp")_();else if(c.key==="ArrowDown")$();else if(c.key==="Enter"){const d=k.value.contents[u.value];n.path!==d.path&&(h(d),l.push(d.path),z())}}}),Le(P.value,{reserveScrollBarGap:!0})}),de(()=>{Ce()}),()=>o("div",{class:["search-pro-result",{empty:s.value===""?v.value.length===0:!O.value}],ref:P},s.value===""?v.value.length?o("ul",{class:"search-pro-result-list"},o("li",{class:"search-pro-result-list-item"},[o("div",{class:"search-pro-result-title"},r.value.history),v.value.map((c,d)=>o(q,{to:c.path,class:["search-pro-result-item",{active:u.value===d}],onClick:()=>{console.log("click"),z()}},()=>[o(T,{class:"search-pro-result-type"}),o("div",{class:"search-pro-result-content"},[c.type==="content"&&c.header?o("div",{class:"content-header"},c.header):null,o("div",R(c))]),o("button",{class:"search-pro-close-icon",onClick:m=>{m.preventDefault(),m.stopPropagation(),f(d)}},o(U))]))])):r.value.emptyHistory:O.value?o("ul",{class:"search-pro-result-list"},p.value.map(({title:c,contents:d},m)=>{const S=i.value===m;return o("li",{class:["search-pro-result-list-item",{active:S}]},[o("div",{class:"search-pro-result-title"},c||"Documentation"),d.map((y,ee)=>{const M=S&&u.value===ee;return o(q,{to:y.path,class:["search-pro-result-item",{active:M,"aria-selected":M}],onClick:()=>{h(y),z()}},()=>[y.type==="content"?null:o(y.type==="title"?X:y.type==="heading"?F:K,{class:"search-pro-result-type"}),o("div",{class:"search-pro-result-content"},[y.type==="content"&&y.header?o("div",{class:"content-header"},y.header):null,o("div",R(y))])])})])})):r.value.emptyResult)}});export{Pe as default};
