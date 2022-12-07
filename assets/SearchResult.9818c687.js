import{u as te,g as b,h as E,i as oe,j as le,k as ne,l as M,S as ae,t as se,m as re,n as ie,p as ue,q as l,R as k,s as A,v as ce,x as de,y as ve}from"./app.884e9959.js";function he(t){if(Array.isArray(t)){for(var e=0,o=Array(t.length);e<t.length;e++)o[e]=t[e];return o}else return Array.from(t)}var F=!1;if(typeof window<"u"){var j={get passive(){F=!0}};window.addEventListener("testPassive",null,j),window.removeEventListener("testPassive",null,j)}var X=typeof window<"u"&&window.navigator&&window.navigator.platform&&(/iP(ad|hone|od)/.test(window.navigator.platform)||window.navigator.platform==="MacIntel"&&window.navigator.maxTouchPoints>1),w=[],B=!1,P=-1,C=void 0,L=void 0,z=function(e){return w.some(function(o){return!!(o.options.allowTouchMove&&o.options.allowTouchMove(e))})},N=function(e){var o=e||window.event;return z(o.target)||o.touches.length>1?!0:(o.preventDefault&&o.preventDefault(),!1)},fe=function(e){if(L===void 0){var o=!!e&&e.reserveScrollBarGap===!0,n=window.innerWidth-document.documentElement.clientWidth;o&&n>0&&(L=document.body.style.paddingRight,document.body.style.paddingRight=n+"px")}C===void 0&&(C=document.body.style.overflow,document.body.style.overflow="hidden")},pe=function(){L!==void 0&&(document.body.style.paddingRight=L,L=void 0),C!==void 0&&(document.body.style.overflow=C,C=void 0)},me=function(e){return e?e.scrollHeight-e.scrollTop<=e.clientHeight:!1},ye=function(e,o){var n=e.targetTouches[0].clientY-P;return z(e.target)?!1:o&&o.scrollTop===0&&n>0||me(o)&&n<0?N(e):(e.stopPropagation(),!0)},ge=function(e,o){if(!e){console.error("disableBodyScroll unsuccessful - targetElement must be provided when calling disableBodyScroll on IOS devices.");return}if(!w.some(function(a){return a.targetElement===e})){var n={targetElement:e,options:o||{}};w=[].concat(he(w),[n]),X?(e.ontouchstart=function(a){a.targetTouches.length===1&&(P=a.targetTouches[0].clientY)},e.ontouchmove=function(a){a.targetTouches.length===1&&ye(a,e)},B||(document.addEventListener("touchmove",N,F?{passive:!1}:void 0),B=!0)):fe(o)}},we=function(){X?(w.forEach(function(e){e.targetElement.ontouchstart=null,e.targetElement.ontouchmove=null}),B&&(document.removeEventListener("touchmove",N,F?{passive:!1}:void 0),B=!1),P=-1):pe(),w=[]};const be="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";const G=()=>l(A,{name:"close"},()=>l("path",{d:"m925.468 822.294-303.27-310.288L925.51 201.674c34.683-27.842 38.3-75.802 8.122-107.217-30.135-31.37-82.733-34.259-117.408-6.463L512.001 399.257 207.777 87.993C173.1 60.197 120.504 63.087 90.369 94.456c-30.179 31.415-26.561 79.376 8.122 107.217L401.8 512.005l-303.27 310.29c-34.724 27.82-38.34 75.846-8.117 107.194 30.135 31.437 82.729 34.327 117.408 6.486L512 624.756l304.177 311.22c34.68 27.84 87.272 24.95 117.408-6.487 30.223-31.348 26.56-79.375-8.118-107.195z"}));G.displayName="CloseIcon";const I=()=>l(A,{name:"heading"},()=>l("path",{d:"M250.4 704.6H64V595.4h202.4l26.2-166.6H94V319.6h214.4L352 64h127.8l-43.6 255.4h211.2L691 64h126.2l-43.6 255.4H960v109.2H756.2l-24.6 166.6H930v109.2H717L672 960H545.8l43.6-255.4H376.6L333 960H206.8l43.6-255.4zm168.4-276L394 595.4h211.2l24.6-166.6h-211z"}));I.displayName="HeadingIcon";const K=()=>l(A,{name:"heart"},()=>l("path",{d:"M1024 358.156C1024 195.698 892.3 64 729.844 64c-86.362 0-164.03 37.218-217.844 96.49C458.186 101.218 380.518 64 294.156 64 131.698 64 0 195.698 0 358.156 0 444.518 37.218 522.186 96.49 576H96l320 320c32 32 64 64 96 64s64-32 96-64l320-320h-.49c59.272-53.814 96.49-131.482 96.49-217.844zM841.468 481.232 517.49 805.49a2981.962 2981.962 0 0 1-5.49 5.48c-1.96-1.95-3.814-3.802-5.49-5.48L182.532 481.234C147.366 449.306 128 405.596 128 358.156 128 266.538 202.538 192 294.156 192c47.44 0 91.15 19.366 123.076 54.532L512 350.912l94.768-104.378C638.696 211.366 682.404 192 729.844 192 821.462 192 896 266.538 896 358.156c0 47.44-19.368 91.15-54.532 123.076z"}));K.displayName="HeartIcon";const U=()=>l(A,{name:"history"},()=>l("path",{d:"M512 1024a512 512 0 1 1 512-512 512 512 0 0 1-512 512zm0-896a384 384 0 1 0 384 384 384 384 0 0 0-384-384zm192 448H512a64 64 0 0 1-64-64V320a64 64 0 0 1 128 0v128h128a64 64 0 0 1 0 128z"}));U.displayName="HistoryIcon";const Q=()=>l(A,{name:"title"},()=>l("path",{d:"M512 256c70.656 0 134.656 28.672 180.992 75.008A254.933 254.933 0 0 1 768 512c0 83.968-41.024 157.888-103.488 204.48C688.96 748.736 704 788.48 704 832c0 105.984-86.016 192-192 192-106.048 0-192-86.016-192-192h128a64 64 0 1 0 128 0 64 64 0 0 0-64-64 255.19 255.19 0 0 1-181.056-75.008A255.403 255.403 0 0 1 256 512c0-83.968 41.024-157.824 103.488-204.544C335.04 275.264 320 235.584 320 192A192 192 0 0 1 512 0c105.984 0 192 85.952 192 192H576a64.021 64.021 0 0 0-128 0c0 35.328 28.672 64 64 64zM384 512c0 70.656 57.344 128 128 128s128-57.344 128-128-57.344-128-128-128-128 57.344-128 128z"}));Q.displayName="TitleIcon";const Ce={},Le=300,q=5,Ae={"/":{cancel:"取消",placeholder:"搜索",search:"搜索",select:"选择",navigate:"切换",exit:"关闭",history:"搜索历史",emptyHistory:"无搜索历史",emptyResult:"没有找到结果",loading:"正在加载搜索索引..."}},He="search-pro-history-results",y=te(He,[]),Se=()=>({history:y,addHistory:t=>{y.value.length<q?y.value=[t,...y.value]:y.value=[t,...y.value.slice(0,q-1)]},removeHistory:t=>{y.value=[...y.value.slice(0,t),...y.value.slice(t+1)]}}),Ee=b(be),Be=E(()=>JSON.parse(ce(Ee.value))),S=(t,e)=>{const o=t.toLowerCase(),n=e.toLowerCase(),a=[];let r=0,v=0;const h=(s,f=!1)=>{let i="";v===0?i=s.length>20?`… ${s.slice(-20)}`:s:f?i=s.length+v>100?`${s.slice(0,100-v)}… `:s:i=s.length>20?`${s.slice(0,20)} … ${s.slice(-20)}`:s,i&&a.push(i),v+=i.length,f||(a.push(["strong",e]),v+=e.length,v>=100&&a.push(" …"))};let p=o.indexOf(n,r);if(p===-1)return null;for(;p>=0;){const s=p+n.length;if(h(t.slice(r,p)),r=s,v>100)break;p=o.indexOf(n,r)}return v<100&&h(t.slice(r),!0),a},J=t=>t.reduce((e,{type:o})=>e+(o==="title"?50:o==="heading"?20:o==="custom"?10:1),0),Ne=(t,e)=>{var o;const n={};for(const[a,r]of Object.entries(e)){const v=((o=e[a.replace(/\/[^\\]*$/,"")])==null?void 0:o.title)||"",h=`${v?`${v} > `:""}${r.title}`,p=S(r.title,t);p&&(n[h]=[...n[h]||[],{type:"title",path:a,display:p}]),r.customFields&&Object.entries(r.customFields).forEach(([s,f])=>{f.forEach(i=>{const u=S(i,t);u&&(n[h]=[...n[h]||[],{type:"custom",path:a,index:s,display:u}])})});for(const s of r.contents){const f=S(s.header,t);f&&(n[h]=[...n[h]||[],{type:"heading",path:a+(s.slug?`#${s.slug}`:""),display:f}]);for(const i of s.contents){const u=S(i,t);u&&(n[h]=[...n[h]||[],{type:"content",header:s.header,path:a+(s.slug?`#${s.slug}`:""),display:u}])}}}return Object.keys(n).sort((a,r)=>J(n[a])-J(n[r])).map(a=>({title:a,contents:n[a]}))},Te=t=>{const e=M(),o=b([]),n=E(()=>Be.value[e.value]),a=ve(r=>{o.value=r?Ne(r,n.value):[]},Le);return de([t,e],()=>{a(t.value)}),o};var De=oe({name:"SearchResult",props:{query:{type:String,required:!0}},emits:{close:()=>!0,updateQuery:t=>!0},setup(t,{emit:e}){const o=le(),n=ne(),a=M(),r=ae(Ae),{history:v,addHistory:h,removeHistory:p}=Se(),s=se(t,"query"),f=Te(s),i=b(0),u=b(0),W=b(),T=E(()=>f.value.length>0),V=E(()=>f.value[i.value]||null),O=()=>{i.value=i.value>0?i.value-1:f.value.length-1,u.value=V.value.contents.length-1},Z=()=>{i.value=i.value<f.value.length-1?i.value+1:0,u.value=0},$=()=>{u.value<V.value.contents.length-1?u.value=u.value+1:Z()},_=()=>{u.value>0?u.value=u.value-1:O()},Y=c=>c.map(d=>typeof d=="string"?d:l(d[0],d[1])),x=c=>{if(c.type==="custom"){const d=Ce[c.index]||"$content",[g,H=""]=typeof d=="object"?d[a.value].split("$content"):d.split("$content");return Y([g,...c.display,H])}return Y(c.display)},D=()=>{i.value=0,u.value=0,e("updateQuery",""),e("close")};return re(()=>{ie("keydown",c=>{if(T.value){if(c.key==="ArrowUp")_();else if(c.key==="ArrowDown")$();else if(c.key==="Enter"){const d=V.value.contents[u.value];n.path!==d.path&&(h(d),o.push(d.path),D())}}}),ge(W.value,{reserveScrollBarGap:!0})}),ue(()=>{we()}),()=>l("div",{class:["search-pro-result",{empty:s.value===""?v.value.length===0:!T.value}],ref:W},s.value===""?v.value.length?l("ul",{class:"search-pro-result-list"},l("li",{class:"search-pro-result-list-item"},[l("div",{class:"search-pro-result-title"},r.value.history),v.value.map((c,d)=>l(k,{to:c.path,class:["search-pro-result-item",{active:u.value===d}],onClick:()=>{console.log("click"),D()}},()=>[l(U,{class:"search-pro-result-type"}),l("div",{class:"search-pro-result-content"},[c.type==="content"&&c.header?l("div",{class:"content-header"},c.header):null,l("div",x(c))]),l("button",{class:"search-pro-close-icon",onClick:g=>{g.preventDefault(),g.stopPropagation(),p(d)}},l(G))]))])):r.value.emptyHistory:T.value?l("ul",{class:"search-pro-result-list"},f.value.map(({title:c,contents:d},g)=>{const H=i.value===g;return l("li",{class:["search-pro-result-list-item",{active:H}]},[l("div",{class:"search-pro-result-title"},c||"Documentation"),d.map((m,ee)=>{const R=H&&u.value===ee;return l(k,{to:m.path,class:["search-pro-result-item",{active:R,"aria-selected":R}],onClick:()=>{h(m),D()}},()=>[m.type==="content"?null:l(m.type==="title"?Q:m.type==="heading"?I:K,{class:"search-pro-result-type"}),l("div",{class:"search-pro-result-content"},[m.type==="content"&&m.header?l("div",{class:"content-header"},m.header):null,l("div",x(m))])])})])})):r.value.emptyResult)}});export{De as default};
