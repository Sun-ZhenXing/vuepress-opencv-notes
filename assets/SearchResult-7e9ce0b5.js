import{u as b,y as F,a as E,A as $,m as _,b as ee,l as te,s as le,n as oe,o as ae,e as ne,c as se,h as re,r as M,d as ie,H as ue,x as T,f as ce}from"./app-c2c2ce0f.js";import{r as H,h as L,c as ve,u as de,ab as he,o as fe,I as pe,j as i,a5 as z,z as ye,K as me,i as ge}from"./framework-8980b429.js";function we(t){if(Array.isArray(t)){for(var e=0,l=Array(t.length);e<t.length;e++)l[e]=t[e];return l}else return Array.from(t)}var j=!1;if(typeof window<"u"){var V={get passive(){j=!0}};window.addEventListener("testPassive",null,V),window.removeEventListener("testPassive",null,V)}var Q=typeof window<"u"&&window.navigator&&window.navigator.platform&&(/iP(ad|hone|od)/.test(window.navigator.platform)||window.navigator.platform==="MacIntel"&&window.navigator.maxTouchPoints>1),w=[],R=!1,A=-1,S=void 0,x=void 0,k=function(e){return w.some(function(l){return!!(l.options.allowTouchMove&&l.options.allowTouchMove(e))})},Z=function(e){var l=e||window.event;return k(l.target)||l.touches.length>1?!0:(l.preventDefault&&l.preventDefault(),!1)},He=function(e){if(x===void 0){var l=!!e&&e.reserveScrollBarGap===!0,o=window.innerWidth-document.documentElement.clientWidth;l&&o>0&&(x=document.body.style.paddingRight,document.body.style.paddingRight=o+"px")}S===void 0&&(S=document.body.style.overflow,document.body.style.overflow="hidden")},Se=function(){x!==void 0&&(document.body.style.paddingRight=x,x=void 0),S!==void 0&&(document.body.style.overflow=S,S=void 0)},xe=function(e){return e?e.scrollHeight-e.scrollTop<=e.clientHeight:!1},Ye=function(e,l){var o=e.targetTouches[0].clientY-A;return k(e.target)?!1:l&&l.scrollTop===0&&o>0||xe(l)&&o<0?Z(e):(e.stopPropagation(),!0)},Ie=function(e,l){if(!e){console.error("disableBodyScroll unsuccessful - targetElement must be provided when calling disableBodyScroll on IOS devices.");return}if(!w.some(function(a){return a.targetElement===e})){var o={targetElement:e,options:l||{}};w=[].concat(we(w),[o]),Q?(e.ontouchstart=function(a){a.targetTouches.length===1&&(A=a.targetTouches[0].clientY)},e.ontouchmove=function(a){a.targetTouches.length===1&&Ye(a,e)},R||(document.addEventListener("touchmove",Z,j?{passive:!1}:void 0),R=!0)):He(l)}},Le=function(){Q?(w.forEach(function(e){e.targetElement.ontouchstart=null,e.targetElement.ontouchmove=null}),R&&(document.removeEventListener("touchmove",Z,j?{passive:!1}:void 0),R=!1),A=-1):Se(),w=[]};const Re="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",Ze="search-pro-history-results",m=b(Ze,[]),Ce=()=>({history:m,addHistory:t=>{m.value.length<M?m.value=[t,...m.value]:m.value=[t,...m.value.slice(0,M-1)]},removeHistory:t=>{m.value=[...m.value.slice(0,t),...m.value.slice(t+1)]}}),Oe=H(Re),We=L(()=>JSON.parse(F(Oe.value))),I=(t,e)=>{const l=t.toLowerCase(),o=e.toLowerCase(),a=[];let s=0,d=0;const h=(n,f=!1)=>{let r="";d===0?r=n.length>20?`… ${n.slice(-20)}`:n:f?r=n.length+d>100?`${n.slice(0,100-d)}… `:n:r=n.length>20?`${n.slice(0,20)} … ${n.slice(-20)}`:n,r&&a.push(r),d+=r.length,f||(a.push(["strong",e]),d+=e.length,d>=100&&a.push(" …"))};let p=l.indexOf(o,s);if(p===-1)return null;for(;p>=0;){const n=p+o.length;if(h(t.slice(s,p)),s=n,d>100)break;p=l.indexOf(o,s)}return d<100&&h(t.slice(s),!0),a},D=t=>t.reduce((e,{type:l})=>e+(l==="title"?50:l==="heading"?20:l==="custom"?10:1),0),je=(t,e)=>{var l;const o={};for(const[a,s]of T(e)){const d=((l=e[a.replace(/\/[^\\]*$/,"")])==null?void 0:l.title)||"",h=`${d?`${d} > `:""}${s.title}`,p=I(s.title,t);p&&(o[h]=[...o[h]||[],{type:"title",path:a,display:p}]),s.customFields&&T(s.customFields).forEach(([n,f])=>{f.forEach(r=>{const u=I(r,t);u&&(o[h]=[...o[h]||[],{type:"custom",path:a,index:n,display:u}])})});for(const n of s.contents){const f=I(n.header,t);f&&(o[h]=[...o[h]||[],{type:"heading",path:a+(n.slug?`#${n.slug}`:""),display:f}]);for(const r of n.contents){const u=I(r,t);u&&(o[h]=[...o[h]||[],{type:"content",header:n.header,path:a+(n.slug?`#${n.slug}`:""),display:u}])}}}return ce(o).sort((a,s)=>D(o[a])-D(o[s])).map(a=>({title:a,contents:o[a]}))},Ae=t=>{const e=E(),l=H([]),o=L(()=>We.value[e.value]),a=ie(s=>{l.value=s?je(s,o.value):[]},ue);return ye([t,e],()=>{a(t.value)}),l};var Ge=ve({name:"SearchResult",props:{query:{type:String,required:!0}},emits:{close:()=>!0,updateQuery:t=>!0},setup(t,{emit:e}){const l=se(),o=de(),a=E(),s=$(_),{history:d,addHistory:h,removeHistory:p}=Ce(),n=he(t,"query"),f=Ae(n),r=H(0),u=H(0),K=H(),C=L(()=>f.value.length>0),O=L(()=>f.value[r.value]||null),B=()=>{r.value=r.value>0?r.value-1:f.value.length-1,u.value=O.value.contents.length-1},J=()=>{r.value=r.value<f.value.length-1?r.value+1:0,u.value=0},P=()=>{u.value<O.value.contents.length-1?u.value=u.value+1:J()},U=()=>{u.value>0?u.value=u.value-1:B()},N=c=>c.map(v=>ge(v)?v:i(v[0],v[1])),G=c=>{if(c.type==="custom"){const v=re[c.index]||"$content",[g,Y=""]=me(v)?v[a.value].split("$content"):v.split("$content");return N([g,...c.display,Y])}return N(c.display)},W=()=>{r.value=0,u.value=0,e("updateQuery",""),e("close")};return fe(()=>{ee("keydown",c=>{if(C.value){if(c.key==="ArrowUp")U();else if(c.key==="ArrowDown")P();else if(c.key==="Enter"){const v=O.value.contents[u.value];l.value.path!==v.path&&(h(v),o.push(v.path),W())}}}),Ie(K.value,{reserveScrollBarGap:!0})}),pe(()=>{Le()}),()=>i("div",{class:["search-pro-result",{empty:n.value===""?d.value.length===0:!C.value}],ref:K},n.value===""?d.value.length?i("ul",{class:"search-pro-result-list"},i("li",{class:"search-pro-result-list-item"},[i("div",{class:"search-pro-result-title"},s.value.history),d.value.map((c,v)=>i(z,{to:c.path,class:["search-pro-result-item",{active:u.value===v}],onClick:()=>{W()}},()=>[i(te,{class:"search-pro-result-type"}),i("div",{class:"search-pro-result-content"},[c.type==="content"&&c.header?i("div",{class:"content-header"},c.header):null,i("div",G(c))]),i("button",{class:"search-pro-close-icon",onClick:g=>{g.preventDefault(),g.stopPropagation(),p(v)}},i(le))]))])):s.value.emptyHistory:C.value?i("ul",{class:"search-pro-result-list"},f.value.map(({title:c,contents:v},g)=>{const Y=r.value===g;return i("li",{class:["search-pro-result-list-item",{active:Y}]},[i("div",{class:"search-pro-result-title"},c||"Documentation"),v.map((y,X)=>{const q=Y&&u.value===X;return i(z,{to:y.path,class:["search-pro-result-item",{active:q,"aria-selected":q}],onClick:()=>{h(y),W()}},()=>[y.type==="content"?null:i(y.type==="title"?oe:y.type==="heading"?ae:ne,{class:"search-pro-result-type"}),i("div",{class:"search-pro-result-content"},[y.type==="content"&&y.header?i("div",{class:"content-header"},y.header):null,i("div",G(y))])])})])})):s.value.emptyResult)}});export{Ge as default};
