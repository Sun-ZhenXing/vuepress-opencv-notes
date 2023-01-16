import{u as te,a as A,k as se,b as le,P as C,B as ae,c as ne}from"./app-9940734d.js";import{r as H,h as P,c as oe,u as re,L as ie,aa as ue,o as ce,n as ve,j as l,ab as q,z as de,K as he,i as fe}from"./framework-d3922052.js";function pe(t){if(Array.isArray(t)){for(var e=0,s=Array(t.length);e<t.length;e++)s[e]=t[e];return s}else return Array.from(t)}var F=!1;if(typeof window<"u"){var E={get passive(){F=!0}};window.addEventListener("testPassive",null,E),window.removeEventListener("testPassive",null,E)}var R=typeof window<"u"&&window.navigator&&window.navigator.platform&&(/iP(ad|hone|od)/.test(window.navigator.platform)||window.navigator.platform==="MacIntel"&&window.navigator.maxTouchPoints>1),w=[],V=!1,M=-1,L=void 0,b=void 0,Y=function(e){return w.some(function(s){return!!(s.options.allowTouchMove&&s.options.allowTouchMove(e))})},j=function(e){var s=e||window.event;return Y(s.target)||s.touches.length>1?!0:(s.preventDefault&&s.preventDefault(),!1)},ye=function(e){if(b===void 0){var s=!!e&&e.reserveScrollBarGap===!0,a=window.innerWidth-document.documentElement.clientWidth;s&&a>0&&(b=document.body.style.paddingRight,document.body.style.paddingRight=a+"px")}L===void 0&&(L=document.body.style.overflow,document.body.style.overflow="hidden")},ge=function(){b!==void 0&&(document.body.style.paddingRight=b,b=void 0),L!==void 0&&(document.body.style.overflow=L,L=void 0)},me=function(e){return e?e.scrollHeight-e.scrollTop<=e.clientHeight:!1},we=function(e,s){var a=e.targetTouches[0].clientY-M;return Y(e.target)?!1:s&&s.scrollTop===0&&a>0||me(s)&&a<0?j(e):(e.stopPropagation(),!0)},He=function(e,s){if(!e){console.error("disableBodyScroll unsuccessful - targetElement must be provided when calling disableBodyScroll on IOS devices.");return}if(!w.some(function(n){return n.targetElement===e})){var a={targetElement:e,options:s||{}};w=[].concat(pe(w),[a]),R?(e.ontouchstart=function(n){n.targetTouches.length===1&&(M=n.targetTouches[0].clientY)},e.ontouchmove=function(n){n.targetTouches.length===1&&we(n,e)},V||(document.addEventListener("touchmove",j,F?{passive:!1}:void 0),V=!0)):ye(s)}},Le=function(){R?(w.forEach(function(e){e.targetElement.ontouchstart=null,e.targetElement.ontouchmove=null}),V&&(document.removeEventListener("touchmove",j,F?{passive:!1}:void 0),V=!1),M=-1):ge(),w=[]};const be="eJylW1lTG9kV/itdvM50PNrAnreUXTUzVeNxElcmlUqlUhpQbDIycpDwxJmaKmEQAszmsViMsVkMtryAwAtCG/yXiW5368l/Id/t2xJ9b3cfUPnFS59zvnPXs1793HOh50vnj8xgJpno+bKn5/Oe/tRQJjGUSfd8+be///J5z4X4T4l06lZCYrt2OzF0+Xvt94KktTYr5uM9H9lkIj48pKfA3X9H/+Gu/sPIYHJgcOiGfns49a9EfybtgxrVfssWNGtv09xbNtaq1osJ68WU0MD2nhqTK+bqOP93Y9FHoaNqIHEnkUzdvgUSVMX7M4P9vhNg9SybfwBYq/zByNbM2sNgzNt3MzdTQ3pmJJMaHownJbg/2DTNQTUWH5nF+8RS/jOezuhDiZHheFJPZ+4mfcfWGiuajbfs5HUru956Nmds1K2TUfNFTcb9uedmIj6QGIZc6Heal9vcyzZrfCzp5MgNMP0jpBNM0oDPuYP9N+O3M4nhL0LSJMw3b7SQZr7e+FhfbS92bqe1XCSGH9KaVWz2s2a1ah7UxMab77Ar69L4QzrBJi+5BB8GPJdgz+81G4vmAsdgj4/Z2DybyLG9iqQkDCVnMAeqah5Vm7VsszKJ48seFVurx83aDqXv/ALE9CJas7FhjTXMQrF9ENnoqsCVphbRO4ztjXUzUiqwRRiK2dgzFveN2T3cF+PpOPt1xjp+iN29wP/KVxVlOHDBInpHhFQb1px12B7HshgPZ5uNNUVNWPdlIWEj2ld//kbBieg3RgbPEoxqOHatrQdscsJ4uqAgRHWFSELFtMgVrZWfZTV14WJ6ZEDvUEiQXs2cqrDjUWN+gc0vKTi9ukIkofo0VqpYB5vGs6zxQTYcEb1PV4gk1EVNnGe2+7xZ2VCgLuoKkYS61HYID8ZZ40CBuqQrRPoIf6GxxpY5fei7eaEvdIVKg+FC5KaMD6Nc/TvukSQwnH2ZSoPZFsoaP8L1EOtslSbY5GsV1bZNvmw0fESzNoutJ1tsptLKzaqoEV2h0mBRzVjKG2tvxFHgF9p3rFE9iI2Gj2lG+YBVn/uenVBMV6gEWFRje1PWs5xjFCWoqC5ojh08AwjW7y+DQwOpn9IKSEj/qfOdBAhrV+P92rXrinxYvxXv11Nnike0bweHRv6jSEf0pPP1U9x32OO+w4r7Xq+am1lzfYfvYKlu7SlWyTXWMFbKmM4aayXr+ahxcM814DAWSyEFTjnM7f56FadHu3w1/iPizNwsdyRL+WbtUMKE8bf59P5b4NMVPkJBRGOTj2Fgm0fZ5tErVuXx3ylsRPdSCbCoZsIOLUw0j59YH5YQskpgUd1LJcBisHgvrfFVfncev2fbs8YTeR1jsHo+DARkr+NFm0dz5vrL1soHCa/XcaBuKgHWp1mlmnYB8cAjISeB9emgwvGf0gioixwKrslxm7/OcEPOPfl7CfMixwxiI+AvwUQfNmtL4ogIi8kju6du7xaGFyHYKHgcdsdn+qKG9DuJfsdvng8vTOKF9XR/PBkf7goyQkJG9NupQaRK3SBGScSonh78b6IrwBgJGNOHYa66AuwlAXv14VQmnkkMdInLwwf7hIhLYp0swLnLBg4hhB8HBYowwhYRQTLbXWH3eKbkAkUo4ccRCMrvXm5HtZXK10BpcdNUaeXrp7ibiMfdRBx3I3w6txuIwwpFY7JsLs4gLgn0NjwrCfA2PPs4r7fhWYbj6txqkRibNXewxFMNZ4a+fISCSEfB6rhjZvymeaop0tFECgQfgv0Jo/aAPVmGtWzVN4yNfLN6H5tnlGtseoNN7qs6zy9ATDOq/TFjuxpiZlH937h1XU2G50p8/b76VjMKJWNmVAKM2Ut1I6l3aJ9yPKOe4wnnbh9PHnjmdsyNgvny0Hz83liq8FnAGdW2jXdbSGgCzymPHwPOKQ8hz3tOeRRpFtaNyQUnKnJiADUq4mGl4HOiIoWPUNCOisg95IEnwfYpqx/zrH7MWX0r/4pNF5FUwUrwlGJ953/ZUWG8kbCxqbd8J3ySU9f0YsHbEOtiG2LYhtZqwXxeY2tV41GpebTLL6k9FAkyrBNsn7JKvZ5VQhnAbULbywK3FrgavcGr0dvFavRiNdz5vPH8Hjv54LGdvVgNgi0Qvl1YdYmqCbezYr4sxLCRE8uVZn6AXBisvINCEk/h7Yru6USQLncjSAwBxSRxqjullfUdsWvmdrV5LKtFceksZkoVdjugUsShQ+euFHGoMJZu16yNade/v6oZxU05xuJ4YawR59DTd27pHQ4SFPthFwZF3tDaGpeLUBwVK+/H8glXKeRNflGLEXep2ThxVVStk8fW5kxw8cGdsBNZMEo4579bAMLlcqn01vR5BuzLQYF2ohFhk4LqLTwNdtbQl5FSEdX+eu3bazyealbnePdm+TCousgT5LupZIoHVUHMpCqsNld2JyKtlTh1xtNDVJ1VdSFb4Z2ItHSKAK0yLOYnYvLW4wk4I6s0axzkUUrj9kC9FLbasJjnmUK0avjo/QmhHtUYXAef+hbXBie9PyE0Cr7z1Lp4+cE5+NevX3FOvSg6+u1cDB0Fzqyn0wO6LzOpCjvHtQRvFffLHLqLveHumS8QB3aPhVgr7qr5WnFNXpGAZTuz69ifunU7OZjQ40MDwgDx/sKAb8fv8mefaWZ9ySrZxQ10SWpVYybPZpXkUWn5iVTTFutwu5p9fuTuZjAwNATC7WTqrrgvfX5jd07Lle++cxqX4jL2ddmjTQwPp4bT+uBQu6Lqp8spwmqtwiOrVGILk7hEgUtk7Y/xyLRdKuJGJV815/aFsCu3PYuvu0VL/ziYTNoTGUn7brZYMPgAVn7h13pWdhkNQKT89UXIyPsrE7obJIbm5AbiZDp+kdhfkXGIs2mVi6zyjs3vs2rBo10Z/t5qszKFbJLlJ9jMEo+AyzssV5anEsxElE6cIQlhqWQiZtYhEHmlM6HOM4bTrFIhEBid3gDc6X3nvtnCSjnCOdfwpSpX4N6pLw/O7rirHY/gjvtpQUJzHjG0aivW3jaeRcCKm1Ny6zjkqke0h+XlJ3vwYmjG1Et2sMhmctzSLW/wPiwCU/S/3/CR+HRqYJr9BLkxRdDqCJ7DRget5ZntDxH7OAOtGNkXZPcDsak5lVf6Hp2PVCUQwqLMhqYDq/M6mBco5FTCFRYSFiHkyjFyBJ9iPV9dhUhC4aXByRNudfyg8LpAJpJQqKrU6rDgnkyHQ6FjIhNJqE7EcjWegbPKJAd/UPA6YcotNwfZfhK9BqXX1PlIifLgf3bRmiv7NzPsBMBDJwHRHa7NifINDqPx665ROPYdH/rDwYykivbOipHhvPvit7fYy0V22syafSjyDUS+SptNIVEwPIMt1MzqiQKB3LX9mRTnxbMaKvOoWCkIvFx2SiFBMJkCCjo86nAjYCLtz6Q4Tn2hZmzvehFw5F0UEgQlMSjbXrWOK965xOyRuIgkFBqDa0VhRppH06LWi/PDduXGBn9KQjCSKvqk7e8EPG7wPukYBDzHUxqu7hq10m1VSBQMN9jzMyxf4zHH2j5qunwkk4fq/vCUkWDs2ukEN0GMjSN2Mub3ikkJOjz+wX5wdaZHgCh3NHNleBFWOWye7BmFCsyQ+X6DZesKHNxNMCOpAgZUSIqnWQdPhd90g8Oc+rGQsMh9bW7tT9fcz7c4HNJdm4TuHvmKi8PgCct9PEaZYPPT8DKYnVNRyq62Rt2Hk+PiHQvJSyqCazpyfCviF9zJj/VJtvma5XIf6+6YlSuCk5J5dcF5Vuuq3byf5s9jeVmy5GlNnj6S82Wj4MVJ4xuECy+9c+Kg9olzE0mo9liNchlLqUC1R9ghklCo9cxMNRuzvlNF9C4Tyc4cyrSsOo/Dh5eRTt4/P92sryATUVpyNCelhL9wlcqJQtwBXJw05+SbzM+DE7K0y9t+AqTKTlNTBLAiWvedV6et6ctJKsGFXCtq3zgZBA86zpoY7ulaUR90LGOQBKkU15e/e1nt6OWD3i3gBLIHjWZtW9GIG2yzd5R62Ul1eJFmD0usiCjeKirwLM2PpWu/ENx9lPbS/eCV7Dq2ttasqbfmy2pr+b3VeGPMyn28kO7LQDQIbedh87klFVC4DT8WEhbR2dTLZuOhVV63jl4pgAjQZCIJBduQxRP3sjF1jL41yuxCzHnpgVTu6+t41Jatf6y7n5dzPTAcpKR+M40n07T347Vhlt/A23E2j0BZXnAYOZlEwfBEwk56zfpLBPIKELIImUhCYUyVd+I3BG4QjKf9mRTHki7fx6Yq4liv9mdSPIoE5oTtj/uuSBRJi0QkoWJaKzuPtMMXKqYrRLrVLRz5yqQc+NiVc5lEwfC0ZC/LZhd9gZCcyEQSClFT/lUrew91NV80hEweOgmITk8mPfJb9hGKU9UZCMjGn2OixwMWHXUpFwMJGvUBNVZK4gcTovuJu6LeLm6HVU1eKbKI1H7GiDOLtyruKlH7CWOHEhz0X9HYbNl84bbg4QG98y34RY2IToqb5rtpHsW59EmxnJvv3KMK8W48r6k+cbtLTAsBVvsrJYwXCa9XjKWSUCtD4BmCTKOAUAc42sXm+AEh+5dpFBAeX8/jpx4V8TpGBsLja5kWDIQfJTS2DLz02H3u8V8h/CjBQwzev7l14S3b+9T+P6GcO7vcgrWalbXCv7W/UsK4zRNFa0wRxh1uf6WEYZ7qWU90C3kYJxeBgkCSvLzrB4EU2UWgIGKny29sldAqlYFipxvQIVNweKayXWdHDRkFP1Npf6WE+1CgfuAR7kMZ2vlKCV90TcTOrZuV+4hKeCyVe4ffkMmgF13zCuAOVOb8zmlmieXGkCufAnsJwQPGcIUZ8Sw6hiZTgkEuaZfjQ0N38a6iYtZXvO3bS3o/p+NRhUQPNp14ACv8Yv4BCj9ss+C2oXj76iEGQ3X6DyKgtRp7TC4LdRoOCkMwJCLI9qs8NxBiR9fnYPF2POAJLMLtcODsqCJsF2bu5Vr5GTZ/6IcEe+4hB8Mh0SluGk9OPL/bCSOnkSnBIDHt69TIjZt8dVAl9Q4qpt/kdL5MbnowYK8DiNqXOVfyAvY6gAo9GBBGHr+52n/P3u6Kl1vOetvv+tzIMPgko5Jv/fLL/wH0yjxt";const G=()=>l(C,{name:"close"},()=>l("path",{d:"m925.468 822.294-303.27-310.288L925.51 201.674c34.683-27.842 38.3-75.802 8.122-107.217-30.135-31.37-82.733-34.259-117.408-6.463L512.001 399.257 207.777 87.993C173.1 60.197 120.504 63.087 90.369 94.456c-30.179 31.415-26.561 79.376 8.122 107.217L401.8 512.005l-303.27 310.29c-34.724 27.82-38.34 75.846-8.117 107.194 30.135 31.437 82.729 34.327 117.408 6.486L512 624.756l304.177 311.22c34.68 27.84 87.272 24.95 117.408-6.487 30.223-31.348 26.56-79.375-8.118-107.195z"}));G.displayName="CloseIcon";const I=()=>l(C,{name:"heading"},()=>l("path",{d:"M250.4 704.6H64V595.4h202.4l26.2-166.6H94V319.6h214.4L352 64h127.8l-43.6 255.4h211.2L691 64h126.2l-43.6 255.4H960v109.2H756.2l-24.6 166.6H930v109.2H717L672 960H545.8l43.6-255.4H376.6L333 960H206.8l43.6-255.4zm168.4-276L394 595.4h211.2l24.6-166.6h-211z"}));I.displayName="HeadingIcon";const Q=()=>l(C,{name:"heart"},()=>l("path",{d:"M1024 358.156C1024 195.698 892.3 64 729.844 64c-86.362 0-164.03 37.218-217.844 96.49C458.186 101.218 380.518 64 294.156 64 131.698 64 0 195.698 0 358.156 0 444.518 37.218 522.186 96.49 576H96l320 320c32 32 64 64 96 64s64-32 96-64l320-320h-.49c59.272-53.814 96.49-131.482 96.49-217.844zM841.468 481.232 517.49 805.49a2981.962 2981.962 0 0 1-5.49 5.48c-1.96-1.95-3.814-3.802-5.49-5.48L182.532 481.234C147.366 449.306 128 405.596 128 358.156 128 266.538 202.538 192 294.156 192c47.44 0 91.15 19.366 123.076 54.532L512 350.912l94.768-104.378C638.696 211.366 682.404 192 729.844 192 821.462 192 896 266.538 896 358.156c0 47.44-19.368 91.15-54.532 123.076z"}));Q.displayName="HeartIcon";const W=()=>l(C,{name:"history"},()=>l("path",{d:"M512 1024a512 512 0 1 1 512-512 512 512 0 0 1-512 512zm0-896a384 384 0 1 0 384 384 384 384 0 0 0-384-384zm192 448H512a64 64 0 0 1-64-64V320a64 64 0 0 1 128 0v128h128a64 64 0 0 1 0 128z"}));W.displayName="HistoryIcon";const X=()=>l(C,{name:"title"},()=>l("path",{d:"M512 256c70.656 0 134.656 28.672 180.992 75.008A254.933 254.933 0 0 1 768 512c0 83.968-41.024 157.888-103.488 204.48C688.96 748.736 704 788.48 704 832c0 105.984-86.016 192-192 192-106.048 0-192-86.016-192-192h128a64 64 0 1 0 128 0 64 64 0 0 0-64-64 255.19 255.19 0 0 1-181.056-75.008A255.403 255.403 0 0 1 256 512c0-83.968 41.024-157.824 103.488-204.544C335.04 275.264 320 235.584 320 192A192 192 0 0 1 512 0c105.984 0 192 85.952 192 192H576a64.021 64.021 0 0 0-128 0c0 35.328 28.672 64 64 64zM384 512c0 70.656 57.344 128 128 128s128-57.344 128-128-57.344-128-128-128-128 57.344-128 128z"}));X.displayName="TitleIcon";const Ce={},Ne=300,x=5,ke={"/":{cancel:"取消",placeholder:"搜索",search:"搜索",select:"选择",navigate:"切换",exit:"关闭",history:"搜索历史",emptyHistory:"无搜索历史",emptyResult:"没有找到结果",loading:"正在加载搜索索引..."}},Pe="search-pro-history-results",g=te(Pe,[]),Ve=()=>({history:g,addHistory:t=>{g.value.length<x?g.value=[t,...g.value]:g.value=[t,...g.value.slice(0,x-1)]},removeHistory:t=>{g.value=[...g.value.slice(0,t),...g.value.slice(t+1)]}}),je=H(be),Se=P(()=>JSON.parse(ae(je.value))),k=(t,e)=>{const s=t.toLowerCase(),a=e.toLowerCase(),n=[];let r=0,d=0;const h=(o,f=!1)=>{let i="";d===0?i=o.length>20?`… ${o.slice(-20)}`:o:f?i=o.length+d>100?`${o.slice(0,100-d)}… `:o:i=o.length>20?`${o.slice(0,20)} … ${o.slice(-20)}`:o,i&&n.push(i),d+=i.length,f||(n.push(["strong",e]),d+=e.length,d>=100&&n.push(" …"))};let p=s.indexOf(a,r);if(p===-1)return null;for(;p>=0;){const o=p+a.length;if(h(t.slice(r,p)),r=o,d>100)break;p=s.indexOf(a,r)}return d<100&&h(t.slice(r),!0),n},J=t=>t.reduce((e,{type:s})=>e+(s==="title"?50:s==="heading"?20:s==="custom"?10:1),0),Be=(t,e)=>{var s;const a={};for(const[n,r]of Object.entries(e)){const d=((s=e[n.replace(/\/[^\\]*$/,"")])==null?void 0:s.title)||"",h=`${d?`${d} > `:""}${r.title}`,p=k(r.title,t);p&&(a[h]=[...a[h]||[],{type:"title",path:n,display:p}]),r.customFields&&Object.entries(r.customFields).forEach(([o,f])=>{f.forEach(i=>{const u=k(i,t);u&&(a[h]=[...a[h]||[],{type:"custom",path:n,index:o,display:u}])})});for(const o of r.contents){const f=k(o.header,t);f&&(a[h]=[...a[h]||[],{type:"heading",path:n+(o.slug?`#${o.slug}`:""),display:f}]);for(const i of o.contents){const u=k(i,t);u&&(a[h]=[...a[h]||[],{type:"content",header:o.header,path:n+(o.slug?`#${o.slug}`:""),display:u}])}}}return Object.keys(a).sort((n,r)=>J(a[n])-J(a[r])).map(n=>({title:n,contents:a[n]}))},Te=t=>{const e=A(),s=H([]),a=P(()=>Se.value[e.value]),n=ne(r=>{s.value=r?Be(r,a.value):[]},Ne);return de([t,e],()=>{n(t.value)}),s};var ze=oe({name:"SearchResult",props:{query:{type:String,required:!0}},emits:{close:()=>!0,updateQuery:t=>!0},setup(t,{emit:e}){const s=re(),a=ie(),n=A(),r=se(ke),{history:d,addHistory:h,removeHistory:p}=Ve(),o=ue(t,"query"),f=Te(o),i=H(0),u=H(0),z=H(),S=P(()=>f.value.length>0),B=P(()=>f.value[i.value]||null),Z=()=>{i.value=i.value>0?i.value-1:f.value.length-1,u.value=B.value.contents.length-1},D=()=>{i.value=i.value<f.value.length-1?i.value+1:0,u.value=0},$=()=>{u.value<B.value.contents.length-1?u.value=u.value+1:D()},_=()=>{u.value>0?u.value=u.value-1:Z()},K=c=>c.map(v=>fe(v)?v:l(v[0],v[1])),O=c=>{if(c.type==="custom"){const v=Ce[c.index]||"$content",[m,N=""]=he(v)?v[n.value].split("$content"):v.split("$content");return K([m,...c.display,N])}return K(c.display)},T=()=>{i.value=0,u.value=0,e("updateQuery",""),e("close")};return ce(()=>{le("keydown",c=>{if(S.value){if(c.key==="ArrowUp")_();else if(c.key==="ArrowDown")$();else if(c.key==="Enter"){const v=B.value.contents[u.value];a.path!==v.path&&(h(v),s.push(v.path),T())}}}),He(z.value,{reserveScrollBarGap:!0})}),ve(()=>{Le()}),()=>l("div",{class:["search-pro-result",{empty:o.value===""?d.value.length===0:!S.value}],ref:z},o.value===""?d.value.length?l("ul",{class:"search-pro-result-list"},l("li",{class:"search-pro-result-list-item"},[l("div",{class:"search-pro-result-title"},r.value.history),d.value.map((c,v)=>l(q,{to:c.path,class:["search-pro-result-item",{active:u.value===v}],onClick:()=>{console.log("click"),T()}},()=>[l(W,{class:"search-pro-result-type"}),l("div",{class:"search-pro-result-content"},[c.type==="content"&&c.header?l("div",{class:"content-header"},c.header):null,l("div",O(c))]),l("button",{class:"search-pro-close-icon",onClick:m=>{m.preventDefault(),m.stopPropagation(),p(v)}},l(G))]))])):r.value.emptyHistory:S.value?l("ul",{class:"search-pro-result-list"},f.value.map(({title:c,contents:v},m)=>{const N=i.value===m;return l("li",{class:["search-pro-result-list-item",{active:N}]},[l("div",{class:"search-pro-result-title"},c||"Documentation"),v.map((y,ee)=>{const U=N&&u.value===ee;return l(q,{to:y.path,class:["search-pro-result-item",{active:U,"aria-selected":U}],onClick:()=>{h(y),T()}},()=>[y.type==="content"?null:l(y.type==="title"?X:y.type==="heading"?I:Q,{class:"search-pro-result-type"}),l("div",{class:"search-pro-result-content"},[y.type==="content"&&y.header?l("div",{class:"content-header"},y.header):null,l("div",O(y))])])})])})):r.value.emptyResult)}});export{ze as default};
