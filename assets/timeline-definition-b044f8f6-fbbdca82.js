import{aC as bt,aD as kt,s as vt,$ as rt,M as $,X as V,e as wt,aE as St,aF as $t}from"./mermaid.esm.min-a4057167.js";import{h as tt}from"./arc-f81a5cae-3f08bf64.js";import{L as Et}from"./is_dark-18838fe5-d52d8805.js";import"./app-39e69976.js";import"./framework-546207d5.js";import"./constant-2fe7eae5-45b4460e.js";var J=function(){var i=function(f,s,o,h){for(o=o||{},h=f.length;h--;o[f[h]]=s);return o},t=[1,2],e=[1,5],r=[6,9,11,17,18,20,22,23,26,27,28],n=[1,15],c=[1,16],l=[1,17],p=[1,18],d=[1,19],g=[1,23],m=[1,24],b=[1,27],x=[4,6,9,11,17,18,20,22,23,26,27,28],_={trace:function(){},yy:{},symbols_:{error:2,start:3,timeline:4,document:5,EOF:6,directive:7,line:8,SPACE:9,statement:10,NEWLINE:11,openDirective:12,typeDirective:13,closeDirective:14,":":15,argDirective:16,title:17,acc_title:18,acc_title_value:19,acc_descr:20,acc_descr_value:21,acc_descr_multiline_value:22,section:23,period_statement:24,event_statement:25,period:26,event:27,open_directive:28,type_directive:29,arg_directive:30,close_directive:31,$accept:0,$end:1},terminals_:{2:"error",4:"timeline",6:"EOF",9:"SPACE",11:"NEWLINE",15:":",17:"title",18:"acc_title",19:"acc_title_value",20:"acc_descr",21:"acc_descr_value",22:"acc_descr_multiline_value",23:"section",26:"period",27:"event",28:"open_directive",29:"type_directive",30:"arg_directive",31:"close_directive"},productions_:[0,[3,3],[3,2],[5,0],[5,2],[8,2],[8,1],[8,1],[8,1],[7,4],[7,6],[10,1],[10,2],[10,2],[10,1],[10,1],[10,1],[10,1],[10,1],[24,1],[25,1],[12,1],[13,1],[16,1],[14,1]],performAction:function(f,s,o,h,u,a,v){var y=a.length-1;switch(u){case 1:return a[y-1];case 3:this.$=[];break;case 4:a[y-1].push(a[y]),this.$=a[y-1];break;case 5:case 6:this.$=a[y];break;case 7:case 8:this.$=[];break;case 11:h.getCommonDb().setDiagramTitle(a[y].substr(6)),this.$=a[y].substr(6);break;case 12:this.$=a[y].trim(),h.getCommonDb().setAccTitle(this.$);break;case 13:case 14:this.$=a[y].trim(),h.getCommonDb().setAccDescription(this.$);break;case 15:h.addSection(a[y].substr(8)),this.$=a[y].substr(8);break;case 19:h.addTask(a[y],0,""),this.$=a[y];break;case 20:h.addEvent(a[y].substr(2)),this.$=a[y];break;case 21:h.parseDirective("%%{","open_directive");break;case 22:h.parseDirective(a[y],"type_directive");break;case 23:a[y]=a[y].trim().replace(/'/g,'"'),h.parseDirective(a[y],"arg_directive");break;case 24:h.parseDirective("}%%","close_directive","timeline");break}},table:[{3:1,4:t,7:3,12:4,28:e},{1:[3]},i(r,[2,3],{5:6}),{3:7,4:t,7:3,12:4,28:e},{13:8,29:[1,9]},{29:[2,21]},{6:[1,10],7:22,8:11,9:[1,12],10:13,11:[1,14],12:4,17:n,18:c,20:l,22:p,23:d,24:20,25:21,26:g,27:m,28:e},{1:[2,2]},{14:25,15:[1,26],31:b},i([15,31],[2,22]),i(r,[2,8],{1:[2,1]}),i(r,[2,4]),{7:22,10:28,12:4,17:n,18:c,20:l,22:p,23:d,24:20,25:21,26:g,27:m,28:e},i(r,[2,6]),i(r,[2,7]),i(r,[2,11]),{19:[1,29]},{21:[1,30]},i(r,[2,14]),i(r,[2,15]),i(r,[2,16]),i(r,[2,17]),i(r,[2,18]),i(r,[2,19]),i(r,[2,20]),{11:[1,31]},{16:32,30:[1,33]},{11:[2,24]},i(r,[2,5]),i(r,[2,12]),i(r,[2,13]),i(x,[2,9]),{14:34,31:b},{31:[2,23]},{11:[1,35]},i(x,[2,10])],defaultActions:{5:[2,21],7:[2,2],27:[2,24],33:[2,23]},parseError:function(f,s){if(s.recoverable)this.trace(f);else{var o=new Error(f);throw o.hash=s,o}},parse:function(f){var s=this,o=[0],h=[],u=[null],a=[],v=this.table,y="",w=0,L=0,T=2,D=1,P=a.slice.call(arguments,1),k=Object.create(this.lexer),O={yy:{}};for(var X in this.yy)Object.prototype.hasOwnProperty.call(this.yy,X)&&(O.yy[X]=this.yy[X]);k.setInput(f,O.yy),O.yy.lexer=k,O.yy.parser=this,typeof k.yylloc>"u"&&(k.yylloc={});var Y=k.yylloc;a.push(Y);var xt=k.options&&k.options.ranges;typeof O.yy.parseError=="function"?this.parseError=O.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError;function _t(){var C;return C=h.pop()||k.lex()||D,typeof C!="number"&&(C instanceof Array&&(h=C,C=h.pop()),C=s.symbols_[C]||C),C}for(var S,A,I,Z,N={},B,H,Q,z;;){if(A=o[o.length-1],this.defaultActions[A]?I=this.defaultActions[A]:((S===null||typeof S>"u")&&(S=_t()),I=v[A]&&v[A][S]),typeof I>"u"||!I.length||!I[0]){var q="";z=[];for(B in v[A])this.terminals_[B]&&B>T&&z.push("'"+this.terminals_[B]+"'");k.showPosition?q="Parse error on line "+(w+1)+`:
`+k.showPosition()+`
Expecting `+z.join(", ")+", got '"+(this.terminals_[S]||S)+"'":q="Parse error on line "+(w+1)+": Unexpected "+(S==D?"end of input":"'"+(this.terminals_[S]||S)+"'"),this.parseError(q,{text:k.match,token:this.terminals_[S]||S,line:k.yylineno,loc:Y,expected:z})}if(I[0]instanceof Array&&I.length>1)throw new Error("Parse Error: multiple actions possible at state: "+A+", token: "+S);switch(I[0]){case 1:o.push(S),u.push(k.yytext),a.push(k.yylloc),o.push(I[1]),S=null,L=k.yyleng,y=k.yytext,w=k.yylineno,Y=k.yylloc;break;case 2:if(H=this.productions_[I[1]][1],N.$=u[u.length-H],N._$={first_line:a[a.length-(H||1)].first_line,last_line:a[a.length-1].last_line,first_column:a[a.length-(H||1)].first_column,last_column:a[a.length-1].last_column},xt&&(N._$.range=[a[a.length-(H||1)].range[0],a[a.length-1].range[1]]),Z=this.performAction.apply(N,[y,L,w,O.yy,I[1],u,a].concat(P)),typeof Z<"u")return Z;H&&(o=o.slice(0,-1*H*2),u=u.slice(0,-1*H),a=a.slice(0,-1*H)),o.push(this.productions_[I[1]][0]),u.push(N.$),a.push(N._$),Q=v[o[o.length-2]][o[o.length-1]],o.push(Q);break;case 3:return!0}}return!0}},M=function(){var f={EOF:1,parseError:function(s,o){if(this.yy.parser)this.yy.parser.parseError(s,o);else throw new Error(s)},setInput:function(s,o){return this.yy=o||this.yy||{},this._input=s,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},input:function(){var s=this._input[0];this.yytext+=s,this.yyleng++,this.offset++,this.match+=s,this.matched+=s;var o=s.match(/(?:\r\n?|\n).*/g);return o?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),s},unput:function(s){var o=s.length,h=s.split(/(?:\r\n?|\n)/g);this._input=s+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-o),this.offset-=o;var u=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),h.length-1&&(this.yylineno-=h.length-1);var a=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:h?(h.length===u.length?this.yylloc.first_column:0)+u[u.length-h.length].length-h[0].length:this.yylloc.first_column-o},this.options.ranges&&(this.yylloc.range=[a[0],a[0]+this.yyleng-o]),this.yyleng=this.yytext.length,this},more:function(){return this._more=!0,this},reject:function(){if(this.options.backtrack_lexer)this._backtrack=!0;else return this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno});return this},less:function(s){this.unput(this.match.slice(s))},pastInput:function(){var s=this.matched.substr(0,this.matched.length-this.match.length);return(s.length>20?"...":"")+s.substr(-20).replace(/\n/g,"")},upcomingInput:function(){var s=this.match;return s.length<20&&(s+=this._input.substr(0,20-s.length)),(s.substr(0,20)+(s.length>20?"...":"")).replace(/\n/g,"")},showPosition:function(){var s=this.pastInput(),o=new Array(s.length+1).join("-");return s+this.upcomingInput()+`
`+o+"^"},test_match:function(s,o){var h,u,a;if(this.options.backtrack_lexer&&(a={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(a.yylloc.range=this.yylloc.range.slice(0))),u=s[0].match(/(?:\r\n?|\n).*/g),u&&(this.yylineno+=u.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:u?u[u.length-1].length-u[u.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+s[0].length},this.yytext+=s[0],this.match+=s[0],this.matches=s,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(s[0].length),this.matched+=s[0],h=this.performAction.call(this,this.yy,this,o,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),h)return h;if(this._backtrack){for(var v in a)this[v]=a[v];return!1}return!1},next:function(){if(this.done)return this.EOF;this._input||(this.done=!0);var s,o,h,u;this._more||(this.yytext="",this.match="");for(var a=this._currentRules(),v=0;v<a.length;v++)if(h=this._input.match(this.rules[a[v]]),h&&(!o||h[0].length>o[0].length)){if(o=h,u=v,this.options.backtrack_lexer){if(s=this.test_match(h,a[v]),s!==!1)return s;if(this._backtrack){o=!1;continue}else return!1}else if(!this.options.flex)break}return o?(s=this.test_match(o,a[u]),s!==!1?s:!1):this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},lex:function(){var s=this.next();return s||this.lex()},begin:function(s){this.conditionStack.push(s)},popState:function(){var s=this.conditionStack.length-1;return s>0?this.conditionStack.pop():this.conditionStack[0]},_currentRules:function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},topState:function(s){return s=this.conditionStack.length-1-Math.abs(s||0),s>=0?this.conditionStack[s]:"INITIAL"},pushState:function(s){this.begin(s)},stateStackSize:function(){return this.conditionStack.length},options:{"case-insensitive":!0},performAction:function(s,o,h,u){switch(h){case 0:return this.begin("open_directive"),28;case 1:return this.begin("type_directive"),29;case 2:return this.popState(),this.begin("arg_directive"),15;case 3:return this.popState(),this.popState(),31;case 4:return 30;case 5:break;case 6:break;case 7:return 11;case 8:break;case 9:break;case 10:return 4;case 11:return 17;case 12:return this.begin("acc_title"),18;case 13:return this.popState(),"acc_title_value";case 14:return this.begin("acc_descr"),20;case 15:return this.popState(),"acc_descr_value";case 16:this.begin("acc_descr_multiline");break;case 17:this.popState();break;case 18:return"acc_descr_multiline_value";case 19:return 23;case 20:return 27;case 21:return 26;case 22:return 6;case 23:return"INVALID"}},rules:[/^(?:%%\{)/i,/^(?:((?:(?!\}%%)[^:.])*))/i,/^(?::)/i,/^(?:\}%%)/i,/^(?:((?:(?!\}%%).|\n)*))/i,/^(?:%(?!\{)[^\n]*)/i,/^(?:[^\}]%%[^\n]*)/i,/^(?:[\n]+)/i,/^(?:\s+)/i,/^(?:#[^\n]*)/i,/^(?:timeline\b)/i,/^(?:title\s[^#\n;]+)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:section\s[^#:\n;]+)/i,/^(?::\s[^#:\n;]+)/i,/^(?:[^#:\n;]+)/i,/^(?:$)/i,/^(?:.)/i],conditions:{open_directive:{rules:[1],inclusive:!1},type_directive:{rules:[2,3],inclusive:!1},arg_directive:{rules:[3,4],inclusive:!1},acc_descr_multiline:{rules:[17,18],inclusive:!1},acc_descr:{rules:[15],inclusive:!1},acc_title:{rules:[13],inclusive:!1},INITIAL:{rules:[0,5,6,7,8,9,10,11,12,14,16,19,20,21,22,23],inclusive:!0}}};return f}();_.lexer=M;function E(){this.yy={}}return E.prototype=_,_.Parser=E,new E}();J.parser=J;const It=J;let R="",st=0;const G=[],W=[],F=[],at=()=>bt,ot=(i,t,e)=>{kt(globalThis,i,t,e)},ct=function(){G.length=0,W.length=0,R="",F.length=0,vt()},lt=function(i){R=i,G.push(i)},ht=function(){return G},dt=function(){let i=et();const t=100;let e=0;for(;!i&&e<t;)i=et(),e++;return W.push(...F),W},pt=function(i,t,e){const r={id:st++,section:R,type:R,task:i,score:t||0,events:e?[e]:[]};F.push(r)},ut=function(i){F.find(t=>t.id===st-1).events.push(i)},yt=function(i){const t={section:R,type:R,description:i,task:i,classes:[]};W.push(t)},et=function(){const i=function(e){return F[e].processed};let t=!0;for(const[e,r]of F.entries())i(e),t=t&&r.processed;return t},Mt={clear:ct,getCommonDb:at,addSection:lt,getSections:ht,getTasks:dt,addTask:pt,addTaskOrg:yt,addEvent:ut,parseDirective:ot},Lt=Object.freeze(Object.defineProperty({__proto__:null,addEvent:ut,addSection:lt,addTask:pt,addTaskOrg:yt,clear:ct,default:Mt,getCommonDb:at,getSections:ht,getTasks:dt,parseDirective:ot},Symbol.toStringTag,{value:"Module"})),Dt=12,U=function(i,t){const e=i.append("rect");return e.attr("x",t.x),e.attr("y",t.y),e.attr("fill",t.fill),e.attr("stroke",t.stroke),e.attr("width",t.width),e.attr("height",t.height),e.attr("rx",t.rx),e.attr("ry",t.ry),t.class!==void 0&&e.attr("class",t.class),e},Ht=function(i,t){const e=i.append("circle").attr("cx",t.cx).attr("cy",t.cy).attr("class","face").attr("r",15).attr("stroke-width",2).attr("overflow","visible"),r=i.append("g");r.append("circle").attr("cx",t.cx-15/3).attr("cy",t.cy-15/3).attr("r",1.5).attr("stroke-width",2).attr("fill","#666").attr("stroke","#666"),r.append("circle").attr("cx",t.cx+15/3).attr("cy",t.cy-15/3).attr("r",1.5).attr("stroke-width",2).attr("fill","#666").attr("stroke","#666");function n(p){const d=tt().startAngle(Math.PI/2).endAngle(3*(Math.PI/2)).innerRadius(7.5).outerRadius(6.8181818181818175);p.append("path").attr("class","mouth").attr("d",d).attr("transform","translate("+t.cx+","+(t.cy+2)+")")}function c(p){const d=tt().startAngle(3*Math.PI/2).endAngle(5*(Math.PI/2)).innerRadius(7.5).outerRadius(6.8181818181818175);p.append("path").attr("class","mouth").attr("d",d).attr("transform","translate("+t.cx+","+(t.cy+7)+")")}function l(p){p.append("line").attr("class","mouth").attr("stroke",2).attr("x1",t.cx-5).attr("y1",t.cy+7).attr("x2",t.cx+5).attr("y2",t.cy+7).attr("class","mouth").attr("stroke-width","1px").attr("stroke","#666")}return t.score>3?n(r):t.score<3?c(r):l(r),e},Ct=function(i,t){const e=i.append("circle");return e.attr("cx",t.cx),e.attr("cy",t.cy),e.attr("class","actor-"+t.pos),e.attr("fill",t.fill),e.attr("stroke",t.stroke),e.attr("r",t.r),e.class!==void 0&&e.attr("class",e.class),t.title!==void 0&&e.append("title").text(t.title),e},gt=function(i,t){const e=t.text.replace(/<br\s*\/?>/gi," "),r=i.append("text");r.attr("x",t.x),r.attr("y",t.y),r.attr("class","legend"),r.style("text-anchor",t.anchor),t.class!==void 0&&r.attr("class",t.class);const n=r.append("tspan");return n.attr("x",t.x+t.textMargin*2),n.text(e),r},Pt=function(i,t){function e(n,c,l,p,d){return n+","+c+" "+(n+l)+","+c+" "+(n+l)+","+(c+p-d)+" "+(n+l-d*1.2)+","+(c+p)+" "+n+","+(c+p)}const r=i.append("polygon");r.attr("points",e(t.x,t.y,50,20,7)),r.attr("class","labelBox"),t.y=t.y+t.labelMargin,t.x=t.x+.5*t.labelMargin,gt(i,t)},Ot=function(i,t,e){const r=i.append("g"),n=K();n.x=t.x,n.y=t.y,n.fill=t.fill,n.width=e.width,n.height=e.height,n.class="journey-section section-type-"+t.num,n.rx=3,n.ry=3,U(r,n),ft(e)(t.text,r,n.x,n.y,n.width,n.height,{class:"journey-section section-type-"+t.num},e,t.colour)};let it=-1;const At=function(i,t,e){const r=t.x+e.width/2,n=i.append("g");it++;const c=300+5*30;n.append("line").attr("id","task"+it).attr("x1",r).attr("y1",t.y).attr("x2",r).attr("y2",c).attr("class","task-line").attr("stroke-width","1px").attr("stroke-dasharray","4 2").attr("stroke","#666"),Ht(n,{cx:r,cy:300+(5-t.score)*30,score:t.score});const l=K();l.x=t.x,l.y=t.y,l.fill=t.fill,l.width=e.width,l.height=e.height,l.class="task task-type-"+t.num,l.rx=3,l.ry=3,U(n,l),t.x+14,ft(e)(t.task,n,l.x,l.y,l.width,l.height,{class:"task"},e,t.colour)},jt=function(i,t){U(i,{x:t.startx,y:t.starty,width:t.stopx-t.startx,height:t.stopy-t.starty,fill:t.fill,class:"rect"}).lower()},Tt=function(){return{x:0,y:0,fill:void 0,"text-anchor":"start",width:100,height:100,textMargin:0,rx:0,ry:0}},K=function(){return{x:0,y:0,width:100,anchor:"start",height:100,rx:0,ry:0}},ft=function(){function i(n,c,l,p,d,g,m,b){const x=c.append("text").attr("x",l+d/2).attr("y",p+g/2+5).style("font-color",b).style("text-anchor","middle").text(n);r(x,m)}function t(n,c,l,p,d,g,m,b,x){const{taskFontSize:_,taskFontFamily:M}=b,E=n.split(/<br\s*\/?>/gi);for(let f=0;f<E.length;f++){const s=f*_-_*(E.length-1)/2,o=c.append("text").attr("x",l+d/2).attr("y",p).attr("fill",x).style("text-anchor","middle").style("font-size",_).style("font-family",M);o.append("tspan").attr("x",l+d/2).attr("dy",s).text(E[f]),o.attr("y",p+g/2).attr("dominant-baseline","central").attr("alignment-baseline","central"),r(o,m)}}function e(n,c,l,p,d,g,m,b){const x=c.append("switch"),_=x.append("foreignObject").attr("x",l).attr("y",p).attr("width",d).attr("height",g).attr("position","fixed").append("xhtml:div").style("display","table").style("height","100%").style("width","100%");_.append("div").attr("class","label").style("display","table-cell").style("text-align","center").style("vertical-align","middle").text(n),t(n,x,l,p,d,g,m,b),r(_,m)}function r(n,c){for(const l in c)l in c&&n.attr(l,c[l])}return function(n){return n.textPlacement==="fo"?e:n.textPlacement==="old"?i:t}}(),Nt=function(i){i.append("defs").append("marker").attr("id","arrowhead").attr("refX",5).attr("refY",2).attr("markerWidth",6).attr("markerHeight",4).attr("orient","auto").append("path").attr("d","M 0,0 V 4 L6,2 Z")};function mt(i,t){i.each(function(){var e=V(this),r=e.text().split(/(\s+|<br>)/).reverse(),n,c=[],l=1.1,p=e.attr("y"),d=parseFloat(e.attr("dy")),g=e.text(null).append("tspan").attr("x",0).attr("y",p).attr("dy",d+"em");for(let m=0;m<r.length;m++)n=r[r.length-1-m],c.push(n),g.text(c.join(" ").trim()),(g.node().getComputedTextLength()>t||n==="<br>")&&(c.pop(),g.text(c.join(" ").trim()),n==="<br>"?c=[""]:c=[n],g=e.append("tspan").attr("x",0).attr("y",p).attr("dy",l+"em").text(n))})}const Rt=function(i,t,e,r){const n=e%Dt-1,c=i.append("g");t.section=n,c.attr("class",(t.class?t.class+" ":"")+"timeline-node "+("section-"+n));const l=c.append("g"),p=c.append("g"),d=p.append("text").text(t.descr).attr("dy","1em").attr("alignment-baseline","middle").attr("dominant-baseline","middle").attr("text-anchor","middle").call(mt,t.width).node().getBBox(),g=r.fontSize&&r.fontSize.replace?r.fontSize.replace("px",""):r.fontSize;return t.height=d.height+g*1.1*.5+t.padding,t.height=Math.max(t.height,t.maxHeight),t.width=t.width+2*t.padding,p.attr("transform","translate("+t.width/2+", "+t.padding/2+")"),Bt(l,t,n),t},Ft=function(i,t,e){const r=i.append("g"),n=r.append("text").text(t.descr).attr("dy","1em").attr("alignment-baseline","middle").attr("dominant-baseline","middle").attr("text-anchor","middle").call(mt,t.width).node().getBBox(),c=e.fontSize&&e.fontSize.replace?e.fontSize.replace("px",""):e.fontSize;return r.remove(),n.height+c*1.1*.5+t.padding},Bt=function(i,t,e){i.append("path").attr("id","node-"+t.id).attr("class","node-bkg node-"+t.type).attr("d",`M0 ${t.height-5} v${-t.height+2*5} q0,-5 5,-5 h${t.width-2*5} q5,0 5,5 v${t.height-5} H0 Z`),i.append("line").attr("class","node-line-"+e).attr("x1",0).attr("y1",t.height).attr("x2",t.width).attr("y2",t.height)},j={drawRect:U,drawCircle:Ct,drawSection:Ot,drawText:gt,drawLabel:Pt,drawTask:At,drawBackgroundRect:jt,getTextObj:Tt,getNoteRect:K,initGraphics:Nt,drawNode:Rt,getVirtualNodeHeight:Ft},zt=function(i){Object.keys(i).forEach(function(t){conf[t]=i[t]})},Vt=function(i,t,e,r){const n=rt(),c=n.leftMargin?n.leftMargin:50;r.db.clear(),r.parser.parse(i+`
`),$.debug("timeline",r.db);const l=n.securityLevel;let p;l==="sandbox"&&(p=V("#i"+t));const d=(l==="sandbox"?V(p.nodes()[0].contentDocument.body):V("body")).select("#"+t);d.append("g");const g=r.db.getTasks(),m=r.db.getCommonDb().getDiagramTitle();$.debug("task",g),j.initGraphics(d);const b=r.db.getSections();$.debug("sections",b);let x=0,_=0,M=0,E=0,f=50+c,s=50;E=50;let o=0,h=!0;b.forEach(function(y){const w={number:o,descr:y,section:o,width:150,padding:20,maxHeight:x},L=j.getVirtualNodeHeight(d,w,n);$.debug("sectionHeight before draw",L),x=Math.max(x,L+20)});let u=0,a=0;$.debug("tasks.length",g.length);for(const[y,w]of g.entries()){const L={number:y,descr:w,section:w.section,width:150,padding:20,maxHeight:_},T=j.getVirtualNodeHeight(d,L,n);$.debug("taskHeight before draw",T),_=Math.max(_,T+20),u=Math.max(u,w.events.length);let D=0;for(let P=0;P<w.events.length;P++){const k={descr:w.events[P],section:w.section,number:w.section,width:150,padding:20,maxHeight:50};D+=j.getVirtualNodeHeight(d,k,n)}a=Math.max(a,D)}$.debug("maxSectionHeight before draw",x),$.debug("maxTaskHeight before draw",_),b&&b.length>0?b.forEach(y=>{const w={number:o,descr:y,section:o,width:150,padding:20,maxHeight:x};$.debug("sectionNode",w);const L=d.append("g"),T=j.drawNode(L,w,o,n);$.debug("sectionNode output",T),L.attr("transform",`translate(${f}, ${E})`),s+=x+50;const D=g.filter(P=>P.section===y);D.length>0&&nt(d,D,o,f,s,_,n,u,a,x,!1),f+=200*Math.max(D.length,1),s=E,o++}):(h=!1,nt(d,g,o,f,s,_,n,u,a,x,!0));const v=d.node().getBBox();$.debug("bounds",v),m&&d.append("text").text(m).attr("x",v.width/2-c).attr("font-size","4ex").attr("font-weight","bold").attr("y",20),M=h?x+_+150:_+100,d.append("g").attr("class","lineWrapper").append("line").attr("x1",c).attr("y1",M).attr("x2",v.width+3*c).attr("y2",M).attr("stroke-width",4).attr("stroke","black").attr("marker-end","url(#arrowhead)"),wt(void 0,d,n.timeline.padding?n.timeline.padding:50,n.timeline.useMaxWidth?n.timeline.useMaxWidth:!1)},nt=function(i,t,e,r,n,c,l,p,d,g,m){for(const b of t){const x={descr:b.task,section:e,number:e,width:150,padding:20,maxHeight:c};$.debug("taskNode",x);const _=i.append("g").attr("class","taskWrapper"),M=j.drawNode(_,x,e,l).height;if($.debug("taskHeight after draw",M),_.attr("transform",`translate(${r}, ${n})`),c=Math.max(c,M),b.events){const E=i.append("g").attr("class","lineWrapper");let f=c;n+=100,f=f+Wt(i,b.events,e,r,n,l),n-=100,E.append("line").attr("x1",r+190/2).attr("y1",n+c).attr("x2",r+190/2).attr("y2",n+c+(m?c:g)+d+120).attr("stroke-width",2).attr("stroke","black").attr("marker-end","url(#arrowhead)").attr("stroke-dasharray","5,5")}r=r+200,m&&!rt().timeline.disableMulticolor&&e++}n=n-10},Wt=function(i,t,e,r,n,c){let l=0;const p=n;n=n+100;for(const d of t){const g={descr:d,section:e,number:e,width:150,padding:20,maxHeight:50};$.debug("eventNode",g);const m=i.append("g").attr("class","eventWrapper"),b=j.drawNode(m,g,e,c).height;l=l+b,m.attr("transform",`translate(${r}, ${n})`),n=n+10+b}return n=p,l},Ut={setConf:zt,draw:Vt},Xt=i=>{let t="";for(let e=0;e<i.THEME_COLOR_LIMIT;e++)i["lineColor"+e]=i["lineColor"+e]||i["cScaleInv"+e],Et(i["lineColor"+e])?i["lineColor"+e]=St(i["lineColor"+e],20):i["lineColor"+e]=$t(i["lineColor"+e],20);for(let e=0;e<i.THEME_COLOR_LIMIT;e++){const r=""+(17-3*e);t+=`
    .section-${e-1} rect, .section-${e-1} path, .section-${e-1} circle, .section-${e-1} path  {
      fill: ${i["cScale"+e]};
    }
    .section-${e-1} text {
     fill: ${i["cScaleLabel"+e]};
    }
    .node-icon-${e-1} {
      font-size: 40px;
      color: ${i["cScaleLabel"+e]};
    }
    .section-edge-${e-1}{
      stroke: ${i["cScale"+e]};
    }
    .edge-depth-${e-1}{
      stroke-width: ${r};
    }
    .section-${e-1} line {
      stroke: ${i["cScaleInv"+e]} ;
      stroke-width: 3;
    }

    .lineWrapper line{
      stroke: ${i["cScaleLabel"+e]} ;
    }

    .disabled, .disabled circle, .disabled text {
      fill: lightgray;
    }
    .disabled text {
      fill: #efefef;
    }
    `}return t},Yt=i=>`
  .edge {
    stroke-width: 3;
  }
  ${Xt(i)}
  .section-root rect, .section-root path, .section-root circle  {
    fill: ${i.git0};
  }
  .section-root text {
    fill: ${i.gitBranchLabel0};
  }
  .icon-container {
    height:100%;
    display: flex;
    justify-content: center;
    align-items: center;
  }
  .edge {
    fill: none;
  }
  .eventWrapper  {
   filter: brightness(120%);
  }
`,Zt=Yt,ee={db:Lt,renderer:Ut,parser:It,styles:Zt};export{ee as diagram};
