(function(root){
'use strict';
const obj=(v,keys)=>v!==null&&typeof v==='object'&&!Array.isArray(v)&&Object.keys(v).sort().join()===keys.split(' ').sort().join();
const str=(v,max=Infinity)=>typeof v==='string'&&v.length>0&&[...v].length<=max;
const strings=v=>Array.isArray(v)&&v.every(x=>typeof x==='string');
const id=v=>str(v,128), product=v=>v===null||['hp-a','hp-b'].includes(v);
function citation(c){return obj(c,'source_id doc_id revision section excerpt url')&&id(c.source_id)&&['doc_id','revision','section','url'].every(k=>str(c[k]))&&typeof c.excerpt==='string'&&[...c.excerpt].length<=600&&c.url==='/api/v1/sources/'+c.source_id;}
function handoff(h){return obj(h,'issue confirmed_product attempted_steps missing_fields reason source_ids user_statements')&&typeof h.issue==='string'&&product(h.confirmed_product)&&['attempted_steps','missing_fields','source_ids','user_statements'].every(k=>strings(h[k]))&&h.source_ids.every(id)&&['no_evidence','conflicting_policy','refund_request','clarification_limit'].includes(h.reason);}
function turn(t){if(!obj(t,'status answer clarification citations handoff_summary release_id request_id')||!str(t.request_id)||!(t.release_id===null||str(t.release_id))||!Array.isArray(t.citations)||!t.citations.every(citation))return false;
if(t.status==='answer')return str(t.answer,4000)&&t.citations.length>0&&t.clarification===null&&t.handoff_summary===null&&str(t.release_id);
if(t.answer!==null||t.citations.length)return false;
if(t.status==='clarify')return t.handoff_summary===null&&obj(t.clarification,'question options')&&str(t.clarification.question,500)&&strings(t.clarification.options)&&t.clarification.options.length<=4&&t.clarification.options.every(x=>[...x].length<=100);
if(t.status==='handoff')return t.clarification===null&&handoff(t.handoff_summary);
return t.status==='unavailable'&&t.clarification===null&&t.handoff_summary===null;}
function source(s){return obj(s,'source_id release_id doc_id revision title section text status')&&['source_id','release_id','doc_id','revision','title','section'].every(k=>str(s[k]))&&id(s.source_id)&&['active','superseded','withdrawn'].includes(s.status)&&(s.status==='withdrawn'?s.text===null:typeof s.text==='string');}
function session(s){return obj(s,'session_id expires_at request_id')&&id(s.session_id)&&str(s.request_id)&&str(s.expires_at)&&/^\d{4}-\d\d-\d\dT/.test(s.expires_at)&&Number.isFinite(Date.parse(s.expires_at));}
function error(e){return obj(e,'error request_id')&&str(e.request_id)&&obj(e.error,'code message')&&str(e.error.code)&&str(e.error.message);}
function payload(previous,message,product_id,uuid){return previous&&previous.message===message&&previous.product_id===product_id?previous:{client_turn_id:uuid(),message,product_id};}
const api={turn,source,session,error,payload};if(typeof module!=='undefined')module.exports=api;else root.Contract=api;
})(globalThis);
