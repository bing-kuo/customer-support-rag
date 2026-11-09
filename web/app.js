'use strict';
const $ = s => document.querySelector(s), demo = new URLSearchParams(location.search).get('demo') === '1';
let session = null, busy = false, fixtures, opener, pending = null;
$('#demo').hidden = !demo;
function node(tag, text, parent) { const n = document.createElement(tag); n.textContent = text; parent.append(n); return n; }
function button(text, parent, action) { const b = node('button', text, parent); b.type = 'button'; b.onclick = action; return b; }
function article(text, user = false) { const a = node('article', '', $('#messages')); if(user) a.className='user'; node('small', user ? '你' : '服務助手', a); node('p', text, a); a.scrollIntoView({block:'nearest'}); return a; }
async function request(path, body) {
 const controller = new AbortController(), timer = setTimeout(()=>controller.abort(), 78000);
 try { const r = await fetch(path,{method:body?'POST':'GET',headers:body?{'Content-Type':'application/json'}:{},body:body?JSON.stringify(body):undefined,signal:controller.signal,credentials:'same-origin'}); const data=await r.json(); if(!r.ok) {if(!Contract.error(data))throw Error('invalid_response');throw new Error(data.error.code);} const valid=path.includes('/turns')?Contract.turn(data):path.includes('/sources/')?Contract.source(data):Contract.session(data);if(!valid)throw Error('invalid_response');return data; } finally {clearTimeout(timer);}
}
function errorText(e) {return ({invalid_response:'服務回應格式異常，已停止顯示內容。',idempotency_conflict:'重播內容不一致，請開始新對話。',capacity_exceeded:'服務忙碌，請稍後手動再試。',session_not_found:'對話已到期，請開始新對話。',no_active_release:'尚無可用知識，請稍後再試。',dependency_unavailable:'服務暫時無法使用。',turn_timeout:'回應未完成，請稍後再試。',session_limit:'已達對話上限，請開始新對話。',validation_error:'輸入格式有誤，請檢查內容。',turn_conflict:'上一輪仍在處理，請稍後再試。'})[e.message] || '連線或回應異常，這輪未完成。請檢查服務後手動再試。';}
async function source(c, b) {
 opener=b; $('#source-text').textContent='載入中…'; $('#source-meta').textContent=''; $('dialog').showModal();
 try { if(c.url !== '/api/v1/sources/'+c.source_id) throw Error('invalid_source');
 const scenario=$('#scenario').value;
 const s=demo ? {...(await load()).source,text:scenario==='source-withdrawn'?null:(await load()).source.text,status:scenario==='source-withdrawn'?'withdrawn':scenario==='source-superseded'?'superseded':'active'} : await request('/api/v1/sources/'+encodeURIComponent(c.source_id));
 if(!Contract.source(s)||s.source_id!==c.source_id)throw Error('invalid_response'); $('#source-title').textContent=s.title; $('#source-meta').textContent=`${s.doc_id} / revision ${s.revision} / ${s.release_id} · ${s.status}`;
 $('#source-text').textContent=s.status==='withdrawn'?'此來源已撤回，內容不再提供。':(s.status==='superseded'?'此為已取代的歷史版本。\n\n':'')+(s.text || '無可用內容。');
 } catch {$('#source-text').textContent='來源不存在或暫時無法讀取，請勿據此採取行動。';}
}
$('#close').onclick=()=>$('dialog').close(); $('dialog').addEventListener('close',()=>opener?.focus());
async function load(){return fixtures ||= await (await fetch('demo-fixtures.json')).json();}
function render(t){
 if(!Contract.turn(t))throw Error('invalid_response');
 const a=article(t.status==='answer'?t.answer:t.status==='clarify'?t.clarification.question:t.status==='handoff'?'目前需要人工確認。可複製以下摘要，尚未送交任何人。':'服務暫時無法回答，請稍後手動再試。');
 if(t.status==='clarify') for(const o of t.clarification.options) button(o,a,()=>{$('#message').value=o;$('#message').focus();});
 for(const c of t.citations || []) { const b=button(`${c.doc_id} · v${c.revision} ↗`,a,()=>source(c,b)); }
 if(t.handoff_summary){const h=t.handoff_summary; const text=`使用者陳述（未驗證）：${h.issue}\n型號：${h.confirmed_product || '未知'}\n已嘗試：${h.attempted_steps.join('、') || '無'}\n待補：${h.missing_fields.join('、') || '無'}\n原因：${h.reason}\n使用者原述：${h.user_statements.join('；')}\n來源：${h.source_ids.join('、') || '無'}`;node('p',text,a);button('複製人工摘要',a,async()=>{try{await navigator.clipboard.writeText(text);$('#status').textContent='已複製，尚未寄送。';}catch{$('#status').textContent='無法複製，請手動選取摘要。';}});}
 node('small',`request ${t.request_id}`,a);
}
$('form').onsubmit=async e=>{e.preventDefault();if(busy)return;const message=$('#message').value.trim();if(!message)return;pending=Contract.payload(pending,message,$('#product').value || null,()=>crypto.randomUUID());busy=true;$('#retry').hidden=true;$('#send').disabled=true;$('#reset').disabled=true;$('#product').disabled=true;$('#status').textContent='正在確認可用來源，請稍候…';article(message,true);
 try{let t;if(demo){const f=await load(),s=$('#scenario').value;await new Promise(r=>setTimeout(r,450));if(s==='error')throw Error(f.error.error.code);t=f[s] || f.answer;}else{session ||= (await request('/api/v1/sessions',{})).session_id;t=await request('/api/v1/sessions/'+encodeURIComponent(session)+'/turns',pending);}render(t);pending=null;$('#message').value='';$('#status').textContent='';}catch(err){$('#status').textContent=errorText(err);$('#retry').hidden=['session_not_found','session_limit','idempotency_conflict'].includes(err.message);}finally{busy=false;$('#send').disabled=false;$('#reset').disabled=false;$('#product').disabled=false;$('#message').focus();}};
$('#reset').onclick=()=>{session=null;pending=null;$('#retry').hidden=true;$('#messages').replaceChildren();article('新對話已開始。想先解決哪個問題？');$('#status').textContent='';$('#message').value='';$('#message').focus();};

$('#retry').onclick=()=>{if(!pending||busy)return;$('#message').value=pending.message;$('#product').value=pending.product_id || ''; $('form').requestSubmit();};
