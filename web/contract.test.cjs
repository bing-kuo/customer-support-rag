const assert=require('node:assert/strict'), c=require('./contract.js'),f=require('./demo-fixtures.json');
let checks=0;function test(v){assert.ok(v);checks++;}
for(const k of ['answer','clarify','handoff','unavailable'])test(c.turn(f[k]));
test(c.source(f.source));test(c.error(f.error));
for(const mutate of [t=>t.role='system',t=>t.citations[0].url='https://evil.test',t=>t.answer=null,t=>t.handoff_summary={},t=>t.citations=[]]){const t=structuredClone(f.answer);mutate(t);test(!c.turn(t));}
for(const t of [null,{}, {...f.clarify,clarification:{question:'x',options:[1]}},{...f.handoff,handoff_summary:{...f.handoff.handoff_summary,reason:'execute'}},{...f.unavailable,answer:'invented'}])test(!c.turn(t));
test(c.source({...f.source,status:'withdrawn',text:null}));test(!c.source({...f.source,status:'withdrawn'}));test(c.source({...f.source,status:'superseded'}));
let n=0,uuid=()=>String(++n),p=c.payload(null,'same','hp-a',uuid);test(c.payload(p,'same','hp-a',uuid)===p);test(c.payload(p,'changed','hp-a',uuid).client_turn_id!==p.client_turn_id);test(c.payload(p,'same','hp-b',uuid).client_turn_id!==p.client_turn_id);
test(!c.session({session_id:'s',expires_at:'bad',request_id:'r'}));test(c.session({session_id:'s',expires_at:'2026-11-08T00:00:00Z',request_id:'r'}));
console.log(`${checks} contract assertions passed`);
