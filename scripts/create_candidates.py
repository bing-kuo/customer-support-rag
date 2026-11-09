"""Generate synthetic candidates, never approval records."""
import json
from app.knowledge import ROOT, digest

if (ROOT / 'manifest.json').exists():
    raise SystemExit('Refusing to overwrite existing corpus')
items = []
topics = [('pairing','配對','將耳機放回充電盒，再於藍牙設定移除舊配對並重新搜尋。'), ('charging','充電','請先擦乾充電接點，使用隨附充電線充電。'), ('compatibility','相容性','本展示耳機使用藍牙連線，無法保證所有裝置相容。'), ('cleaning','清潔','使用乾燥軟布清潔，不可浸泡耳機。'), ('audio','聲音','請確認裝置音量與左右聲道設定。'), ('microphone','麥克風','請檢查應用程式的麥克風權限。'), ('reset','重設','請先關閉裝置藍牙，再將耳機放回盒內。'), ('safety','安全','發現異常發熱時請停止使用並聯絡人工。')]
for product in ['hp-a', 'hp-b']:
    for slug, title, text in topics:
        items.append((f'{product}-{slug}', '1', product, title, text, '2026-01-01', None, None))
for revision, start, end in [('1','2026-01-01','2026-07-01'),('2','2026-07-01',None)]:
    items.append(('general-warranty',revision,None,'保固政策',f'合成展示政策版本 {revision}：保固資格由人工依購買證明判斷，不保證受理。',start,end,'1' if revision=='2' else None))
items.extend([('general-refund','1',None,'退款','退款須由人工審查，本系統無法承諾或執行退款。','2026-01-01',None,None),('general-contact','1',None,'人工聯絡','請複製摘要交由人工處理；展示沒有實際寄送功能。','2026-01-01',None,None)])
manifest=[]
for ident, rev, product, title, text, start, end, prior in items:
    body=f'# {title}\n\n合成展示資料，非真實商品政策。\n\n## 操作與限制\n{text}\n'
    filename=f'{ident}-v{rev}.md'
    (ROOT/'documents'/filename).write_text(body)
    manifest.append(dict(doc_id=ident,revision=rev,title=title,product_ids=[product] if product else [],locale='zh-TW',policy_scope=ident,effective_from=start,effective_to=end,status='candidate',content_sha256=digest(body.encode()),approved_by=None,approved_at=None,supersedes=prior,filename=filename))
(ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(ROOT/'approval.schema.json').write_text(json.dumps({'type':'object','additionalProperties':False,'required':['reviewer','manifest_sha256','document_hashes','approved_at'],'properties':{'reviewer':{'const':'Marcus'},'manifest_sha256':{'type':'string','pattern':'^[a-f0-9]{64}$'},'document_hashes':{'type':'object','additionalProperties':{'type':'string','pattern':'^[a-f0-9]{64}$'}},'approved_at':{'type':'string','format':'date-time'}}},indent=2)+'\n')
