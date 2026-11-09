"""Collect actual installed distribution notices without redistributing weights."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
out=Path('eval/licenses');out.mkdir(exist_ok=True)
packages=[]
for d in sorted(metadata.distributions(),key=lambda p:p.metadata['Name'].lower()):
    if d.metadata['Name'].lower() in ['pip','setuptools']:continue
    notices=[]
    for file in d.files or []:
        if file.name.lower() not in ['license','license.txt','license.md','copying','notice','notice.txt','licenses_bundled.txt']:
            continue
        raw=d.locate_file(file).read_bytes()
        name=d.metadata['Name']+'-'+str(file).replace('/','_')
        (out/name).write_bytes(raw)
        notices.append({'file':name,'sha256':hashlib.sha256(raw).hexdigest()})
    packages.append({'name':d.metadata['Name'],'version':d.version,'license_expression':d.metadata.get('License-Expression'),'license_metadata':d.metadata.get('License'),'classifiers':[c for c in d.metadata.get_all('Classifier',[]) if c.startswith('License')],'notices':notices})
Path('eval/dependency-notices.json').write_text(json.dumps(packages,indent=2)+'\n')
print(json.dumps({'packages':len(packages),'missing_notice':[p['name'] for p in packages if not p['notices']]}))
