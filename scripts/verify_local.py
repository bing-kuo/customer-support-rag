"""Produce actual test/scan evidence using installed local tools only."""
import json
import argparse
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from app.knowledge import configuration,digest
from app.lifecycle import canonical
from app.evaluation import verification_sources,verification_commands

root=Path(__file__).resolve().parents[1];out=root/'eval'/'checks';out.mkdir(exist_ok=True)
parser=argparse.ArgumentParser()
parser.add_argument('--release',required=True)
a=parser.parse_args()
if not re.fullmatch('[a-f0-9]{64}',a.release):raise SystemExit('Invalid release ID')
configuration_hash=digest(canonical(configuration()))
sources=verification_sources()
from app.knowledge import ROOT,connect
from app.lifecycle import verify
db=connect(ROOT/'runtime.sqlite3')
try:
    if verify(db,a.release)['configuration_sha256']!=configuration_hash:raise SystemExit('Release configuration mismatch')
finally:db.close()
result=subprocess.run([str(root/'.venv/bin/python'),'-m','pytest','-q','--junitxml='+str(out/'pytest-raw.xml')],cwd=root,check=False)
if result.returncode:raise SystemExit(result.returncode)
xml=ET.parse(out/'pytest-raw.xml')
for element in xml.iter():
    element.attrib.pop('hostname',None)
xml.write(out/'pytest.xml',encoding='utf-8',xml_declaration=True)
image='ghcr.io/gitleaks/gitleaks@sha256:e1b35e12a8c6fa8901f060459cfb6b2fc4c484d3afbe3b029733a3bbfab07055'
from scripts.scan_public_sources import scan,SCAN_POLICY
scan_exit=scan(root,out)
if scan_exit:raise SystemExit(scan_exit)
if configuration_hash!=digest(canonical(configuration())) or sources!=verification_sources():raise SystemExit('Configuration changed during checks')
report={'release_id':a.release,'source_fingerprints':sources,'commands':verification_commands(),'configuration_sha256':configuration_hash,'test_xml_sha256':digest((out/'pytest.xml').read_bytes()),'gitleaks_report_sha256':digest((out/'gitleaks.json').read_bytes()),'scanner_image':image,'scan_scope':SCAN_POLICY,'scan_manifest_sha256':digest((out/'scan-manifest.json').read_bytes()),'test_exit':result.returncode,'scan_exit':scan_exit}
(out/'verification.json').write_bytes(canonical(report))
print(json.dumps(report))

import shutil
evidence_hash=digest(canonical(report))
archive=out/evidence_hash
archive.mkdir(exist_ok=True)
for name in ['verification.json','pytest.xml','gitleaks.json','scan-manifest.json']:
    source=out/name
    target=archive/name
    if target.exists() and target.read_bytes()!=source.read_bytes():raise SystemExit('Immutable evidence conflict')
    shutil.copyfile(source,target)
