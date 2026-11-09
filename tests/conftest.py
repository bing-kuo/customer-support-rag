"""Isolate SYNTHETIC UNIT report fixtures from real frozen evaluation evidence.

Unit lifecycle tests use exact-support queries, never asserting that the real
frozen regression passes. Real measurements are produced only by the CLI.
"""
import json
from pathlib import Path
import pytest

@pytest.fixture(autouse=True)
def unit_evaluation_split(tmp_path,monkeypatch):
    import app.knowledge as knowledge
    import app.evaluation as evaluation
    from app.lifecycle import canonical
    original=Path(__file__).resolve().parents[1]/'eval'/'frozen-v4'
    folder=tmp_path/'unit-frozen';folder.mkdir()
    manifest=json.loads((original/'manifest.json').read_bytes())
    cases=json.loads((original/'regression.json').read_bytes())
    for q in cases:
        q['id']='unit-fixture-'+q['id']
        if q['gold']:
            topic=q['gold'][0]['doc_id'].rsplit('-',1)[-1]
            prompts={'safety':'如何處理異常發熱？','refund':'請說明退款審查政策','contact':'展示如何複製摘要','warranty':'一般保固期與資格'}
            labels={'pairing':'配對','charging':'充電','compatibility':'相容性','cleaning':'清潔','audio':'聲音','microphone':'麥克風','reset':'重設'}
            q['message']=prompts.get(topic,'如何'+labels.get(topic,topic))
            q['prior_turns']=[]
            doc=q['gold'][0]['doc_id']
            if doc.startswith('hp-a'):q['product_id']='hp-a'
            if doc.startswith('hp-b'):q['product_id']='hp-b'
    raw=canonical(cases)
    (folder/'regression.json').write_bytes(raw)
    manifest['datasets']['regression']['sha256']=knowledge.digest(raw)
    manifest['fixture']='synthetic-unit-only'
    (folder/'manifest.json').write_bytes(canonical(manifest))
    monkeypatch.setattr(knowledge,'FREEZE_MANIFEST',folder/'manifest.json')
    monkeypatch.setattr(evaluation,'FROZEN',folder)
    monkeypatch.setattr(evaluation,'EVIDENCE_ROOT',tmp_path/'unit-check-evidence')



@pytest.fixture(autouse=True)
def no_live_upstream_in_fast_tests(monkeypatch):
    """Fast tests must use explicit fake upstreams or in-process ASGI transports."""
    import httpx
    async def asynchronous(*args,**kwargs):
        raise AssertionError('Live HTTP upstream is forbidden in fast tests')
    def synchronous(*args,**kwargs):
        raise AssertionError('Live HTTP upstream is forbidden in fast tests')
    monkeypatch.setattr(httpx.AsyncHTTPTransport,'handle_async_request',asynchronous)
    monkeypatch.setattr(httpx.HTTPTransport,'handle_request',synchronous)
