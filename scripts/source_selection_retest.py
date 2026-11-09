"""Freeze-only public retest plan. No execution or provider API entrypoint."""
import json
from pathlib import Path
from app.lifecycle import canonical
from app.knowledge import digest
from app.cloud_config import SetupUnavailable
from scripts.public_dev10 import BoundedPublicAdapter
from scripts.public_rejection_capture import PublicSyntheticCapture

BASE=Path(__file__).resolve().parents[1]
IDS=('dev-public-v2-07','dev-public-v2-02','dev-normal-07','dev-no_answer-00','dev-injection-00')


def plan():
    frozen=BASE/'eval'/'frozen-v4';raw=(frozen/'dev.json').read_bytes();meta=json.loads((frozen/'manifest.json').read_bytes())
    if digest(raw)!=meta['datasets']['dev']['sha256']:raise ValueError('Public frozen data changed')
    data={q['id']:q for q in json.loads(raw)}
    return {'label':'source-selection-retest-v1-known-not-blind','data_classification':'public_synthetic','dataset_sha256':digest(raw),'frozen_manifest_sha256':digest((frozen/'manifest.json').read_bytes()),'selection_protocol':'model-source-ID-selection-server-approved-block-render-v1','cases':[data[i] for i in IDS],'maximum_provider_calls':4,'expected_call_budget':{'dev-public-v2-07':2,'dev-public-v2-02':1,'dev-normal-07':0,'dev-no_answer-00':0,'dev-injection-00':1},'request_bounds':{'maximum_total_upper':4096,'output':512},'model':'gpt-6-luna','tier':'default','automatic_retries':0,'old_stage_resume':False,'gold_or_thresholds_changed':False,'capture':'Opt-in public_eval only; runtime/public-synthetic-eval-rejections, mode private; never logs or private holdout.','execution':'Not implemented or authorised by this freeze-only command. Calvin authorises a separate max-four attempt stage after review; stop on first rejected prior/unknown/technical error, no retry.'}


class RetestAdapter(BoundedPublicAdapter):
    """Future authorised public runner adapter: hard four attempts, capture opt-in."""
    def __init__(self):
        super().__init__();self.capture=PublicSyntheticCapture(plan());self.generation.rejection_capture=self.capture
    async def generate(self,message,evidence,history):
        if len(self.calls)>=4:raise SetupUnavailable('retest_request_limit')
        self.capture.select_case(self.case)
        return await super().generate(message,evidence,history)


if __name__=='__main__':
    value=plan();raw=canonical(value);out=BASE/'eval'/'source-selection-retest-v1';out.mkdir(exist_ok=True);target=out/'manifest.json'
    if target.exists() and target.read_bytes()!=raw:raise SystemExit('Frozen plan conflict')
    if not target.exists():
        with target.open('xb') as stream:stream.write(raw)
    print(json.dumps({'manifest_sha256':digest(raw),'maximum_future_provider_calls':4,'actual_provider_calls':0}))
