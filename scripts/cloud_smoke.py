"""Exactly one explicitly authorised synthetic request; no release or batch runner."""
import argparse,asyncio,json,hmac
from app.cloud import GenerationAdapter
from app.cloud_config import load_config,SetupUnavailable
from scripts.cloud_setup import check

MESSAGE='合成展示：如何清潔接點？'
EVIDENCE=[{'source_id':'synthetic-smoke','text':'請擦乾接點。'}]
REQUEST_ID='developer-smoke-one-shot-v1'
NEW_REQUEST_ID='developer-smoke-cache-compatible-v2'
NEW_MESSAGE='合成相容性檢查：請逐字回覆本輪證據中的接點處理步驟。'


async def run(*,dry_run=True,authorised=False,loader=load_config,adapter=None,new_request=False):
    # Dry run is structural only: no config/key/ledger read and no network.
    local=GenerationAdapter() if adapter is None else adapter
    message=NEW_MESSAGE if new_request else MESSAGE
    request_id=NEW_REQUEST_ID if new_request else REQUEST_ID
    _,upper,_,_,_=local.prepare(message,EVIDENCE,[])
    if upper>3072 or upper+512>3584:raise SetupUnavailable()
    if dry_run:return {'dry_run':True,'payload_within_limits':True,'network_attempted':False}
    if not authorised:raise SetupUnavailable()
    cfg=loader()
    if not all(check(lambda:cfg).values()):raise SetupUnavailable()
    def fresh_config():
        # Re-read the original source on admission and immediately before send.
        # Never persist a credential fingerprint; any in-flight change stops this run.
        fresh=loader()
        if not all(check(lambda:fresh).values()) or fresh.binding()!=cfg.binding() or not hmac.compare_digest(fresh.key,cfg.key):
            raise SetupUnavailable('approval_changed_before_send')
        return fresh
    local.config_loader=fresh_config
    try:
        _,metrics=await local.generate(message,EVIDENCE,[],request_id=request_id,scope='developer_smoke')
    except Exception as exc:
        exc.safe_cloud_status=local.safe_status()
        raise
    return {'completed':True,'input_tokens':metrics['input_tokens'],'output_tokens':metrics['output_tokens'],'total_tokens':metrics['total_tokens']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run',action='store_true');mode.add_argument('--execute-authorized-one-request',action='store_true')
    mode.add_argument('--dry-run-new-request',action='store_true');mode.add_argument('--execute-authorized-new-request',action='store_true')
    args=parser.parse_args()
    try:result=asyncio.run(run(dry_run=args.dry_run or args.dry_run_new_request,authorised=args.execute_authorized_one_request or args.execute_authorized_new_request,new_request=args.dry_run_new_request or args.execute_authorized_new_request))
    except BaseException as exc:
        result={'completed':False,'status':'stopped_no_automatic_retry'}
        diagnostic=getattr(exc,'safe_cloud_status',None)
        if isinstance(diagnostic,dict):result['accounting_diagnostic']=diagnostic
        print(json.dumps(result));return 1
    print(json.dumps(result));return 0

if __name__=='__main__':raise SystemExit(main())
