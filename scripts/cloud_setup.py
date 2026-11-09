"""Read-only key-only setup status: no network, reservation or private values."""
import argparse,json
from app.cloud_config import load_config,PROJECT_ENV,PRIVATE_POLICY,SetupUnavailable,key_values


def check(loader=load_config):
    result=dict(config_valid=False,command_scope_valid=False,request_bounds_valid=False,reservation_covered=False)
    try:
        c=loader();c.validate('developer_smoke')
        result.update(config_valid=True,command_scope_valid='developer_smoke' in c.scopes,request_bounds_valid=c.max_requests>=1 and c.max_input_tokens>=3072 and c.max_output_tokens>=512 and c.max_total_tokens>=3584,reservation_covered=c.price(3072,512)<c.cap_units)
    except Exception:pass
    return result


SAFE_CODES={'private_policy_unsafe','private_policy_unavailable','private_policy_invalid','key_file_format_invalid','forbidden_provider_override','key_file_unavailable'}


def diagnose(path=PROJECT_ENV,policy_path=PRIVATE_POLICY):
    try:
        values=key_values(path)
        key=values.get('OPENAI_API_KEY','')
        if not key:return {'ready':False,'issues':[{'category':'missing_or_blank','field':'OPENAI_API_KEY'}]}
        if not 8<=len(key)<=512 or any(c.isspace() or not 33<=ord(c)<=126 for c in key):return {'ready':False,'issues':[{'category':'invalid_format','field':'OPENAI_API_KEY'}]}
        result=check(lambda:load_config(path,policy_path))
        if all(result.values()):return {'ready':True,'issues':[]}
        # Keep developer-managed policy failures separate from the user's key.
        try:load_config(path,policy_path)
        except SetupUnavailable as exc:
            code=exc.code if exc.code in SAFE_CODES else 'private_policy_review_required'
            return {'ready':False,'issues':[{'category':code}]}
        return {'ready':False,'issues':[{'category':'private_policy_review_required'}]}
    except SetupUnavailable as exc:
        return {'ready':False,'issues':[{'category':exc.code if exc.code in SAFE_CODES else 'configuration_unavailable'}]}
    except Exception:return {'ready':False,'issues':[{'category':'configuration_unavailable'}]}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--diagnose',action='store_true');args=parser.parse_args()
    result=diagnose() if args.diagnose else check();print(json.dumps(result))
    return 0 if result.get('ready',all(result.values())) else 1

if __name__=='__main__':raise SystemExit(main())
