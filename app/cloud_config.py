"""One explicit project-local secret source, loaded only on generation admission."""
import hashlib
import json
from dataclasses import dataclass,field
from datetime import datetime,timezone
from decimal import Decimal,InvalidOperation,ROUND_CEILING,ROUND_FLOOR
from pathlib import Path
import re
from app.ollama import DependencyError

PROJECT_ENV=Path(__file__).resolve().parents[1]/'.env'
MODEL='gpt-6-luna'
ENDPOINT='https://api.openai.com/v1/responses'
PRICING_REVISION='gpt-6-luna-standard-2026-11-08-reviewed'
# Dana-reviewed official standard revision; not a live pricing/account probe.
# Overrides must never undercut these revision floors. Extra fees need explicit approval.
STANDARD_INPUT='0.10'
STANDARD_OUTPUT='0.50'
# Reserve all input at uncached plus longest documented cache-write rate,
# with documented regional uplift, even if the actual request receives no discount.
RESERVATION_INPUT=str((Decimal(STANDARD_INPUT)+Decimal('0.25'))*Decimal('1.10'))
RESERVATION_OUTPUT=str(Decimal(STANDARD_OUTPUT)*Decimal('1.10'))
# Cover every documented short-text Luna tier, including Fast, while requesting default.
PRICE_BOUND_REVISION='luna-short-text-all-documented-tiers-v1'
TIER_BOUND_MULTIPLIER=2
UNIT=10**12
SCOPES={'developer_smoke','public_eval','holdout','http_ui','serve'}
POLICY={'configuration':'key-only-private-policy-v1','service_tier':'default','provider':'openai','model':MODEL,'endpoint':ENDPOINT,'reasoning_effort':'none','store':False,'background':False,'stream':False,'max_output_tokens':512,'max_total_token_upper_bound':4096,'budget_policy':'durable-cumulative-strict-v3','price_bound_revision':PRICE_BOUND_REVISION,'pricing_revision':PRICING_REVISION,'identity':'mutable provider alias, not a local digest','automatic_retries':0}

class SetupUnavailable(DependencyError):
    def __init__(self,code='cloud_setup_unavailable'):
        super().__init__('Cloud generation unavailable');self.code=code


def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def now():return datetime.now(timezone.utc)

def timestamp(value):
    try:
        parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
        if parsed.tzinfo is None:raise ValueError()
        return parsed
    except (ValueError,AttributeError):raise SetupUnavailable() from None


def money_units(value,*,cap=False):
    try:
        number=Decimal(value)
        if not number.is_finite() or number<0 or number>Decimal(2**60)/UNIT:raise ValueError()
        units=int((number*UNIT).to_integral_value(rounding=ROUND_FLOOR if cap else ROUND_CEILING))
        if units>2**60:raise ValueError()
        return units
    except (InvalidOperation,ValueError,TypeError,OverflowError):raise SetupUnavailable() from None

@dataclass(frozen=True)
class CloudConfig:
    key:str=field(repr=False)
    budget_id:str
    scopes:tuple
    expires_at:datetime
    pricing_expires_at:datetime
    max_requests:int
    max_input_tokens:int
    max_output_tokens:int
    max_total_tokens:int
    cap_units:int=field(repr=False)
    input_rate_units:int=field(repr=False)
    output_rate_units:int=field(repr=False)
    extra_units:int=field(repr=False)
    pricing_revision:str=PRICING_REVISION
    def validate(self,scope,at=None):
        at=at or now()
        if not isinstance(self.key,str) or not 8<=len(self.key)<=512 or any(c.isspace() or ord(c)<33 or ord(c)>126 for c in self.key):raise SetupUnavailable()
        if not self.key or scope not in self.scopes or scope not in SCOPES or at>=self.expires_at or at>=self.pricing_expires_at or self.pricing_revision!=PRICING_REVISION:raise SetupUnavailable()
        for n in [self.max_requests,self.max_input_tokens,self.max_output_tokens,self.max_total_tokens,self.cap_units,self.input_rate_units,self.output_rate_units]:
            if type(n) is not int or not 0<n<=2**60:raise SetupUnavailable()
        if type(self.extra_units) is not int or not 0<=self.extra_units<=2**60:raise SetupUnavailable()
        if self.key in self.budget_id or self.key in MODEL or any(self.key in scope for scope in self.scopes):raise SetupUnavailable()
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',self.budget_id):raise SetupUnavailable()
    def price(self,input_tokens,output_tokens):
        return TIER_BOUND_MULTIPLIER*((input_tokens*self.input_rate_units+output_tokens*self.output_rate_units+999999)//1000000+self.extra_units)
    def binding(self):
        # Private ledger binding only; do not publish amounts or a key hash.
        return fingerprint({k:str(v) if isinstance(v,datetime) else v for k,v in vars(self).items() if k!='key'})
    def pricing_hash(self):
        return fingerprint({'bound_revision':PRICE_BOUND_REVISION,'revision':self.pricing_revision,'input':self.input_rate_units,'output':self.output_rate_units,'extra':self.extra_units,'expires':str(self.pricing_expires_at),'model':MODEL})


def config_from_mapping(values):
    try:
        if 'OPENAI_BASE_URL' in values or 'OPENAI_MODEL' in values:raise SetupUnavailable()
        if any(values.get(k)!='true' for k in ['CLOUD_GENERATION_APPROVED','CLOUD_PRICING_APPROVED','CLOUD_SYNTHETIC_ONLY']):raise SetupUnavailable()
        if values.get('CLOUD_BILLING_COMPONENTS')!='input,output':raise SetupUnavailable()
        def positive(key):
            value=values[key]
            if not re.fullmatch(r'[0-9]{1,15}',value):raise SetupUnavailable()
            return int(value)
        config=CloudConfig(key=values['OPENAI_API_KEY'],budget_id=values['CLOUD_BUDGET_ID'],scopes=tuple(values['CLOUD_SCOPE'].split(',')),expires_at=timestamp(values['CLOUD_EXPIRES_AT']),pricing_expires_at=timestamp(values['CLOUD_PRICING_VALID_UNTIL']),max_requests=positive('CLOUD_MAX_REQUESTS'),max_input_tokens=positive('CLOUD_MAX_INPUT_TOKENS'),max_output_tokens=positive('CLOUD_MAX_OUTPUT_TOKENS'),max_total_tokens=positive('CLOUD_MAX_TOTAL_TOKENS'),cap_units=money_units(values['CLOUD_CAP_USD'],cap=True),input_rate_units=money_units(values.get('CLOUD_INPUT_USD_PER_MILLION') or STANDARD_INPUT),output_rate_units=money_units(values.get('CLOUD_OUTPUT_USD_PER_MILLION') or STANDARD_OUTPUT),extra_units=money_units(values['CLOUD_MAX_EXTRA_USD_PER_REQUEST']),pricing_revision=values['CLOUD_PRICING_REVISION'])
        if config.input_rate_units<money_units(STANDARD_INPUT) or config.output_rate_units<money_units(STANDARD_OUTPUT):raise SetupUnavailable()
        if any(scope not in SCOPES for scope in config.scopes):raise SetupUnavailable()
        return config
    except (KeyError,TypeError,ValueError):raise SetupUnavailable() from None


PRIVATE_POLICY=PROJECT_ENV.parent/'runtime'/'cloud-policy.json'


def read_private_json(path):
    import os,stat
    try:
        fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
        try:
            meta=os.fstat(fd)
            if not stat.S_ISREG(meta.st_mode) or meta.st_size>16384 or meta.st_mode & 0o077:raise SetupUnavailable('private_policy_unsafe')
            with os.fdopen(fd,'r',closefd=False) as f:return json.load(f)
        finally:os.close(fd)
    except Exception:raise SetupUnavailable('private_policy_unavailable') from None


def key_values(path):
    import os,stat
    try:
        fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
        try:
            meta=os.fstat(fd)
            if not stat.S_ISREG(meta.st_mode) or meta.st_size>16384:raise SetupUnavailable()
            with os.fdopen(fd,'r',closefd=False) as f:lines=f.read().splitlines()
        finally:os.close(fd)
        values={}
        for line in lines:
            line=line.strip()
            if not line or line.startswith('#'):continue
            key,sep,value=line.partition('=')
            if not sep or key in values or not re.fullmatch('[A-Z_]+',key):raise SetupUnavailable('key_file_format_invalid')
            values[key]=value.strip()
        if any(k in values for k in ('OPENAI_MODEL','OPENAI_BASE_URL')):raise SetupUnavailable('forbidden_provider_override')
        return values
    except SetupUnavailable:raise
    except Exception:raise SetupUnavailable('key_file_unavailable') from None


def load_config(path=PROJECT_ENV,policy_path=PRIVATE_POLICY):
    # Only key comes from .env. Legacy CLOUD_* values are ignored, never written.
    values=key_values(path);policy=read_private_json(policy_path)
    try:
        expected={'version','model','budget_id','cap_units','input_rate_units','output_rate_units','extra_units','pricing_revision','pricing_expires_at','enabled_scopes','enabled'}
        if set(policy)!=expected or policy['version']!=1 or policy['model']!=MODEL or policy['enabled'] is not True or policy['pricing_revision']!=PRICING_REVISION:raise SetupUnavailable('private_policy_invalid')
        if policy['input_rate_units']<money_units(RESERVATION_INPUT) or policy['output_rate_units']<money_units(RESERVATION_OUTPUT):raise SetupUnavailable('private_policy_invalid')
        scopes=tuple(policy['enabled_scopes'])
        if not scopes or any(s not in SCOPES for s in scopes):raise SetupUnavailable('private_policy_invalid')
        cfg=CloudConfig(key=values.get('OPENAI_API_KEY',''),budget_id=policy['budget_id'],scopes=scopes,expires_at=datetime.max.replace(tzinfo=timezone.utc),pricing_expires_at=timestamp(policy['pricing_expires_at']),max_requests=200,max_input_tokens=200*3584,max_output_tokens=200*512,max_total_tokens=200*4096,cap_units=policy['cap_units'],input_rate_units=policy['input_rate_units'],output_rate_units=policy['output_rate_units'],extra_units=policy['extra_units'])
        cfg.validate(scopes[0]);return cfg
    except SetupUnavailable:raise
    except Exception:raise SetupUnavailable('private_policy_invalid') from None
