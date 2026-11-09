"""Pure synthetic reconciliation preview. No ledger handle, network or apply path."""
from datetime import datetime
from app.cloud import GenerationAdapter
from app.cloud_config import MODEL,fingerprint,timestamp,SetupUnavailable


def preview(snapshot,cfg,evidence,review):
    """Structural review only; independent provider provenance is not verified here.

    Caller supplies an independently reviewed evidence hash, not a response's own
    'verified' flag. Even a matching manifest never authorises real mutation.
    """
    def reject():raise SetupUnavailable('reconciliation_evidence_rejected')
    try:
        fields={'project','local_request_id','provider_request_id','provider_response_id','body_sha256','configuration_sha256','approval_binding','pricing_hash','model','service_tier','status','terminal_at','usage','source'}
        if set(evidence)!=fields:reject()
        source=evidence['source']
        if set(source)!={'method','artifact_sha256'} or source['method'] not in ('provider_export','complete_response'):reject()
        if len(source['artifact_sha256'])!=64 or any(c not in '0123456789abcdef' for c in source['artifact_sha256']):reject()
        # This review manifest is an external independent input, not an evidence boolean.
        if set(review)!={'reviewer','evidence_sha256','source_artifact_sha256','configuration_sha256'} or review['reviewer'] not in ('Dana','Marcus') or review['evidence_sha256']!=fingerprint(evidence) or review['source_artifact_sha256']!=source['artifact_sha256'] or review['configuration_sha256']!=snapshot['configuration_sha256']:reject()
        if snapshot['state']!='unknown' or evidence['project']!=cfg.budget_id or snapshot['project']!=cfg.budget_id:reject()
        for key in ('local_request_id','provider_request_id','provider_response_id','body_sha256','configuration_sha256','approval_binding','pricing_hash'):
            if not snapshot[key] or snapshot[key]!=evidence[key]:reject()
        if evidence['approval_binding']!=cfg.binding() or evidence['pricing_hash']!=cfg.pricing_hash():reject()
        if evidence['model']!=MODEL or evidence['service_tier']!='default' or evidence['status'] not in ('completed','incomplete','failed','cancelled'):reject()
        terminal=timestamp(evidence['terminal_at'])
        if terminal<timestamp(snapshot['created_at']):reject()
        usage=GenerationAdapter.usage({'usage':evidence['usage']})
        # Real reconciliation needs all applicable component identities, not guesses.
        if any(v is None for v in usage.metadata().values()):reject()
        if usage.input_tokens>snapshot['reserved_input'] or usage.output_tokens>snapshot['reserved_output'] or cfg.price(usage.input_tokens,usage.output_tokens)>snapshot['reserved_cost']:reject()
        if snapshot['ceiling_fingerprint']!=fingerprint({'ceiling':cfg.cap_units}) or snapshot['spent_and_reserved']>=cfg.cap_units:reject()
        return {'dry_run':True,'structurally_eligible_for_independent_review':True,'would_mutate':False,'real_reconciliation_authorized':False,'original_state':'unknown','proposed_accounting_state':'settled','evidence_sha256':fingerprint(evidence),'schema_validated_usage':usage.metadata(),'provider_authenticity_verified':False}
    except Exception:raise SetupUnavailable('reconciliation_evidence_rejected') from None
