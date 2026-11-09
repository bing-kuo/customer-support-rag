import json
from typing import Annotated,Literal
from pydantic import BaseModel,ConfigDict,Field
from app.knowledge import ROOT

Hash=Annotated[str,Field(pattern=r'^[a-f0-9]{64}$')]
class CorpusApproval(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    reviewer:Literal['Marcus']
    manifest_sha256:Hash
    corpus_sha256:Hash
    document_hashes:dict[str,Hash]
    approved_at:str


def test_json_approval_schema_matches_actual_independent_record():
    schema=json.loads((ROOT/'approval.schema.json').read_text())
    record=json.loads((ROOT/'approvals/marcus-synthetic-corpus-v2-2026-11-08.json').read_text())
    CorpusApproval.model_validate(record)
    assert set(schema['required'])==set(CorpusApproval.model_fields)
    assert set(schema['properties'])==set(CorpusApproval.model_fields)
