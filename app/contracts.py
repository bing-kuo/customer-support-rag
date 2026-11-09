from typing import Literal, Optional, Annotated
from pydantic import Field, model_validator
from pydantic import BaseModel, ConfigDict

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

Opaque = Annotated[str, Field(min_length=1,max_length=128,pattern=r'^[A-Za-z0-9_-]+$')]

Product = Literal['hp-a', 'hp-b']

class Citation(StrictModel):
    source_id: Opaque
    doc_id: str
    revision: str
    section: str
    excerpt: str = Field(max_length=600)
    url: str

    @model_validator(mode='after')
    def relative(self):
        if self.url != '/api/v1/sources/' + self.source_id or '/' in self.source_id or '..' in self.source_id:
            raise ValueError('Invalid source URL')
        return self

class Clarification(StrictModel):
    question: str = Field(min_length=1, max_length=500)
    options: list[Annotated[str, Field(max_length=100)]] = Field(max_length=4)

class Handoff(StrictModel):
    issue: str
    confirmed_product: Optional[Product]
    attempted_steps: list[str]
    missing_fields: list[str]
    reason: Literal['no_evidence', 'conflicting_policy', 'refund_request', 'clarification_limit']
    source_ids: list[str]
    user_statements: list[str]

class Turn(StrictModel):
    status: Literal['answer', 'clarify', 'handoff', 'unavailable']
    answer: Optional[Annotated[str, Field(min_length=1, max_length=4000)]]
    clarification: Optional[Clarification]
    citations: list[Citation]
    handoff_summary: Optional[Handoff]
    release_id: Optional[str]
    request_id: str

    @model_validator(mode='after')
    def coherent(self):
        if self.status == 'answer' and not self.release_id:
            raise ValueError('Answer requires release')
        if (self.answer is not None) != (self.status == 'answer') or bool(self.citations) != (self.status == 'answer') or (self.clarification is not None) != (self.status == 'clarify') or (self.handoff_summary is not None) != (self.status == 'handoff'):
            raise ValueError('Inconsistent turn')
        return self

class Source(StrictModel):
    source_id: Opaque
    release_id: str
    doc_id: str
    revision: str
    title: str
    section: str
    text: Optional[str]
    status: Literal['active', 'superseded', 'withdrawn']

    @model_validator(mode='after')
    def withdrawn(self):
        if self.status == 'withdrawn' and (self.text is not None or self.title != 'Withdrawn source' or self.section):
            raise ValueError('Withdrawn content must be redacted')
        return self

class ErrorDetail(StrictModel):
    code: str
    message: str

class Error(StrictModel):
    error: ErrorDetail
    request_id: str

class Session(StrictModel):
    session_id: Opaque
    expires_at: str
    request_id: str

    @model_validator(mode='after')
    def timestamp(self):
        from datetime import datetime
        value=datetime.fromisoformat(self.expires_at.replace('Z','+00:00'))
        if value.tzinfo is None:
            raise ValueError('Timezone required')
        return self

class Components(StrictModel):
    storage: Literal['ok', 'unavailable']
    embedding: Literal['ok', 'unavailable']
    generation: Literal['ok', 'unavailable']

class Health(StrictModel):
    status: Literal['ok', 'degraded']
    components: Components
    request_id: str
