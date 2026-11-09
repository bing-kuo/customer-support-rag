"""Runtime split: embedding only on Ollama, generation only on Responses."""
from contextvars import ContextVar
from app.ollama import Adapter as HistoricalAdapter,MODELS,DependencyError
from app.cloud import GenerationAdapter

_request=ContextVar('generation_request',default=None)
EMBEDDING_MODEL='qwen3-embedding:0.6b'

class EmbeddingAdapter(HistoricalAdapter):
    async def verify(self,client):
        response=await client.get('/api/tags');response.raise_for_status()
        tags={m['name']:m['digest'] for m in response.json()['models']}
        if tags.get(EMBEDDING_MODEL)!=MODELS[EMBEDDING_MODEL]:raise DependencyError('Embedding digest mismatch')
        return {EMBEDDING_MODEL:tags[EMBEDDING_MODEL]}
    async def generate(self,*args,**kwargs):raise DependencyError('Local generation retired')
    async def unload_owned(self,model):
        if model!=EMBEDDING_MODEL:raise DependencyError('Local generation retired')
        return await super().unload_owned(model)

class ProviderAdapter:
    def __init__(self):self.embedding=EmbeddingAdapter();self.generation=GenerationAdapter();self.generation_scope='serve'
    def bind_request(self,request_id):return _request.set((request_id,self.generation_scope))
    def reset_request(self,token):_request.reset(token)
    def preflight(self,scope):
        config=self.generation.preflight(scope);self.generation_scope=scope;return config
    async def generate(self,message,evidence,history):
        request=_request.get()
        return await self.generation.generate(message,evidence,history,request_id=request[0] if request else None,scope=request[1] if request else self.generation_scope)
    async def embed(self,texts):return await self.embedding.embed(texts)
    async def verify(self,client):return await self.embedding.verify(client)
    async def begin_owned_batch(self,**kwargs):return await self.embedding.begin_owned_batch(**kwargs)
    async def unload_owned(self,model):return await self.embedding.unload_owned(model)
    async def finish_owned_batch(self):return await self.embedding.finish_owned_batch()
    def idle_for_maintenance(self):return self.embedding.idle_for_maintenance() and self.pending==0
    @property
    def pending(self):return self.generation.pending
    @property
    def uncertain(self):return self.generation.uncertain or self.embedding.uncertain
    @property
    def open_until(self):return self.generation.open_until
    @property
    def embedding_pending(self):return self.embedding.embedding_pending
    @property
    def exclusive_batch(self):return self.embedding.exclusive_batch
    @property
    def residency_uncertain(self):return self.embedding.residency_uncertain
    @property
    def lifecycle_events(self):return self.embedding.lifecycle_events
    @property
    def lifecycle_failures(self):return self.embedding.lifecycle_failures
    @property
    def quality_counts(self):return self.generation.quality_counts
    @property
    def dependency_counts(self):return self.generation.dependency_counts
    @property
    def last_embedding_tags(self):return getattr(self.embedding,'last_embedding_tags',None)

adapter=ProviderAdapter()
