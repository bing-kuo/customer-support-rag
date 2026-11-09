"""Offline exact-cosine comparator. Never bypasses approved online retrieval."""
import numpy as np
from app.ollama import adapter

async def rank(rows, query):
    if len(rows)>1000:
        raise ValueError('Chunk limit exceeded')
    if not rows:
        return []
    vectors=await adapter.embed([row[3] for row in rows])
    query_vector=await adapter.embed(['Instruct: Retrieve supporting customer support evidence\nQuery: '+query])
    scores=np.sum(vectors*query_vector[0],axis=1)
    if not np.isfinite(scores).all():
        raise ValueError('Invalid cosine scores')
    return [(rows[int(index)],float(scores[index])) for index in np.argsort(-scores)[:10]]
