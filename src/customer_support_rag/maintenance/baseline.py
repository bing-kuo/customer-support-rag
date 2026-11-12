"""Character-bigram overlap. This is a lexical baseline, not a ranker service.

Tokens are consecutive non-whitespace characters. The score is the Dice
coefficient of the two sets. Ties break by artifact key, then chunk index.
"""

from customer_support_rag.knowledge.models import KnowledgeChunk


def character_bigrams(text: str) -> set[str]:
    characters = [character for character in text if not character.isspace()]
    return {
        characters[index] + characters[index + 1]
        for index in range(len(characters) - 1)
    }


def bigram_overlap(query: str, passage: str) -> float:
    """Return the Dice overlap of two texts. An empty query scores 0."""
    left = character_bigrams(query)
    right = character_bigrams(passage)
    if not left or not right:
        return 0.0
    return (2 * len(left & right)) / (len(left) + len(right))


def baseline_order(
    query: str,
    rows: list[tuple[str, KnowledgeChunk]],
    *,
    top_k: int,
) -> list[tuple[str, KnowledgeChunk, float]]:
    """Rank ``(artifact_key, chunk)`` rows with the fixed tie break."""
    scored = [
        (key, chunk, bigram_overlap(query, chunk.text))
        for key, chunk in rows
        if chunk.text.strip()
    ]
    scored.sort(key=lambda item: (-item[2], item[0], item[1].index))
    return scored[:top_k]
