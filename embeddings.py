"""Embedding provider abstraction. Local (fastembed, on-device, free) is the
default; switch to a cloud API by setting EMBEDDING_PROVIDER=voyage in .env.

IMPORTANT: providers produce different vector dimensions (local bge-small =
384, voyage-3.5-lite = 1024). Vectors from different providers/models are not
comparable and can't share one vec0 table -- switching provider later means
re-embedding every note into a fresh vector table. get_embedder() exposes
.dim and .model_id so the storage layer can detect a mismatch and refuse to
mix instead of silently corrupting search results.
"""

import os


class LocalEmbedder:
    """fastembed (ONNX runtime, ~130MB model downloaded on first use)."""

    model_id = "BAAI/bge-small-en-v1.5"
    dim = 384

    def __init__(self):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(self.model_id)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]


class VoyageEmbedder:
    """Voyage AI cloud API (Anthropic's recommended embedding partner).
    Requires `pip install voyageai` and VOYAGE_API_KEY in .env."""

    model_id = "voyage-3.5-lite"
    dim = 1024

    def __init__(self):
        import voyageai  # not in requirements until the cloud switch happens

        self._client = voyageai.Client()  # reads VOYAGE_API_KEY

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed(texts, model=self.model_id).embeddings


PROVIDERS = {
    "local": LocalEmbedder,
    "voyage": VoyageEmbedder,
}


def get_embedder():
    provider = os.environ.get("EMBEDDING_PROVIDER", "local")
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER {provider!r}; options: {sorted(PROVIDERS)}")
    return PROVIDERS[provider]()
