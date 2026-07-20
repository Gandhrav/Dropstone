"""Chat-model provider abstraction (bring-your-own-model). Mirrors the
embeddings.py pattern: env-var driven, Anthropic by default, swappable
without touching graph.py/digest.py.

Set DROPSTONE_MODEL in .env as "provider:model", e.g.:
    anthropic:claude-sonnet-5      (default; needs ANTHROPIC_API_KEY)
    openai:gpt-4.1-mini            (needs OPENAI_API_KEY + pip install langchain-openai)
    ollama:llama3.1                (needs local Ollama running + pip install langchain-ollama)

Anything langchain's init_chat_model() supports works (groq, google_genai,
mistralai, ...) once its langchain-<provider> package is installed.

Caveat: the router relies on structured output (with_structured_output).
Anthropic/OpenAI handle it natively via tool calling; Ollama needs a model
that supports tool calling or JSON mode (llama3.1+, qwen2.5+, etc.) --
small local models may extract fields less reliably than the default.
Rerun eval.py after ANY model switch.
"""

import os

from langchain.chat_models import init_chat_model

DEFAULT_MODEL = "anthropic:claude-sonnet-5"


def get_chat_model():
    spec = os.environ.get("DROPSTONE_MODEL", DEFAULT_MODEL)
    provider, sep, model = spec.partition(":")
    if not sep:
        # Bare model name ("gpt-4.1-mini") -- let langchain infer the provider.
        return init_chat_model(spec)
    return init_chat_model(model, model_provider=provider)
