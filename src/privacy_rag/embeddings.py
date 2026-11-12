"""Direct Ollama HTTP client pinned to the configured local loopback endpoint."""

from __future__ import annotations

import math
from typing import Any

import httpx

from privacy_rag.config import Settings


class OllamaError(RuntimeError):
    """An Ollama request failed or returned an invalid response."""


class OllamaClient:
    """Small, proxy-free client for local embeddings and structured chat output."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._http = httpx.Client(
            base_url=settings.ollama_base_url,
            timeout=httpx.Timeout(settings.ollama_timeout_seconds),
            trust_env=False,
            follow_redirects=False,
        )

    def close(self) -> None:
        self._http.close()

    def require_installed_local_models(self, model_names: tuple[str, ...]) -> None:
        """Fail closed unless each configured model tag is already installed locally."""
        if any("cloud" in name.casefold() for name in model_names):
            raise OllamaError("Cloud model tags are disabled by the privacy policy")
        try:
            response = self._http.get("/api/tags")
        except httpx.TimeoutException as exc:
            raise OllamaError(
                "Local Ollama model inventory request timed out after "
                f"{self._settings.ollama_timeout_seconds} seconds"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Local Ollama model inventory request failed ({type(exc).__name__})"
            ) from exc
        if response.status_code < 200 or response.status_code >= 300:
            raise OllamaError(f"Local Ollama returned HTTP status {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise OllamaError("Local Ollama returned invalid model inventory") from exc
        models = body.get("models") if isinstance(body, dict) else None
        installed: set[str] = set()
        if isinstance(models, list):
            for item in models:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("model")
                    if isinstance(name, str):
                        installed.add(name.casefold())
        for model_name in model_names:
            candidate = model_name.casefold()
            alternatives = {candidate, candidate if ":" in candidate else candidate + ":latest"}
            if not alternatives.intersection(installed):
                raise OllamaError(
                    f"Configured model '{model_name}' is not installed locally. "
                    f"Run `ollama pull {model_name}` or set it to a model listed by `ollama list`."
                )

    def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        """Embed text using the required nomic task prefixes."""
        if not texts:
            return []
        prefix = "search_query: " if query else "search_document: "
        inputs = [prefix + text for text in texts]
        response = self._post("/api/embed", {
            "model": self._settings.ollama_embed_model,
            "input": inputs,
            "keep_alive": self._settings.ollama_keep_alive,
        })
        vectors = response.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise OllamaError("Ollama returned an invalid embedding response")
        checked: list[list[float]] = []
        dimension: int | None = None
        for vector in vectors:
            if not isinstance(vector, list) or not vector:
                raise OllamaError("Ollama returned an invalid embedding vector")
            values = [float(value) for value in vector]
            if any(not math.isfinite(value) for value in values):
                raise OllamaError("Ollama returned a non-finite embedding value")
            if dimension is not None and len(values) != dimension:
                raise OllamaError("Ollama returned inconsistent embedding dimensions")
            dimension = len(values)
            checked.append(values)
        return checked

    def model_details(self, model: str) -> dict[str, str]:
        """Read non-content model metadata from the local Ollama daemon."""
        response = self._post("/api/show", {"model": model})
        raw_details = response.get("details")
        if not isinstance(raw_details, dict):
            raise OllamaError("Local Ollama returned invalid model details")
        return {
            key: raw_details[key]
            for key in ("format", "family", "parameter_size", "quantization_level")
            if isinstance(raw_details.get(key), str)
        }

    def chat_json(self, *, model: str, system: str, user: str,
                  schema: dict[str, Any]) -> dict[str, Any]:
        """Request one non-streaming JSON-schema-constrained local model response."""
        response = self._post("/api/chat", {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": schema,
            "options": {"temperature": self._settings.llm_temperature,
                        "num_ctx": self._settings.ollama_num_ctx,
                        "num_predict": 512},
            "keep_alive": self._settings.ollama_keep_alive,
        })
        message = response.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise OllamaError("Ollama returned an invalid chat response")
        import json

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise OllamaError("Ollama returned invalid structured output") from exc
        if not isinstance(parsed, dict):
            raise OllamaError("Ollama structured output must be a JSON object")
        return parsed

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._http.post(endpoint, json=payload)
        except httpx.TimeoutException as exc:
            raise OllamaError(
                f"Local Ollama request to {endpoint} timed out after "
                f"{self._settings.ollama_timeout_seconds} seconds"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Local Ollama request to {endpoint} failed ({type(exc).__name__})"
            ) from exc
        if response.status_code < 200 or response.status_code >= 300:
            raise OllamaError(f"Local Ollama returned HTTP status {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise OllamaError("Local Ollama returned invalid JSON") from exc
        if not isinstance(body, dict):
            raise OllamaError("Local Ollama returned an invalid response")
        return body
