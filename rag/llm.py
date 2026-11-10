from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Optional, Sequence, Tuple

from google import genai
from google.genai import types

from .exceptions import ConfigError, RAGError
from .schemas import SearchResult

RAG_SYSTEM = (
    "You answer questions strictly from the numbered context passages provided. "
    "Cite the passages you use like [1] or [2][3]. "
    "If the context does not contain the answer, say you could not find it in the documents. "
    "Never invent facts, numbers or page references. Tables are given in Markdown."
)

FIGURE_PROMPT = (
    "Describe this figure from a technical document so it can be found by search. "
    "State what it shows and any key labels, values or trends. Be factual and concise "
    "(under 120 words).\n\nCaption and nearby text:\n{text}"
)


def format_context(results: Sequence[SearchResult]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        c = r.chunk
        where = f"{c.doc_id}, page {c.page_no}"
        if c.section_title:
            where += f", section: {c.section_title}"
        blocks.append(f"[{i}] ({where}, {c.content_type})\n{c.text}")
    return "\n\n".join(blocks)


def build_rag_prompt(question: str, results: Sequence[SearchResult]) -> Tuple[str, str]:
    prompt = f"Context:\n{format_context(results)}\n\nQuestion: {question}\n\nAnswer:"
    return RAG_SYSTEM, prompt


class GeminiLLM:
    def __init__(
        self,
        api_key: Optional[str],
        model: str = "gemini-2.5-flash",
        *,
        temperature: float = 0.2,
    ) -> None:
        if not api_key:
            raise ConfigError("No Gemini API key given. Set gemini_api_key in Settings.")
        self.model = model
        self.temperature = temperature
        self._api_key = api_key
        self._client = None

    @property
    def client(self):
        if self._client is None:

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _config(self, system: Optional[str]):

        return types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
        )

    def generate(self, prompt: str, *, system: Optional[str] = None) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=prompt, config=self._config(system)
            )
        except Exception as exc:
            raise RAGError(f"Gemini request failed: {exc}") from exc
        if not response.text:
            raise RAGError("Gemini returned an empty response (it may have been blocked)")
        return response.text

    def describe_image(self, image_path: str, prompt: str) -> str:

        path = Path(image_path)
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        try:
            part = types.Part.from_bytes(data=path.read_bytes(), mime_type=mime)
            response = self.client.models.generate_content(
                model=self.model, contents=[part, prompt], config=self._config(None)
            )
        except Exception as exc:
            raise RAGError(f"Gemini image description failed: {exc}") from exc
        return (response.text or "").strip()
