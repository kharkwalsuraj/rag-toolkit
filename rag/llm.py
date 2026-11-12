from __future__ import annotations

import dataclasses
import logging
import mimetypes
from collections.abc import Sequence
from pathlib import Path

from google import genai
from google.genai import types

from .core import Chunk, SearchResult

log = logging.getLogger("rag.llm")

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


class GeminiLLM:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", temperature: float = 0.2) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def answer(self, question: str, sources: Sequence[SearchResult]) -> str:
        blocks = []
        for i, source in enumerate(sources, 1):
            chunk = source.chunk
            where = f"{chunk.doc_id}, page {chunk.page_no}"
            if chunk.section_title:
                where += f", section: {chunk.section_title}"
            blocks.append(f"[{i}] ({where}, {chunk.content_type})\n{chunk.text}")
        prompt = "Context:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\n\nAnswer:"
        return self._generate(prompt, system=RAG_SYSTEM)

    def describe_figures(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        described = []
        for chunk in chunks:
            if chunk.content_type == "figure" and chunk.image_path and Path(chunk.image_path).is_file():
                try:
                    description = self._generate(FIGURE_PROMPT.format(text=chunk.text), image=Path(chunk.image_path))
                    chunk = dataclasses.replace(chunk, text=f"{chunk.text}\n\nDescription: {description}")
                except RuntimeError as exc:
                    log.warning("Could not describe %s: %s", chunk.chunk_id, exc)
            described.append(chunk)
        return described

    def _generate(self, prompt: str, system: str | None = None, image: Path | None = None) -> str:
        contents = prompt
        if image is not None:
            mime_type = mimetypes.guess_type(image.name)[0] or "image/jpeg"
            contents = [types.Part.from_bytes(data=image.read_bytes(), mime_type=mime_type), prompt]
        config = types.GenerateContentConfig(system_instruction=system, temperature=self.temperature)
        try:
            response = self.client.models.generate_content(model=self.model, contents=contents, config=config)
        except Exception as exc:
            raise RuntimeError(f"Gemini request failed: {exc}") from exc
        if not response.text:
            raise RuntimeError("Gemini returned an empty response (it may have been blocked)")
        return response.text.strip()
