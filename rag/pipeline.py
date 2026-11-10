from __future__ import annotations

import dataclasses
import logging
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence

from .chunking import Chunker, ChunkerConfig
from .config import Settings
from .embedder import LocalEmbedder
from .exceptions import ConfigError, RAGError
from .llm import FIGURE_PROMPT, GeminiLLM, build_rag_prompt
from .schemas import Answer, Chunk, IngestResult, ParsedDocument, SearchResult
from .parser import MinerUClient, load_mineru_output
from .reranker import LocalReranker
from .store import QdrantStore

log = logging.getLogger("rag")


class RAGPipeline:
    """PDF in, answers out: parse, chunk, embed, store, retrieve, rerank and answer.

        rag = RAGPipeline.from_settings(Settings(mineru_token="...", gemini_api_key="..."))
        rag.ingest("manual.pdf")
        answer = rag.ask("What is the supply voltage?")
    """

    def __init__(self, settings: Settings) -> None:
        s = settings
        self.settings = s
        self.parser = MinerUClient(
            s.mineru_token,
            base_url=s.mineru_base_url,
            model=s.mineru_model,
            language=s.mineru_language,
            ocr=s.mineru_ocr,
            workdir=s.parsed_dir,
            poll_interval=s.mineru_poll_interval,
            timeout=s.mineru_timeout,
        )
        self.chunker = Chunker(
            ChunkerConfig(text_max_tokens=s.chunk_text_tokens, table_max_tokens=s.chunk_table_tokens)
        )
        self.embedder = LocalEmbedder(
            s.embedding_model,
            device=s.embedding_device,
            batch_size=s.embedding_batch_size,
            query_prefix=s.embedding_query_prefix,
        )
        self.reranker = LocalReranker(s.reranker_model, device=s.reranker_device, batch_size=s.reranker_batch_size)
        self.store = QdrantStore(
            collection=s.qdrant_collection, path=s.qdrant_path, url=s.qdrant_url, api_key=s.qdrant_api_key
        )
        # without a Gemini key everything works except ask() and describe_figures()
        self.llm = GeminiLLM(s.gemini_api_key, s.gemini_model, temperature=s.gemini_temperature) if s.gemini_api_key else None
        self._collection_ready = False

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "RAGPipeline":
        """Build the pipeline. Every Settings field has a default; pass only what you need."""
        return cls(settings or Settings())

    def parse_and_chunk(self, pdf_path: str | Path, *, force: bool = False) -> List[Chunk]:
        """Parse a PDF with MinerU (cached after the first time) and split it into chunks."""
        return self._chunk(self.parser.parse(Path(pdf_path), force=force))

    def ingest(self, pdf_path: str | Path, *, force: bool = False, describe_figures: bool = False) -> IngestResult:
        """Parse, chunk, embed and store a PDF. Ingesting the same PDF again replaces its chunks."""
        return self._ingest(self.parser.parse(Path(pdf_path), force=force), describe_figures)

    def ingest_mineru_output(self, path: str | Path, *, describe_figures: bool = False) -> IngestResult:
        """Ingest a MinerU result that is already on disk (no token needed)."""
        return self._ingest(load_mineru_output(path), describe_figures)

    def ingest_chunks(self, chunks: Sequence[Chunk]) -> IngestResult:
        """Embed and store the chunks of one document, replacing any chunks stored for it before."""
        if not chunks:
            raise RAGError("No chunks to ingest")
        self._prepare_collection()
        self.store.delete_document(chunks[0].doc_id)
        vectors = self.embedder.embed_documents([c.embedding_text for c in chunks])
        self.store.upsert(chunks, vectors)
        by_type = dict(Counter(c.content_type for c in chunks))
        log.info("Stored %d chunks %s", len(chunks), by_type)
        return IngestResult(doc_id=chunks[0].doc_id, num_chunks=len(chunks), by_type=by_type)

    def delete_document(self, doc_id: str) -> None:
        """Remove all chunks of a document."""
        self._prepare_collection()
        self.store.delete_document(doc_id)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        doc_ids: Optional[Sequence[str]] = None,
        content_types: Optional[Sequence[str]] = None,
    ) -> List[SearchResult]:
        """Dense retrieval from Qdrant (the closest chunks), then rerank them and return the best top_k.

        doc_ids limits retrieval to those documents, content_types to "text", "table", "figure" or "formula".
        """
        self._prepare_collection()
        vector = self.embedder.embed_query(query)
        candidates = self.store.dense_search(
            vector, max(self.settings.rerank_candidates, top_k), doc_ids=doc_ids, content_types=content_types
        )
        return self.reranker.rerank(query, candidates, top_k)

    def ask(
        self,
        question: str,
        top_k: int = 5,
        *,
        doc_ids: Optional[Sequence[str]] = None,
        content_types: Optional[Sequence[str]] = None,
    ) -> Answer:
        """Search, then have Gemini answer from the best chunks. Needs gemini_api_key."""
        if self.llm is None:
            raise ConfigError("ask() needs gemini_api_key in Settings")
        results = self.retrieve(question, top_k, doc_ids=doc_ids, content_types=content_types)
        if not results:
            return Answer("I could not find anything relevant in the ingested documents.", [])
        system, prompt = build_rag_prompt(question, results)
        return Answer(text=self.llm.generate(prompt, system=system), sources=results)

    def describe_figures(self, chunks: Sequence[Chunk]) -> List[Chunk]:
        """Ask Gemini to describe each figure image and add the description to its chunk."""
        if self.llm is None:
            raise ConfigError("describe_figures() needs gemini_api_key in Settings")
        described = []
        for chunk in chunks:
            if chunk.content_type == "figure" and chunk.image_path and Path(chunk.image_path).is_file():
                try:
                    text = self.llm.describe_image(chunk.image_path, FIGURE_PROMPT.format(text=chunk.text))
                    chunk = dataclasses.replace(chunk, text=f"{chunk.text}\n\nDescription: {text}")
                except RAGError as exc:
                    log.warning("Could not describe %s: %s", chunk.chunk_id, exc)
            described.append(chunk)
        return described

    def _chunk(self, parsed: ParsedDocument) -> List[Chunk]:
        chunks = self.chunker.chunk(parsed.content_list, doc_id=parsed.doc_id, asset_root=parsed.root)
        if not chunks:
            raise RAGError(f"No chunks were produced for {parsed.doc_id}")
        return chunks

    def _ingest(self, parsed: ParsedDocument, describe_figures: bool) -> IngestResult:
        chunks = self._chunk(parsed)
        if describe_figures:
            chunks = self.describe_figures(chunks)
        return self.ingest_chunks(chunks)

    def _prepare_collection(self) -> None:
        # creating the collection needs the vector size, so it waits until the embedder is first used
        if not self._collection_ready:
            self.store.ensure_collection(self.embedder.dim)
            self._collection_ready = True
