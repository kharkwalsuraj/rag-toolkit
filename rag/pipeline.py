from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from .chunking import Chunker, ChunkerConfig
from .core import Answer, IngestResult, Settings
from .embedder import LocalEmbedder
from .llm import GeminiLLM
from .parser import MinerUClient, make_doc_id
from .reranker import LocalReranker
from .store import QdrantStore

log = logging.getLogger("rag")


class RAGPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        s = settings or Settings()
        if s.describe_figures and not s.gemini_api_key:
            raise ValueError("describe_figures needs gemini_api_key in Settings")
        self.settings = s
        self.parser = MinerUClient(
            s.mineru_token,
            base_url=s.mineru_base_url,
            model=s.mineru_model,
            language=s.mineru_language,
            ocr=s.mineru_ocr,
            workdir=s.mineru_dir,
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
            s.qdrant_collection, path=str(s.qdrant_dir), url=s.qdrant_url, api_key=s.qdrant_api_key
        )
        self.llm = GeminiLLM(s.gemini_api_key, s.gemini_model, s.gemini_temperature) if s.gemini_api_key else None

    def ingest(self, pdfs: str | Path | Sequence[str | Path], *, overwrite: bool = False) -> list[IngestResult]:
        """Parse, chunk, embed and store PDFs, and return one IngestResult per PDF.

        pdfs is a path, a list of paths or a folder. PDFs that are already stored are skipped;
        overwrite=True processes them again and replaces their MinerU output and chunks.
        """
        return [self._ingest_pdf(pdf, overwrite) for pdf in self._pdf_paths(pdfs)]

    def ask(self, question: str, top_k: int = 5, *, doc_ids: Sequence[str] | None = None) -> Answer:
        """Answer a question from the stored documents and return the answer with its source chunks.

        top_k is the number of chunks given to Gemini; doc_ids limits the search to those documents.
        """
        if self.llm is None:
            raise ValueError("ask needs gemini_api_key in Settings")
        candidates = self.store.dense_search(
            self.embedder.embed_query(question), max(self.settings.rerank_candidates, top_k), doc_ids
        )
        sources = self.reranker.rerank(question, candidates, top_k)
        if not sources:
            return Answer("I could not find anything relevant in the ingested documents.")
        return Answer(self.llm.answer(question, sources), sources)

    def _pdf_paths(self, pdfs: str | Path | Sequence[str | Path]) -> list[Path]:
        items = [pdfs] if isinstance(pdfs, (str, Path)) else list(pdfs)
        paths: list[Path] = []
        for item in items:
            path = Path(item)
            paths += sorted(path.glob("*.pdf")) if path.is_dir() else [path]
        if not paths:
            raise ValueError("No PDF files given")
        return paths

    def _ingest_pdf(self, pdf: Path, overwrite: bool) -> IngestResult:
        doc_id = make_doc_id(pdf)
        mineru_dir = self.settings.mineru_dir / doc_id

        if not overwrite:
            stored = self.store.count_document(doc_id)
            if stored:
                log.info("%s is already stored (%d chunks), skipping", pdf.name, stored)
                return IngestResult(pdf.name, doc_id, stored, skipped=True, mineru_dir=mineru_dir if mineru_dir.exists() else None)

        parsed = self.parser.parse(pdf, force=overwrite)
        chunks = self.chunker.chunk(parsed.content_list, doc_id=doc_id, asset_root=parsed.root)
        if not chunks:
            raise ValueError(f"No chunks were produced for {pdf.name}")
        if self.settings.describe_figures:
            chunks = self.llm.describe_figures(chunks)

        vectors = self.embedder.embed_documents([c.embedding_text for c in chunks])
        self.store.delete_document(doc_id)
        self.store.upsert(chunks, vectors)
        log.info("Stored %d chunks of %s", len(chunks), pdf.name)
        return IngestResult(pdf.name, doc_id, len(chunks), mineru_dir=mineru_dir)
