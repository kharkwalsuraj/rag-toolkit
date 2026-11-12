import hashlib
import logging
import os
import time
from pathlib import Path

import pytest
import requests
from dotenv import load_dotenv

from rag import RAGPipeline, Settings

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    force=True,
)

log = logging.getLogger("rag.tests")

DATA_DIR = Path(__file__).parent / "data"

PAPERS = {
    "olmo": "https://arxiv.org/pdf/2402.00838",
    "nist-ai-rmf": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
    "jupyter-notebooks": "https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1007007&type=printable",
}

QUESTIONS = [
    "What dataset was OLMo pretrained on?",
    "How many layers does the 7B OLMo model have?",
    "What are the four functions of the AI Risk Management Framework?",
    "What characteristics make an AI system trustworthy?",
    "Which tool helps compare versions of Jupyter notebooks?",
    "How long can a notebook cell be before it is too long?",
]


def section(title):
    log.info("=" * 70)
    log.info("%s", title)
    log.info("=" * 70)


@pytest.fixture(scope="module")
def rag():
    section("INITIALIZING RAG PIPELINE")

    settings = Settings(
        workspace=DATA_DIR / "workspace",
        mineru_token=os.environ.get("MINERU_TOKEN"),
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        qdrant_collection="example",
    )

    log.info("Workspace: %s", settings.workspace)
    log.info("Collection: %s", settings.qdrant_collection)
    log.info("MinerU token configured: %s", bool(settings.mineru_token))
    log.info("Gemini API key configured: %s", bool(settings.gemini_api_key))

    start = time.perf_counter()

    try:
        pipeline = RAGPipeline(settings)
    except Exception:
        log.exception("RAG pipeline initialization failed")
        raise

    log.info("Pipeline initialized in %.2f seconds", time.perf_counter() - start)
    log.info("Embedder dimension: %s", pipeline.embedder.dim)
    log.info("LLM available: %s", pipeline.llm is not None)
    try:
        count = pipeline.store.count()
    except ValueError:
        count = 0

    log.info("Qdrant document count: %s", count)

    return pipeline


@pytest.fixture(scope="module")
def pdfs():
    section("DOWNLOADING PDF DOCUMENTS")

    pdf_dir = DATA_DIR / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    paths = []

    for name, url in PAPERS.items():
        path = pdf_dir / f"{name}.pdf"

        log.info("[%s] Processing PDF", name)
        log.info("URL: %s", url)
        log.info("Destination: %s", path)

        if path.exists():
            size = path.stat().st_size
            log.info("PDF already exists: %.2f MB", size / 1024 / 1024)

            with path.open("rb") as file:
                signature = file.read(1024)

            if b"%PDF-" not in signature:
                log.warning("Cached file does not appear to be a valid PDF. Redownloading.")
                path.unlink()
            else:
                log.info("Cached PDF signature looks valid")
                paths.append(path)
                continue

        start = time.perf_counter()

        try:
            log.info("Sending HTTP GET request...")

            response = requests.get(
                url,
                timeout=120,
                headers={
                    "User-Agent": "Mozilla/5.0 rag-toolkit/1.0",
                    "Accept": "application/pdf,*/*",
                },
            )

            log.info("HTTP status: %s", response.status_code)
            log.info("Content-Type: %s", response.headers.get("Content-Type"))
            log.info("Content-Length: %s", response.headers.get("Content-Length"))
            log.info("Final URL: %s", response.url)

            response.raise_for_status()

            content = response.content

            log.info("Downloaded bytes: %s", len(content))
            log.info("Download time: %.2f seconds", time.perf_counter() - start)

            if b"%PDF-" not in content[:1024]:
                log.error("Response is not a valid PDF")
                log.error("First 300 bytes: %r", content[:300])
                pytest.fail(f"{name}: server did not return a valid PDF")

            path.write_bytes(content)

            checksum = hashlib.sha256(content).hexdigest()

            log.info("Saved PDF: %s", path)
            log.info("File size: %.2f MB", len(content) / 1024 / 1024)
            log.info("SHA256: %s", checksum)

        except requests.RequestException as exc:
            log.exception("Download failed for %s", name)

            if exc.response is not None:
                log.error("HTTP response: %s", exc.response.status_code)
                log.error("Response headers: %s", exc.response.headers)
                log.error("Response body: %r", exc.response.text[:500])

            pytest.skip(f"Could not download {url}: {exc}")

        paths.append(path)

    log.info("Successfully prepared %d PDFs", len(paths))

    for path in paths:
        log.info("  %s (%.2f MB)", path.name, path.stat().st_size / 1024 / 1024)

    return paths


@pytest.fixture(scope="module")
def ingested(rag, pdfs):
    section("INGESTING DOCUMENTS")

    log.info("Documents to ingest: %d", len(pdfs))

    for path in pdfs:
        log.info("  %s", path)

    start = time.perf_counter()

    try:
        results = rag.ingest(pdfs)

    except Exception:
        log.exception("Document ingestion failed")

        if not os.environ.get("MINERU_TOKEN"):
            pytest.skip("Set MINERU_TOKEN in .env")

        raise

    elapsed = time.perf_counter() - start

    log.info("Ingestion completed in %.2f seconds", elapsed)
    log.info("Documents processed: %d", len(results))

    for result in results:
        log.info("-" * 50)
        log.info("PDF: %s", result.pdf)
        log.info("Document ID: %s", result.doc_id)
        log.info("Chunks: %d", result.num_chunks)
        log.info("Skipped: %s", result.skipped)
        log.info("MinerU output: %s", result.mineru_dir)

    log.info("Total chunks in Qdrant: %d", rag.store.count())

    return results


def test_ingest(rag, ingested):
    section("TEST: INGESTION")

    assert ingested, "No documents were ingested"

    for result in ingested:
        log.info(
            "%s: %d chunks (%s)",
            result.pdf,
            result.num_chunks,
            "already stored" if result.skipped else "newly ingested",
        )

        assert result.num_chunks > 0, f"No chunks generated for {result.pdf}"

    count = rag.store.count()

    log.info("Total chunks in Qdrant: %d", count)

    assert count > 0


def test_chunks(rag, ingested):
    section("TEST: CHUNK INSPECTION")

    for result in ingested:
        log.info("Inspecting document: %s", result.doc_id)

        try:
            chunks = rag.store.document_chunks(result.doc_id)
        except Exception:
            log.exception("Failed to retrieve chunks for %s", result.doc_id)
            raise

        log.info("Total chunks: %d", len(chunks))

        assert chunks, f"No chunks found for {result.pdf}"

        content_types = {}

        for chunk in chunks:
            content_types[chunk.content_type] = (
                content_types.get(chunk.content_type, 0) + 1
            )

        log.info("Content type distribution: %s", content_types)

        for index, chunk in enumerate(chunks[:5]):
            log.info(
                "Chunk %d | page=%s | type=%s | section=%s | chars=%d",
                index,
                chunk.page_no,
                chunk.content_type,
                chunk.section_title,
                len(chunk.text),
            )
            log.info("Content preview: %s", chunk.text[:300])

        text = next(
            (chunk for chunk in chunks if chunk.content_type == "text"),
            None,
        )

        assert text is not None, f"No text chunks found in {result.pdf}"

        log.info("First text chunk:")
        log.info("Page: %s", text.page_no)
        log.info("Section: %s", text.section_title)
        log.info("%s", text.text[:500])

        table = next(
            (chunk for chunk in chunks if chunk.content_type == "table"),
            None,
        )

        if table:
            log.info("First table chunk (Markdown):")
            log.info("%s", table.text[:500])
        else:
            log.info("No table chunks found")


def test_embedding(rag, ingested):
    section("TEST: EMBEDDING")

    chunk = rag.store.document_chunks(ingested[0].doc_id)[0]

    log.info("Embedding chunk:")
    log.info("Document: %s", chunk.doc_id)
    log.info("Content type: %s", chunk.content_type)
    log.info("Text length: %d", len(chunk.embedding_text))
    log.info("Text preview: %s", chunk.embedding_text[:300])

    start = time.perf_counter()

    try:
        vector = rag.embedder.embed_documents([chunk.embedding_text])[0]
    except Exception:
        log.exception("Embedding generation failed")
        raise

    elapsed = time.perf_counter() - start

    log.info("Embedding generated in %.3f seconds", elapsed)
    log.info("Vector dimension: %d", len(vector))
    log.info("Expected dimension: %d", rag.embedder.dim)
    log.info("First five values: %s", [round(x, 5) for x in vector[:5]])
    log.info("Vector magnitude: %.5f", sum(x * x for x in vector) ** 0.5)

    assert len(vector) == rag.embedder.dim
    assert all(isinstance(x, (int, float)) for x in vector)


def test_ingest_again(rag, pdfs, ingested):
    section("TEST: DUPLICATE INGESTION")

    log.info("Running ingestion for the second time...")

    start = time.perf_counter()

    try:
        again = rag.ingest(pdfs)
    except Exception:
        log.exception("Second ingestion failed")
        raise

    log.info("Second ingestion completed in %.2f seconds", time.perf_counter() - start)

    for result in again:
        log.info(
            "%s | doc_id=%s | skipped=%s | chunks=%d",
            result.pdf,
            result.doc_id,
            result.skipped,
            result.num_chunks,
        )

    assert len(again) == len(ingested)
    assert all(result.skipped for result in again)


def test_ask(rag, ingested):
    section("TEST: QUESTION ANSWERING")

    if rag.llm is None:
        pytest.skip("Set GEMINI_API_KEY in .env")

    for index, question in enumerate(QUESTIONS, 1):
        log.info("=" * 70)
        log.info("Question %d/%d: %s", index, len(QUESTIONS), question)

        start = time.perf_counter()

        try:
            answer = rag.ask(question)
        except Exception:
            log.exception("Question answering failed: %s", question)
            raise

        elapsed = time.perf_counter() - start

        log.info("Answer generated in %.2f seconds", elapsed)
        log.info("Answer: %s", answer.text)
        log.info("Retrieved sources: %d", len(answer.sources))

        for i, source in enumerate(answer.sources, 1):
            chunk = source.chunk

            log.info(
                "[%d] document=%s | page=%s | type=%s | rerank_score=%s",
                i,
                chunk.doc_id.rsplit("_", 1)[0],
                chunk.page_no,
                chunk.content_type,
                source.rerank_score,
            )

            log.info("Source preview: %s", chunk.text[:300])

        assert answer.text, f"Empty answer for: {question}"
        assert answer.sources, f"No sources retrieved for: {question}"


def test_ask_one_document(rag, ingested):
    section("TEST: DOCUMENT-SPECIFIC RETRIEVAL")

    if rag.llm is None:
        pytest.skip("Set GEMINI_API_KEY in .env")

    nist = next(
        (result for result in ingested if result.pdf.startswith("nist")),
        None,
    )

    assert nist is not None, "NIST document was not ingested"

    question = "What makes a system trustworthy?"

    log.info("Target document: %s", nist.pdf)
    log.info("Target document ID: %s", nist.doc_id)
    log.info("Question: %s", question)

    start = time.perf_counter()

    try:
        answer = rag.ask(question, doc_ids=[nist.doc_id])
    except Exception:
        log.exception("Document-specific question answering failed")
        raise

    log.info("Answer generated in %.2f seconds", time.perf_counter() - start)
    log.info("Answer: %s", answer.text)
    log.info("Retrieved sources: %d", len(answer.sources))

    for index, source in enumerate(answer.sources, 1):
        chunk = source.chunk

        log.info(
            "[%d] doc_id=%s | page=%s | type=%s | score=%s",
            index,
            chunk.doc_id,
            chunk.page_no,
            chunk.content_type,
            source.rerank_score,
        )

        log.info("Source preview: %s", chunk.text[:300])

    assert answer.text
    assert answer.sources
    assert all(
        source.chunk.doc_id == nist.doc_id
        for source in answer.sources
    )
