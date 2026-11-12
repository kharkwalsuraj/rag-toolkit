import os
from pathlib import Path

import pytest
import requests
from dotenv import load_dotenv

from rag import RAGPipeline, Settings

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"

PAPERS = {
    "olmo": "https://arxiv.org/pdf/2402.00838v4",
    "nist-ai-rmf": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
    "jupyter-notebooks": "https://escholarship.org/content/qt3hp3p179/qt3hp3p179.pdf",
}

QUESTIONS = [
    "What dataset was OLMo pretrained on?",
    "How many layers does the 7B OLMo model have?",
    "What are the four functions of the AI Risk Management Framework?",
    "What characteristics make an AI system trustworthy?",
    "Which tool helps compare versions of Jupyter notebooks?",
    "How long can a notebook cell be before it is too long?",
]


@pytest.fixture(scope="module")
def rag():
    settings = Settings(
        workspace=DATA_DIR / "workspace",
        mineru_token=os.environ.get("MINERU_TOKEN"),
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        qdrant_collection="example",
    )
    return RAGPipeline(settings)


@pytest.fixture(scope="module")
def pdfs():
    (DATA_DIR / "pdfs").mkdir(parents=True, exist_ok=True)
    paths = []
    for name, url in PAPERS.items():
        path = DATA_DIR / "pdfs" / f"{name}.pdf"
        if not path.exists():
            try:
                response = requests.get(url, timeout=120, headers={"User-Agent": "rag-toolkit-example"})
                response.raise_for_status()
            except requests.RequestException as exc:
                pytest.skip(f"could not download {url}: {exc}")
            path.write_bytes(response.content)
        paths.append(path)
    return paths


@pytest.fixture(scope="module")
def ingested(rag, pdfs):
    try:
        return rag.ingest(pdfs)
    except ValueError:
        if not os.environ.get("MINERU_TOKEN"):
            pytest.skip("set MINERU_TOKEN in .env")
        raise


def test_ingest(rag, ingested):
    for result in ingested:
        state = "already stored, skipped" if result.skipped else "parsed, chunked, embedded, stored"
        print(f"\n{result.pdf}: {result.num_chunks} chunks ({state})")
        print(f"  doc_id: {result.doc_id}")
        print(f"  MinerU output: {result.mineru_dir}")
        assert result.num_chunks > 0
    print(f"\nchunks in Qdrant: {rag.store.count()}")


def test_chunks(rag, ingested):
    chunks = rag.store.document_chunks(ingested[0].doc_id)
    print(f"\n{ingested[0].pdf}: {len(chunks)} chunks, types: {sorted({c.content_type for c in chunks})}")

    text = next(c for c in chunks if c.content_type == "text")
    print(f"\na text chunk (page {text.page_no}, section '{text.section_title}'):")
    print(text.text[:300])

    table = next((c for c in chunks if c.content_type == "table"), None)
    if table:
        print(f"\na table chunk (page {table.page_no}), kept as Markdown:")
        print(table.text[:500])
    assert chunks


def test_embedding(rag, ingested):
    chunk = rag.store.document_chunks(ingested[0].doc_id)[0]
    vector = rag.embedder.embed_documents([chunk.embedding_text])[0]
    print(f"\nembedding of the first chunk: {len(vector)} numbers, starting {[round(x, 3) for x in vector[:5]]}")
    assert len(vector) == rag.embedder.dim


def test_ingest_again(rag, pdfs, ingested):
    again = rag.ingest(pdfs)
    print("\nsecond ingest:", [(r.pdf, "skipped" if r.skipped else "redone") for r in again])
    assert all(r.skipped for r in again)


def test_ask(rag, ingested):
    if rag.llm is None:
        pytest.skip("set GEMINI_API_KEY in .env")
    for question in QUESTIONS:
        answer = rag.ask(question)
        print(f"\nQuestion: {question}\nAnswer: {answer.text}")
        for i, source in enumerate(answer.sources, 1):
            chunk = source.chunk
            print(f"  [{i}] {chunk.doc_id.rsplit('_', 1)[0]}, page {chunk.page_no}, {chunk.content_type}, rerank {source.rerank_score:.2f}")
        assert answer.text and answer.sources


def test_ask_one_document(rag, ingested):
    if rag.llm is None:
        pytest.skip("set GEMINI_API_KEY in .env")
    nist = next(r for r in ingested if r.pdf.startswith("nist"))
    answer = rag.ask("What makes a system trustworthy?", doc_ids=[nist.doc_id])
    print(f"\nonly {nist.pdf}: {answer.text}")
    assert answer.sources and all(s.chunk.doc_id == nist.doc_id for s in answer.sources)
