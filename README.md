# rag-toolkit

PDFs in, answers out.

## Lore

This repo is a major project, and ngl it is not the project I wanted to build.

The original plan was something way more ambitious. Then I remembered who grades this stuff: people who like their project topics tried, tested and ideally already in the syllabus. Pitching them something spicy was a guaranteed "interesting... but have you considered RAG?" So I considered RAG.

Since it is not a personal project, it had to be built to their spec, not mine. My dream project got left on read.

But if I am doing RAG, I am doing it right, no cap. Not a 900-line notebook called `final_final_v3`, but a small clean library my team can `uv add` and move on with their lives.

Somewhere out there is a far cooler project I never built. This is the one that exists. It has a reranker though, so lowkey still a flex.

## What we do

```
PDF → MinerU → chunks → embeddings → Qdrant → reranker → Gemini
```

| Step | What | Where |
|---|---|---|
| Parse | MinerU turns the PDF into structured blocks | cloud |
| Chunk | paragraphs merged per section, tables split by rows with the header repeated, every figure and formula kept, headers and footers dropped | local |
| Embed | `BAAI/bge-small-en-v1.5` | local |
| Store and search | Qdrant (dense vector search) | local |
| Rerank | `mixedbread-ai/mxbai-rerank-base-v1` | local |
| Answer | Gemini, with the source chunks cited | cloud |

Everything it creates is kept in one folder you choose (the workspace), and a PDF that is already stored is never parsed or embedded again.

## Requirements

| You need | For | Where |
|---|---|---|
| Python 3.10+ and [uv](https://docs.astral.sh/uv/) | installing and running | |
| MinerU API token | parsing new PDFs | [mineru.net](https://mineru.net/apiManage) |
| Gemini API key | `ask` and `describe_figures` | [Google AI Studio](https://aistudio.google.com/apikey) |
| Internet on the first run | downloading the two local models and the example PDFs | |
| About 1 GB or more of disk | PyTorch and the models | |

Gemini's free tier has per-minute and per-day limits.

## Install

```bash
uv add "rag-toolkit @ git+https://github.com/<owner>/rag-toolkit.git"
```

Pin a tag, branch or commit with `@v0.1.0` at the end of the URL. To work on the library: clone it and run `uv sync`.

## Usage

```python
from rag import RAGPipeline, Settings

rag = RAGPipeline(
    Settings(workspace="my_rag", mineru_token="...", gemini_api_key="...")
)

results = rag.ingest("manual.pdf")              # a path, a list of paths, or a folder
answer = rag.ask("What is the supply voltage?")
print(answer.text)
```

| Method | What it does |
|---|---|
| `ingest(pdfs, overwrite=False)` | parses, chunks, embeds and stores the PDFs. A PDF that is already stored is skipped, so running your program again goes straight to `ask`. `overwrite=True` redoes the PDFs from scratch and replaces their MinerU output and chunks. |
| `ask(question, top_k=5, doc_ids=None)` | finds the best chunks, reranks them and has Gemini answer from them. `doc_ids` limits the question to some documents. |

## What you get back

`ingest` returns a list with one `IngestResult` per PDF:

```python
for r in rag.ingest(["a.pdf", "b.pdf"]):
    r.pdf          # "a.pdf"
    r.doc_id       # "a_1f3c9a2b"  use it with ask(doc_ids=...)
    r.num_chunks   # chunks stored for this PDF
    r.skipped      # True when it was already stored and nothing was redone
    r.mineru_dir   # folder with this PDF's MinerU output (markdown, JSON, images)
```

`ask` returns an `Answer`:

```python
answer = rag.ask("What is the supply voltage?")
answer.text                    # Gemini's answer, citing the sources as [1], [2]
for s in answer.sources:       # the chunks the answer is based on, best first
    s.score                    # the final score (rerank score)
    s.vector_score             # the Qdrant similarity it was retrieved with
    s.rerank_score             # what the reranker gave it
    s.chunk.text               # the chunk text (tables come as Markdown)
    s.chunk.page_no            # page in the PDF
    s.chunk.section_title      # "Chapter > Section"
    s.chunk.content_type       # "text", "table", "figure" or "formula"
    s.chunk.image_path         # the figure or table image in the MinerU folder, if any
    s.chunk.doc_id             # which document it came from
```

The pieces are attributes of the pipeline (`rag.parser`, `rag.chunker`, `rag.embedder`, `rag.store`, `rag.reranker`, `rag.llm`), so you can look at any stage, for example `rag.store.document_chunks(doc_id)` for the stored chunks.

## The workspace

`Settings(workspace=...)` is the folder you own. The library writes nothing anywhere else.

```
my_rag/
  mineru/<doc_id>/    MinerU output of each PDF (markdown, JSON, images)
  qdrant/             the vector database with the chunks and embeddings
```

Chunks are not stored as files: they live in Qdrant. To remove everything, delete the folder.

Qdrant runs one of two ways:

| Setting | What it is | Use it for |
|---|---|---|
| default | embedded: runs inside your Python process and keeps its data in `<workspace>/qdrant`; no server, one process at a time | development and small setups |
| `qdrant_url` | connects to a Qdrant server (for example `docker run -p 6333:6333 qdrant/qdrant`, or Qdrant Cloud with `qdrant_api_key`) | several processes or people sharing the data |

## Configuration

`Settings` is a plain dataclass; the library reads no environment variables or files. Pass values from wherever your application keeps its configuration. Everything has a default.

| Field | Purpose | Default |
|---|---|---|
| `workspace` | your folder for MinerU output and the embedded Qdrant | `rag_workspace` |
| `mineru_token` | parse new PDFs | none |
| `mineru_language`, `mineru_ocr` | `"ch"` for Chinese, `True` for scanned PDFs | `"en"`, `False` |
| `gemini_api_key` | `ask` and `describe_figures` | none |
| `gemini_model` | Gemini model name | `gemini-2.5-flash` |
| `describe_figures` | Gemini describes figure images while ingesting | `False` |
| `embedding_model` | changing it needs a new collection | `BAAI/bge-small-en-v1.5` |
| `reranker_model` | | `mixedbread-ai/mxbai-rerank-base-v1` |
| `rerank_candidates` | chunks taken from Qdrant and rescored before cutting to `top_k` | `30` |
| `qdrant_url`, `qdrant_api_key` | use a Qdrant server | none |
| `qdrant_collection` | collection name | `documents` |
| `chunk_text_tokens`, `chunk_table_tokens` | chunk size limits | `400`, `300` |

## Run the example

`tests/test_example.py` runs the pipeline on three real documents and prints every step: the ingest results, the stored chunks, an embedding, ingesting again (skipped), and Gemini's answers to six questions with their sources, including a question limited to one document.

| File | Document | License |
|---|---|---|
| `olmo.pdf` | [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838) (Groeneveld et al., 2024) | CC BY 4.0 |
| `nist-ai-rmf.pdf` | [Artificial Intelligence Risk Management Framework (AI RMF 1.0)](https://doi.org/10.6028/NIST.AI.100-1) (NIST AI 100-1, 2023) | US government work, public domain in the US |
| `jupyter-notebooks.pdf` | [Ten simple rules for writing and sharing computational analyses in Jupyter Notebooks](https://doi.org/10.1371/journal.pcbi.1007007) (Rule et al., 2019) | CC BY 4.0 |

The PDFs are downloaded on the first run into `tests/data/pdfs/`, which is git-ignored, so they are not part of this repository.

```bash
cp .env.example .env     # fill in MINERU_TOKEN and GEMINI_API_KEY
uv run pytest tests/test_example.py
```

The first run downloads the models and the PDFs, then parses, chunks, embeds and stores them in a workspace under `tests/data/workspace`. Run it again and the documents are found in the database, so it goes straight to retrieval. The two `ask` steps are skipped without a Gemini key.

## Layout

```
rag/
  pipeline.py     RAGPipeline
  core.py         Settings, Chunk, SearchResult, Answer, IngestResult
  parser.py       MinerU client and the document id
  chunking/       chunker, tables, block readers
  embedder.py     local embeddings
  reranker.py     local reranker
  store.py        Qdrant
  llm.py          Gemini and prompts
tests/            example test
```

Notes: build `RAGPipeline` once at startup because it loads the models, and `ingest()` blocks while MinerU parses. If a program crashes in the middle of ingesting a PDF, run `ingest(pdf, overwrite=True)` for it.

## License

MIT, see [LICENSE](LICENSE).
