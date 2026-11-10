"""rag-toolkit: PDF -> MinerU -> chunks -> local embeddings -> Qdrant -> rerank -> Gemini answers.

    from rag import RAGPipeline, Settings

    rag = RAGPipeline.from_settings(Settings(mineru_token="...", gemini_api_key="..."))
    rag.ingest("manual.pdf")
    print(rag.ask("What is the operating temperature range?").text)
"""

from .config import Settings
from .exceptions import ConfigError, RAGError
from .schemas import Answer, Chunk, IngestResult, ParsedDocument, SearchResult
from .pipeline import RAGPipeline

__version__ = "0.1.0"

__all__ = [
    "RAGPipeline",
    "Settings",
    "Chunk",
    "SearchResult",
    "Answer",
    "IngestResult",
    "ParsedDocument",
    "RAGError",
    "ConfigError",
]
