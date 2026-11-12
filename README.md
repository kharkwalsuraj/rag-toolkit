## Lore

This repo is a major project, and ngl it is not the project I wanted to build.

The original plan was something way more ambitious. Then I remembered who grades this stuff: people who like their project topics tried, tested and ideally already in the syllabus. Pitching them something spicy was a guaranteed "interesting... but have you considered RAG?" So I considered RAG.

Since it is not a personal project, it had to be built to their spec, not mine.

## Usage

```python
from rag import RAGPipeline, Settings

settings = Settings(
    workspace="rag_data",
    mineru_token="...",
    gemini_api_key="..."
)

rag = RAGPipeline(settings)

results = rag.ingest("manual.pdf")

answer = rag.ask("What is the supply voltage?")
print(answer.text)
```
