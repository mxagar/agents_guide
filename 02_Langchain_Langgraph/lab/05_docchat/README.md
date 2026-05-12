# DocChat: Multi-Agent RAG for Document Question Answering

Original repository: [docchat](https://github.com/ibm-developer-skills-network/zzpwx-docchat). In this course repo it is included as the [`docchat`](./docchat) Git submodule.

The lab in [`Instructions.pdf`](./Instructions.pdf) builds DocChat, a document question-answering app for long PDFs, Word files, text files, and Markdown files. The core idea is to make RAG more reliable by splitting the workflow into retrieval, relevance checking, answer generation, and verification instead of asking a single model to answer directly.

## What the Project Does

DocChat lets a user upload one or more documents, ask a question, and receive an answer plus a verification report. It is designed for dense reports with tables, figures, footnotes, and long sections where general chatbots often miss the right passage or invent unsupported claims.

The application combines:

- **Docling document processing** in [`document_processor/file_handler.py`](./docchat/document_processor/file_handler.py), which converts supported files to Markdown, chunks them by headings, deduplicates chunks, and caches processed documents.
- **Hybrid retrieval** in [`retriever/builder.py`](./docchat/retriever/builder.py), which combines BM25 keyword search with Chroma vector search through LangChain's `EnsembleRetriever`.
- **Relevance checking** in [`agents/relevance_checker.py`](./docchat/agents/relevance_checker.py), which classifies a query as `CAN_ANSWER`, `PARTIAL`, or `NO_MATCH` before the expensive answer-generation path runs.
- **Research and verification agents** in [`agents/research_agent.py`](./docchat/agents/research_agent.py) and [`agents/verification_agent.py`](./docchat/agents/verification_agent.py), which draft an answer from retrieved context and then check whether the answer is supported, relevant, and free of contradictions.
- **LangGraph orchestration** in [`agents/workflow.py`](./docchat/agents/workflow.py), where the graph starts with relevance checking, routes relevant questions to research, routes research to verification, and loops back to research when verification fails.
- **Gradio UI** in [`app.py`](./docchat/app.py), which provides document upload, example selection, question input, and answer/verification outputs.

## Workflow

The implemented graph is:

```text
question + retriever
        |
        v
check_relevance -- NO_MATCH --> END
        |
        v
research
        |
        v
verify -- unsupported/irrelevant --> research
        |
        v
       END
```

The lab PDF emphasizes this as a verification-driven RAG pattern: retrieve relevant passages, generate a draft, verify that the draft is grounded in the source documents, and retry when unsupported claims or contradictions are detected.

## Running the Lab

From the repository root, initialize the submodule if needed:

```bash
git submodule update --init --recursive
```

Then use the root `agents` environment:

```bash
conda activate agents
cd 02_Langchain_Langgraph/lab/05_docchat/docchat
python app.py
```

The app launches a Gradio interface on port `5000` in the cloned project. The upstream lab also includes example documents under [`docchat/examples`](./docchat/examples).
