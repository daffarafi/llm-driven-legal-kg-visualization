# LLM-Driven Legal Knowledge Graph Visualization

Sistem end-to-end untuk membangun dan memvisualisasikan Knowledge Graph hukum Indonesia menggunakan LLM. Dari ekstraksi dokumen PDF peraturan perundang-undangan hingga antarmuka web interaktif.

## Arsitektur Sistem

```
┌──────────────────────────────────────────────────────────────────┐
│                       SISTEM KESELURUHAN                         │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  TASK 1      │    │  TASK 2 & 3  │    │     TASK 4       │   │
│  │  pipeline/   │    │  finetuning/ │    │     frontend/    │   │
│  │  Ekstraksi   │    │  Fine-tuning │    │     backend/     │   │
│  │  Dokumen→KG  │    │  LLM         │    │     Website      │   │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────────┘   │
│         ▼                   ▼                   ▼               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Neo4j Graph Database                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

## Struktur Folder

```
├── pipeline/                # Task 1: ETL Pipeline (PDF → KG)
│   ├── config.yaml
│   ├── extract/             # PDF extraction, structure parsing, chunking
│   ├── transform/           # LLM extraction, validation, deduplication, embedding
│   └── load/                # Neo4j ingestion
├── data/                    # Data stages (raw, extracted, parsed, chunks, triples, etc.)
├── finetuning/              # Task 2 & 3: LLM Fine-tuning
│   ├── query_model/         # Task 2: NL → Cypher query generation
│   └── response_model/      # Task 3: KG results → NL answer generation
├── notebooks/               # Jupyter notebooks (01-06)
│   ├── 01_pdf_extraction.ipynb
│   ├── 02_llm_extraction.ipynb
│   ├── 03_neo4j_ingestion.ipynb
│   ├── 04_full_pipeline.ipynb
│   ├── 05a_qa_generator.ipynb    # Task 2: Query model training data
│   ├── 05b_inference.ipynb       # Task 2: Query model evaluation
│   ├── 06a_qa_generator.ipynb    # Task 3: Response model training data
│   └── 06b_inference.ipynb       # Task 3: Response model evaluation
├── backend/                 # Task 4: FastAPI REST API
│   ├── app/
│   │   ├── config.py
│   │   ├── main.py
│   │   ├── models/          # Pydantic schemas
│   │   ├── routers/         # API routes (graph, search, QA, stats, documents)
│   │   └── services/        # Neo4j driver, LLM service
│   └── requirements.txt
├── frontend/                # Task 4: Next.js Interactive UI
│   ├── src/
│   │   ├── app/             # Pages (Home, Explorer, QA, Analytics, Documents)
│   │   ├── components/ui/   # shadcn/ui components
│   │   └── lib/             # API client, types, utilities
│   ├── package.json
│   └── tsconfig.json
├── .env.example
└── .gitignore
```

## Features

### Task 1: Knowledge Graph Construction Pipeline
- PDF text extraction (PyMuPDF + PaddleOCR fallback)
- Legal document structure parsing (BAB → Bagian → Pasal → Ayat)
- Chunking (400-800 tokens, 100 overlap)
- LLM-based triple extraction (Gemini)
- Validation, deduplication, embedding generation
- Neo4j ingestion with vector + full-text indexes

### Task 2: Query Model (NL → Cypher)
- Template-based + LLM-assisted training data generation
- SFT fine-tuning with LoRA/QLoRA
- Evaluation: Cypher validity, execution accuracy

### Task 3: Response Model (KG Results → NL Answer)
- Training data from KG query results
- SFT fine-tuning for Indonesian legal domain
- Evaluation: factual accuracy, fluency

### Task 4: Interactive Web Application
- **KG Explorer**: Force-directed graph visualization, node highlighting, detail panel
- **QA System**: Hybrid RAG pipeline with live side-panel graph
- **Analytics Dashboard**: Node/relation statistics
- **Document Viewer**: Browse UU documents

## Prerequisites

- **Python** ≥ 3.10
- **Node.js** ≥ 18
- **Neo4j** database on `bolt://localhost:7687`
- **Google Gemini API key**

## Setup

```bash
# 1. Environment
cp .env.example .env
# Edit .env with your credentials

# 2. Pipeline (Task 1)
cd pipeline
pip install -r requirements.txt
python run_pipeline.py --config config.yaml

# 3. Backend (Task 4)     
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# 4. Frontend (Task 4)
cd frontend
npm install
npm run dev
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| PDF Extraction | PyMuPDF, PaddleOCR |
| KG Construction | Gemini LLM, Neo4j |
| Fine-tuning | LoRA/QLoRA via Unsloth/HuggingFace PEFT |
| Embedding | text-embedding-004 (Google) |
| Backend | FastAPI, Uvicorn |
| Frontend | Next.js 16, React 19, TailwindCSS v4 |
| Graph Viz | react-force-graph-2d |
| UI Kit | shadcn/ui, Lucide icons |

## License

Tugas Akhir — Universitas Indonesia, 2026.
