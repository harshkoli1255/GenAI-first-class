# MirAI Student Policy Advisor

This project builds a beginner-friendly Retrieval-Augmented Generation (RAG) application for a university policy assistant. It is designed to answer student questions using the official `Mirai_SoT_Policy_Handbook_2026.pdf` while refusing to invent policies when the handbook does not contain enough information.

## Project architecture

- `rag_pipeline.py`: shared document ingestion, retrieval, generation, and evaluation logic.
- `backend.py`: FastAPI backend with `/ingest` and `/chat` endpoints.
- `frontend.py`: Streamlit frontend for student queries and PDF ingestion.
- `evaluate.py`: LLM-as-a-judge evaluation runner for four sample policy questions.
- `data/`: place the official handbook PDF here.
- `chroma_db/`: persistent local vector store for ChromaDB.

## Prerequisites

- Python 3.9+
- VS Code terminal
- A Google API key with access to Gemini models
- The official handbook PDF named `Mirai_SoT_Policy_Handbook_2026.pdf`

## Python virtual environment setup

From the project root:

```bash
cd /Users/kartikupadhyay/Desktop/MIRAI_C++/MiraiStudentAdvisor
python3 -m venv .venv
source .venv/bin/activate
```

On macOS in VS Code, you can also use the built-in terminal selection tools and then run:

```bash
python -m pip install --upgrade pip
```

## Install dependencies

```bash
pip install -r requirements.txt
```

If you want to install them manually, use the packages listed in `requirements.txt`.

## Configure environment variables

1. Copy `.env.example` to `.env`.
2. Add your Google API key.
3. Keep the model names as-is unless the Gemini API rejects them.

```bash
cp .env.example .env
```

Example `.env`:

```env
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_MODEL=gemini-3.8-flash
GOOGLE_EMBEDDING_MODEL=text-embedding-004
CHROMA_DB_DIR=./chroma_db
HANDBOOK_PATH=./data/Mirai_SoT_Policy_Handbook_2026.pdf
BACKEND_URL=http://127.0.0.1:8000
```

## Place the official handbook PDF

Put the official policy PDF in this path exactly:

```text
/Users/kartikupadhyay/Desktop/MIRAI_C++/MiraiStudentAdvisor/data/Mirai_SoT_Policy_Handbook_2026.pdf
```

If the PDF is missing, the backend will tell you where to place it and the evaluation script will mark tests as not executed rather than fabricating a result.

## Run the backend

From the project root:

```bash
source .venv/bin/activate
uvicorn backend:app --host 127.0.0.1 --port 8000 --reload
```

Then open the API docs at:

```text
http://127.0.0.1:8000/docs
```

## Run the Streamlit frontend

Open a second terminal and run:

```bash
cd /Users/kartikupadhyay/Desktop/MIRAI_C++/MiraiStudentAdvisor
source .venv/bin/activate
streamlit run frontend.py --server.port 8501
```

Then open:

```text
http://127.0.0.1:8501
```

## Ingest the PDF through the app

Use the frontend upload button or call the backend directly:

```bash
curl -X POST "http://127.0.0.1:8000/ingest" \
  -F "file=@/Users/kartikupadhyay/Desktop/MIRAI_C++/MiraiStudentAdvisor/data/Mirai_SoT_Policy_Handbook_2026.pdf"
```

The ingestion route validates the file, loads the PDF, chunks it, embeds the chunks, and stores them in the local ChromaDB vector store.

## Run the automated evaluation

From the project root:

```bash
source .venv/bin/activate
python evaluate.py
```

This runs four test questions and saves the results to `rag_eval_scores.csv`.

## How to interpret the CSV scores

The CSV contains these columns:

- `test_id`
- `question`
- `expected_result`
- `retrieved_context`
- `generated_answer`
- `score`
- `reasoning`
- `passed`

Score meaning:

- 5 = fully correct and grounded in handbook context
- 3 = partly correct but incomplete
- 1 = incorrect or unsupported
- 0 = not executed because the PDF or API credentials are missing

Pass values are boolean, and rows are marked as `False` when the evaluation cannot complete accurately.

## Troubleshooting

### Backend is offline

Check that the API server is running:

```bash
uvicorn backend:app --host 127.0.0.1 --port 8000 --reload
```

### Streamlit cannot reach the backend

Verify that `BACKEND_URL` in `.env` matches the running backend URL.

### Google API key errors

Make sure `.env` exists and contains a valid `GOOGLE_API_KEY`.

### The PDF is missing

Place the handbook at:

```text
./data/Mirai_SoT_Policy_Handbook_2026.pdf
```

### No context retrieved

The database may not be initialized. Upload the PDF again or re-run ingestion.

## Safely pushing to GitHub

Before pushing, ensure all sensitive files are ignored.

```bash
git status
```

The project ignores:

- `.env`
- `data/*.pdf`
- local Chroma files
- Python cache files

Do not publish API keys or personal data. If you use a public repository, keep only the template files and source code.

## Support and next steps

- Add more policy questions to `evaluate.py`.
- Improve the evaluation judge prompts.
- Add user authentication or logging if needed.
- Replace test PDF ingestion with your real handbook once the official document is available.
