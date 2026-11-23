# Lumen AI

Lumen AI is a responsive college helpdesk that uses retrieval-augmented generation (RAG) to answer questions about admissions, fees, examinations, timetables, and campus activities. The React frontend talks to a Flask API; SQLite stores conversations and indexed source records.

> **Dataset notice:** The included AIT / University of Mumbai corpus is fictional demonstration data. Fees, dates, contacts, policies, and events are illustrative and are not official information about any real college. Answers show this disclosure in their Sources panel.

## Requirements

- Python 3.11 or newer
- Node.js 22 or newer and npm
- A Gemini API key and model name for generated answers

## Setup on Windows

Run these commands from the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY`, `GEMINI_MODEL`, and a private `SECRET_KEY`. Set a private `ADMIN_API_KEY` before using protected admin endpoints. Keep `.env` private; it is ignored by Git.

Install the frontend dependencies and build the production interface:

```powershell
Set-Location frontend
npm ci
npm run build
Set-Location ..
```

The build writes to `backend/static/lumen-dist/`, which Flask serves at `/`.

## Run locally

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m flask --app backend.app:create_app run --host 127.0.0.1 --port 5000
```

Open <http://127.0.0.1:5000>. The API health check is at `/api/v1/health`.

# Lumen AI — College Helpdesk

An AI-powered college helpdesk that helps students find college-related information through intelligent retrieval, grounded responses, and source citations.

🚀 **Live Demo:** [Try Lumen AI](https://lumen-ai-college-helpdesk.onrender.com)
The first session would fail or take some time to initialize as it is hosted on free tier render services

💻 **GitHub Repository:** [View Source Code](https://github.com/Pranjal767/lumen-ai-college-helpdesk)



### Frontend development mode

Leave Flask running on port 5000. In a second terminal:

```powershell
Set-Location frontend
npm ci
npm run dev
```

Open the Vite URL shown in the terminal. Vite proxies `/api` requests to Flask.

## Dataset and search index

The fictional records are in `backend/datasets/ait_mu_dataset/`. Rebuild the SQLite corpus and local vector index after changing dataset files:

```powershell
.\.venv\Scripts\python.exe -m backend.services.ait_demo_dataset
```

This replaces indexed knowledge records and preserves conversation history. The generated SQLite database and vector index live under `data/` and are not committed; recreate them with the command above.

## Useful commands

```powershell
# Run backend tests
.\.venv\Scripts\python.exe -m pytest -q

# Rebuild the frontend
Set-Location frontend; npm run build; Set-Location ..

# Seed approved source registry entries
.\.venv\Scripts\python.exe -m backend.database.seed_sources
```

## Project layout

```text
backend/                  Flask API, database, RAG services, and demo corpus
  datasets/ait_mu_dataset Fictional admissions, fees, exams, calendar, and events
  routes/                 Health, chat, history, and admin endpoints
  services/               Retrieval, embeddings, indexing, and answer generation
frontend/                 React, TypeScript, and Vite application
data/                     Local SQLite database and generated vector index (ignored)
```

## API routes

- `GET /api/v1/health` — application health.
- `POST /api/v1/chat` — question, answer, and cited source excerpts.
- `GET /api/v1/conversations?session_id=...` — conversation list for a browser session.
- `GET /api/v1/conversations/<id>?session_id=...` — load one conversation.

The admin endpoints require the configured admin key. The backend is responsible for all model calls; API credentials must never be placed in frontend code.
