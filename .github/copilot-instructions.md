# Project guidance

## Product and data

- The product is Lumen AI, a source-grounded college helpdesk.
- The bundled AIT / University of Mumbai corpus is fictional demo data. Never present its fees, dates, contacts, policies, or events as verified facts about a real institution.
- Keep the demo-data disclosure visible in source citations and preserve provenance through retrieval and answers.
- Do not place model credentials in frontend code.

## Architecture

- Flask routes live in `backend/routes/`; business logic belongs in `backend/services/`.
- SQLite schema and queries live in `backend/database/`.
- The React and TypeScript app lives in `frontend/`; it calls same-origin `/api/v1` endpoints.
- Vite builds into `backend/static/lumen-dist/`, which Flask serves at `/`.
- Keep API responses and existing conversation/session behavior backward-compatible unless a change requires a documented migration.

## Data and retrieval

- Edit corpus records in `backend/datasets/ait_mu_dataset/`, not the generated database or vector index.
- Rebuild the corpus and vector index with `python -m backend.services.ait_demo_dataset`.
- Preserve source labels, metadata, and citation validation. Answers must be limited to retrieved evidence and should say when the corpus does not answer a question.
- Do not add unverified real college information to the fictional corpus.

## Security and reliability

- Keep secrets in `.env`; only commit `.env.example` with blank or placeholder values.
- Validate URLs, requests, and uploaded or fetched content on the backend.
- Keep network-dependent and model-dependent behavior out of unit tests; mock those boundaries.
- Avoid broad exception swallowing and never log credentials or full secret-bearing environment values.

## Common commands

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m flask --app backend.app:create_app run --host 127.0.0.1 --port 5000
Set-Location frontend; npm ci; npm run build; Set-Location ..
```
