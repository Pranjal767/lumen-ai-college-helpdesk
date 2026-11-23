FROM node:22-bookworm-slim AS frontend-build

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN npm ci --prefix ./frontend
COPY frontend ./frontend
COPY backend ./backend
RUN npm run build --prefix ./frontend

FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EMBEDDING_MODEL=synaptix-feature-hash-v3 \
    DATABASE_PATH=/app/data/lumen.db \
    VECTOR_INDEX_PATH=/app/data/vector_index

COPY requirements-render.txt ./requirements-render.txt
RUN pip install --no-cache-dir -r requirements-render.txt
COPY backend ./backend
COPY --from=frontend-build /app/backend/static/lumen-dist ./backend/static/lumen-dist
RUN python -m backend.services.ait_demo_dataset

EXPOSE 10000
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 4 --timeout 120 'backend.app:create_app()'"]
