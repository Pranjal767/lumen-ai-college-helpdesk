"""Replace the knowledge corpus with the labeled AIT fictional demo dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from backend.database.db import PROJECT_ROOT, get_connection, get_database_path, initialize_database
from backend.database.models import insert_document_version, insert_source, upsert_document
from backend.services.chunker import chunk_document_version
from backend.services.embeddings import HASHING_EMBEDDING_MODEL, HashingEmbedder
from backend.services.vector_store import build_vector_index

DATASET_DIR = PROJECT_ROOT / "backend" / "datasets" / "ait_mu_dataset"
SOURCE_URL = "dataset://ait-mu-fictional-demo-v2"
SOURCE_NAME = "AIT / University of Mumbai fictional demo dataset"
DISCLAIMER = (
    "DEMO DATA - FICTIONAL AIT COLLEGE SCENARIO. Not official college information. "
    "All dates, fees, contacts, policies, and events are illustrative."
)
_FRONTMATTER = re.compile(r"\A---\s*\r?\n(.*?)\r?\n---\s*\r?\n?", re.DOTALL)


def ingest_ait_demo_dataset(
    dataset_dir: str | Path = DATASET_DIR,
    *,
    database_path: str | Path | None = None,
    index_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Clear old knowledge documents and index every Markdown/CSV dataset item."""
    load_dotenv(PROJECT_ROOT / ".env")
    root = Path(dataset_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"AIT dataset directory not found: {root}")
    records = _load_records(root)
    if not records:
        raise ValueError("The AIT demo dataset contains no importable records.")

    db_path = Path(database_path) if database_path else get_database_path()
    now = datetime.now(timezone.utc).isoformat()
    initialize_database(db_path)

    # Replace knowledge records only. Conversations and messages are preserved.
    with get_connection(db_path) as connection:
        connection.execute("UPDATE sources SET enabled = 0")
        connection.execute("DELETE FROM documents")
        existing_source = connection.execute(
            "SELECT id FROM sources WHERE url = ?", (SOURCE_URL,)
        ).fetchone()
        if existing_source:
            source_id = int(existing_source["id"])
            connection.execute(
                "UPDATE sources SET name = ?, source_type = 'simulated_demo', "
                "category = 'fictional-demo', enabled = 0 WHERE id = ?",
                (SOURCE_NAME, source_id),
            )
        else:
            source_id = insert_source(
                connection,
                name=SOURCE_NAME,
                url=SOURCE_URL,
                source_type="simulated_demo",
                category="fictional-demo",
                enabled=False,
                crawl_depth=1,
            )
        document_versions: list[tuple[int, str, str, str, str]] = []
        for record in records:
            body = f"{DISCLAIMER}\n\n{record['text']}"
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            metadata = {
                "dataset": "ait_mu_dataset_v2",
                "fictional": True,
                "source_type": "simulated_demo",
                "relative_path": record["path"],
            }
            source_url = f"{SOURCE_URL}/{record['id']}"
            document_id = upsert_document(
                connection,
                source_id=source_id,
                title=f"[DEMO DATA] {record['title']}",
                source_url=source_url,
                document_type="html",
                category=record["category"],
                content_hash=digest,
                file_path=str(root / record["path"]),
                fetched_at=now,
                status="fetched",
                source_type="simulated_demo",
                record_id=record["id"],
                metadata_json=json.dumps(metadata, ensure_ascii=False),
            )
            version_id = insert_document_version(
                connection,
                document_id=document_id,
                version_number=1,
                content_hash=digest,
                file_path=str(root / record["path"]),
                extracted_text=body,
                extraction_status="extracted",
                created_at=now,
            )
            document_versions.append((version_id, record["id"], record["title"], record["category"], body))

    chunks_added = 0
    for version_id, *_ in document_versions:
        chunks_added += len(chunk_document_version(version_id, database_path=db_path))

    # Match the existing local hashing index when available; this avoids model downloads.
    configured_index = Path(index_directory or "data/vector_index")
    if not configured_index.is_absolute():
        configured_index = PROJECT_ROOT / configured_index
    try:
        old_index = json.loads((configured_index / "index_metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        old_index = {}
    embedder = HashingEmbedder() if old_index.get("model_name") == HASHING_EMBEDDING_MODEL else None
    index = build_vector_index(
        database_path=db_path,
        index_directory=configured_index,
        embedder=embedder,
    )
    return {
        "dataset": str(root),
        "source_type": "simulated_demo",
        "fictional_notice": DISCLAIMER,
        "documents": len(records),
        "chunks": chunks_added,
        "vectors": index.vector_count,
        "embedding_model": index.model_name,
        "index_path": str(index.index_path),
    }


def _load_records(root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.md")):
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8-sig").strip()
        title = path.stem.replace("_", " ")
        category = path.parent.name
        match = _FRONTMATTER.match(text)
        if match:
            fields = dict(re.findall(r"(?m)^([\w-]+):\s*(.*?)\s*$", match.group(1)))
            title = fields.get("title", title)
            category = fields.get("category", category)
            text = text[match.end():].strip()
        records.append({"id": _record_id(relative), "path": relative, "title": title, "category": category, "text": text})

    for path in sorted(root.rglob("*.csv")):
        relative = path.relative_to(root).as_posix()
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            for row_number, row in enumerate(csv.DictReader(source), start=1):
                fields = [f"{key.replace('_', ' ').title()}: {value}" for key, value in row.items() if value]
                text = "\n".join(fields)
                if not text:
                    continue
                title = row.get("question") or row.get("event") or f"{path.stem} row {row_number}"
                records.append({
                    "id": _record_id(f"{relative}:{row_number}"),
                    "path": relative,
                    "title": title,
                    "category": row.get("category") or path.stem.replace("_", " "),
                    "text": text,
                })
    return records


def _record_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


if __name__ == "__main__":
    print(json.dumps(ingest_ait_demo_dataset(), ensure_ascii=False, indent=2))
