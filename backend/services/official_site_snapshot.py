"""Crawl the college sitemap, index its sources, and export a PDF archive."""

from __future__ import annotations

import html
import json
import os
import re
import unicodedata
import calendar
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
from xml.etree import ElementTree

import pymupdf
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from backend.database.db import PROJECT_ROOT, get_connection, get_database_path
from backend.database.models import (
    get_latest_document_version,
    get_source,
    list_document_version_chunks,
)
from backend.database.seed_sources import seed_sources
from backend.services.chunker import chunk_document_version
from backend.services.crawler import CrawlError, FetchedResource, SafeCrawler, normalize_allowed_url
from backend.services.document_fetcher import fetch_and_store_document
from backend.services.vector_store import build_vector_index

SITEMAP_URL = "https://bkbck.edu.in/sitemap_index.xml"
SITEMAP_NAMES = {
    "page-sitemap.xml",
    "post-sitemap.xml",
    "courses-sitemap.xml",
    "category-sitemap.xml",
    "course-category-sitemap.xml",
}
MAX_SITEMAP_URLS = 1200
MAX_PAGE_BYTES = 10 * 1024 * 1024
LARGE_PDF_BYTES = 40 * 1024 * 1024
OVERSIZED_PDF_URLS = (
    "https://bkbck.edu.in/wp-content/uploads/2026/03/AICTE-Broucher.pdf",
    "https://bkbck.edu.in/wp-content/uploads/2026/06/2025-2026-IQAC-Minutes.pdf",
    "https://bkbck.edu.in/wp-content/uploads/2026/03/IQAC-Meeting-of-Minutes-2024-25.pdf",
)
PDF_OUTPUT = PROJECT_ROOT / "output" / "pdf" / "Lumen_AI_BK_Birla_College_Knowledge_Base.pdf"
DOCUMENT_SUFFIXES = {".pdf"}
CURRENT_STUDENT_PAGES = {
    "https://bkbck.edu.in/notices/",
    "https://bkbck.edu.in/category/notices/",
    "https://bkbck.edu.in/examination/circular-notifications/",
    "https://bkbck.edu.in/examination/exam-timetable/",
}
STUDENT_TERM_PATTERNS = (
    r"\bnotices?\b", r"\bcirculars?\b", r"\bexam(?:ination)?s?\b",
    r"\btimetables?\b", r"\bresults?\b", r"\bfees?\b", r"\badmissions?\b",
    r"\bscholarships?\b", r"\bacademics?\b", r"\bsemesters?\b",
    r"\bcommencement\b", r"\bregistration\b",
)


def sync_official_site() -> dict[str, object]:
    """Fetch all public pages in the college sitemap and their linked PDFs."""
    load_dotenv(PROJECT_ROOT / ".env")
    seed_sources()
    database_path = get_database_path()
    domain = os.environ.get("COLLEGE_DOMAIN", "bkbck.edu.in").strip()
    timeout = float(os.environ.get("CRAWL_TIMEOUT", "15"))
    delay = float(os.environ.get("CRAWL_DELAY_SECONDS", "1"))
    max_bytes = int(os.environ.get("MAX_DOCUMENT_BYTES", str(MAX_PAGE_BYTES)))
    crawler = SafeCrawler(
        domain,
        timeout=timeout,
        delay_seconds=delay,
        max_response_bytes=max_bytes,
    )

    page_urls, sitemap_pdfs = _sitemap_urls(crawler)
    if len(page_urls) + len(sitemap_pdfs) > MAX_SITEMAP_URLS:
        raise ValueError(
            f"The approved sitemap contains more than {MAX_SITEMAP_URLS} URLs; "
            "increase MAX_SITEMAP_URLS after reviewing the sitemap inventory."
        )

    source_ids = _source_ids(database_path)
    pending_pdfs: dict[str, int] = {
        url: _source_id_for_url(url, source_ids) for url in sitemap_pdfs
    }
    errors: list[dict[str, str]] = []
    counts: Counter[str] = Counter()

    for position, url in enumerate(page_urls, start=1):
        try:
            resource = _cached_resource(url, database_path) or crawler.fetch_resource(
                url, max_bytes=max_bytes
            )
            if "text/html" not in resource.content_type:
                if "pdf" in resource.content_type or _is_pdf(url):
                    pending_pdfs[url] = _source_id_for_url(url, source_ids)
                else:
                    counts["non_html_skipped"] += 1
                continue

            source_id = _source_id_for_url(url, source_ids)
            stored = fetch_and_store_document(
                source_id,
                url,
                crawler,
                category=_category_for_url(url),
                database_path=database_path,
                max_bytes=max_bytes,
                resource=resource,
                force_reextract=True,
            )
            counts["html_fetched"] += 1
            counts["unchanged"] += stored.status == "unchanged"
            counts["text_documents"] += stored.extraction_status == "extracted"
            counts["scanned_documents"] += stored.extraction_status == "no_text"
            counts["chunks_added"] += _ensure_chunks(stored.id, database_path)

            soup = BeautifulSoup(resource.content, "html.parser")
            for anchor in soup.find_all("a", href=True):
                candidate = urljoin(resource.url, anchor["href"])
                try:
                    candidate = normalize_allowed_url(candidate, domain)
                except (ValueError, CrawlError):
                    continue
                if _is_pdf(candidate):
                    pending_pdfs.setdefault(candidate, source_id)
        except Exception as error:
            errors.append({"url": url, "message": str(error)[:400]})

        if position % 25 == 0 or position == len(page_urls):
            print(
                f"Pages {position}/{len(page_urls)}; PDFs queued {len(pending_pdfs)}; "
                f"documents {counts['html_fetched']}", flush=True
            )

    for position, (url, source_id) in enumerate(pending_pdfs.items(), start=1):
        try:
            resource = _cached_resource(url, database_path)
            stored = fetch_and_store_document(
                source_id,
                url,
                crawler,
                category=_category_for_url(url),
                database_path=database_path,
                max_bytes=max_bytes,
                resource=resource,
            )
            counts["pdf_fetched"] += 1
            counts["unchanged"] += stored.status == "unchanged"
            counts["text_documents"] += stored.extraction_status == "extracted"
            counts["scanned_documents"] += stored.extraction_status == "no_text"
            counts["chunks_added"] += _ensure_chunks(stored.id, database_path)
        except Exception as error:
            errors.append({"url": url, "message": str(error)[:400]})

        if position % 25 == 0 or position == len(pending_pdfs):
            print(
                f"Attachments {position}/{len(pending_pdfs)}; "
                f"PDFs indexed {counts['pdf_fetched']}", flush=True
            )

    selected_ids, cutoff = _current_student_document_ids(database_path)
    index = build_vector_index(
        database_path=database_path,
        document_ids=selected_ids,
        chunk_filter=_current_student_chunk,
    )
    report: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "student_data_cutoff": cutoff.isoformat(),
        "selected_documents": len(selected_ids),
        "selection_policy": "Student notices and academic, fee, and exam sources published or uploaded in the last 60 days, plus current notice and timetable landing pages.",
        "sitemap_pages": len(page_urls),
        "linked_pdfs": len(pending_pdfs),
        "counts": dict(counts),
        "index_vectors": index.vector_count,
        "index_model": index.model_name,
        "errors": errors,
    }
    report_path = PROJECT_ROOT / "data" / "processed" / "official_site_snapshot.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Sync complete:", json.dumps({key: value for key, value in report.items() if key != "errors"}))
    print("Fetch errors:", len(errors))
    return report


def _cached_resource(url: str, database_path: Path) -> FetchedResource | None:
    """Reuse a previously downloaded same-URL file when available."""
    with get_connection(database_path) as connection:
        row = connection.execute(
            "SELECT file_path, document_type FROM documents WHERE source_url = ?",
            (url,),
        ).fetchone()
    if row is None:
        return None
    path = Path(row["file_path"])
    if not path.is_file() or path.stat().st_size > MAX_PAGE_BYTES:
        return None
    content = path.read_bytes()
    content_type = "application/pdf" if row["document_type"] == "pdf" else "text/html"
    return FetchedResource(url=url, content_type=content_type, content=content)


def _current_student_document_ids(database_path: Path) -> tuple[set[int], date]:
    """Select recent student-facing records and the current notices/timetable pages."""
    today = datetime.now(timezone.utc).date()
    cutoff = today - timedelta(days=60)
    with get_connection(database_path) as connection:
        rows = connection.execute(
            """
            SELECT d.id, d.title, d.source_url, d.document_type, d.category,
                   d.file_path, d.source_type, d.metadata_json
            FROM documents AS d
            JOIN document_versions AS v ON v.id = (
                SELECT MAX(latest.id) FROM document_versions AS latest
                WHERE latest.document_id = d.id
            )
            """
        ).fetchall()

    selected: set[int] = set()
    for row in rows:
        url = str(row["source_url"])
        if row["source_type"] == "simulated_demo":
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except json.JSONDecodeError:
                continue
            expiry = metadata.get("expiry_date")
            if expiry:
                try:
                    if date.fromisoformat(expiry) < today:
                        continue
                except (TypeError, ValueError):
                    continue
            if str(metadata.get("status_as_of_2026_11_20", "")).lower() in {
                "expired", "inactive", "withdrawn"
            }:
                continue
            selected.add(int(row["id"]))
            continue
        if url in CURRENT_STUDENT_PAGES:
            selected.add(int(row["id"]))
            continue
        haystack = f"{row['title']} {url}".lower()
        relevant = any(re.search(pattern, haystack) for pattern in STUDENT_TERM_PATTERNS)
        if not relevant:
            continue
        published = _publication_date(row)
        if published is not None and cutoff <= published <= today:
            selected.add(int(row["id"]))
    return selected, cutoff


def _publication_date(row) -> date | None:
    """Read a date from WordPress publication metadata or a dated upload path."""
    if row["document_type"] == "pdf":
        match = re.search(r"/((?:19|20)\d{2})/(0?[1-9]|1[0-2])(?:/|$)", row["source_url"])
        if match:
            year, month = int(match.group(1)), int(match.group(2))
            return date(year, month, calendar.monthrange(year, month)[1])
        return None

    path = Path(row["file_path"])
    if not path.is_file():
        return None
    try:
        soup = BeautifulSoup(path.read_bytes(), "html.parser")
    except OSError:
        return None
    for tag in soup.find_all(["meta", "time"]):
        value = tag.get("content") or tag.get("datetime")
        key = " ".join(str(tag.get(name, "")) for name in ("property", "name", "itemprop")).lower()
        if value and (
            tag.name == "time"
            or any(token in key for token in ("published", "datepublished", "article:published"))
        ):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
            except ValueError:
                continue
    return None


def _current_student_chunk(row) -> bool:
    """Avoid retrieving stale entries mixed into live college archive pages."""
    url = str(row["source_url"])
    content = str(row["content"]).lower()
    if urlsplit(url).path.lower().endswith(".pdf"):
        return _usable_pdf_text(content)
    if url == "https://bkbck.edu.in/examination/exam-timetable/":
        return (
            "winter session 2026" in content
            and "winter session 2025" not in content
            and content.count("welcome to b. k. birla") < 2
            and ("september 2026" in content or "october 2026" in content or "november 2026" in content)
        )
    if url in {
        "https://bkbck.edu.in/notices/",
        "https://bkbck.edu.in/category/notices/",
        "https://bkbck.edu.in/examination/circular-notifications/",
    }:
        return _contains_recent_notice_date(content)
    return True


def _usable_pdf_text(text: str) -> bool:
    """Reject scan artifacts such as repeated digits and OCR placeholder text."""
    words = re.findall(r"\b[a-z]{4,}\b", text.lower())
    return len(set(words)) >= 8


def _contains_recent_notice_date(text: str) -> bool:
    return bool(
        re.search(r"\bSeptember\s+(?:2[1-9]|30)(?:st|nd|rd|th)?[,]?\s+2026\b", text, re.I)
        or re.search(r"\b(?:2[1-9]|30)(?:st|nd|rd|th)?\s+September[,]?\s+2026\b", text, re.I)
        or re.search(rf"\b(?:October|November)\s+\d{{1,2}}(?:st|nd|rd|th)?[,]?\s+2026\b", text, re.I)
        or re.search(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:October|November)\s+2026\b", text, re.I)
    )


def retry_oversized_attachments() -> dict[str, object]:
    """Fetch the three known official attachments that exceed the normal cap."""
    load_dotenv(PROJECT_ROOT / ".env")
    database_path = get_database_path()
    domain = os.environ.get("COLLEGE_DOMAIN", "bkbck.edu.in").strip()
    crawler = SafeCrawler(
        domain,
        timeout=float(os.environ.get("CRAWL_TIMEOUT", "15")),
        delay_seconds=float(os.environ.get("CRAWL_DELAY_SECONDS", "1")),
        max_response_bytes=LARGE_PDF_BYTES,
    )
    source_ids = _source_ids(database_path)
    errors: list[dict[str, str]] = []
    fetched = 0
    scanned = 0
    chunks = 0
    for url in OVERSIZED_PDF_URLS:
        try:
            stored = fetch_and_store_document(
                _source_id_for_url(url, source_ids),
                url,
                crawler,
                category=_category_for_url(url),
                database_path=database_path,
                max_bytes=LARGE_PDF_BYTES,
            )
            fetched += 1
            scanned += stored.extraction_status == "no_text"
            chunks += _ensure_chunks(stored.id, database_path)
        except Exception as error:
            errors.append({"url": url, "message": str(error)[:400]})

    selected_ids, _ = _current_student_document_ids(database_path)
    index = build_vector_index(
        database_path=database_path,
        document_ids=selected_ids,
        chunk_filter=_current_student_chunk,
    )
    report_path = PROJECT_ROOT / "data" / "processed" / "official_site_snapshot.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    report["large_pdf_retry"] = {
        "attempted": len(OVERSIZED_PDF_URLS),
        "fetched": fetched,
        "scanned": scanned,
        "chunks_added": chunks,
        "errors": errors,
    }
    counts = report.setdefault("counts", {})
    counts["pdf_fetched"] = counts.get("pdf_fetched", 0) + fetched
    counts["scanned_documents"] = counts.get("scanned_documents", 0) + scanned
    counts["text_documents"] = counts.get("text_documents", 0) + fetched - scanned
    counts["chunks_added"] = counts.get("chunks_added", 0) + chunks
    report["index_vectors"] = index.vector_count
    report["index_model"] = index.model_name
    report["errors"] = [
        error for error in report.get("errors", [])
        if error.get("url") not in OVERSIZED_PDF_URLS
    ] + errors
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report["large_pdf_retry"]


def _sitemap_urls(crawler: SafeCrawler) -> tuple[list[str], list[str]]:
    crawler._load_robots_policy()
    declared_sitemaps = crawler._robots_parser.site_maps() or []
    sitemap_url = next(
        (url for url in declared_sitemaps if urlsplit(url).hostname == crawler.allowed_domain),
        SITEMAP_URL,
    )
    index_resource = crawler.fetch_resource(sitemap_url, max_bytes=5_000_000)
    sitemap_index = ElementTree.fromstring(index_resource.content)
    page_urls: set[str] = set()
    pdf_urls: set[str] = set()

    for entry in sitemap_index.findall("{*}sitemap"):
        location = entry.find("{*}loc")
        if location is None or not location.text:
            continue
        sitemap_name = urlsplit(location.text).path.rsplit("/", 1)[-1]
        if sitemap_name not in SITEMAP_NAMES:
            continue
        sitemap_resource = crawler.fetch_resource(location.text, max_bytes=5_000_000)
        sitemap = ElementTree.fromstring(sitemap_resource.content)
        for url_entry in sitemap.findall("{*}url"):
            page_location = url_entry.find("{*}loc")
            if page_location is None or not page_location.text:
                continue
            try:
                url = normalize_allowed_url(page_location.text, crawler.allowed_domain)
            except ValueError:
                continue
            if _is_pdf(url):
                pdf_urls.add(url)
            elif _is_public_html_url(url):
                page_urls.add(url)

    return sorted(page_urls), sorted(pdf_urls)


def _is_public_html_url(url: str) -> bool:
    path = unquote(urlsplit(url).path).lower()
    if any(part in path for part in ("/dashboard", "/wp-admin", "/elementor-hf/")):
        return False
    suffix = Path(path).suffix
    return not suffix or suffix in {".html", ".htm", ".php"}


def _is_pdf(url: str) -> bool:
    return urlsplit(url).path.lower().endswith(".pdf")


def _source_ids(database_path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    with get_connection(database_path) as connection:
        rows = connection.execute("SELECT id, name FROM sources WHERE enabled = 1").fetchall()
    for row in rows:
        result[str(row["name"])] = int(row["id"])
    return result


def _source_id_for_url(url: str, source_ids: dict[str, int]) -> int:
    path = urlsplit(url).path.lower()
    if "exam-timetable" in path or "timetable" in path:
        name = "Examination timetable"
    elif "examination" in path or "exam" in path:
        name = "Examination circulars"
    elif "notice" in path:
        name = "College notices"
    else:
        name = "College homepage"
    return source_ids.get(name, next(iter(source_ids.values())))


def _category_for_url(url: str) -> str:
    path = urlsplit(url).path.lower()
    if any(word in path for word in ("exam", "timetable", "circular")):
        return "examinations"
    if "notice" in path:
        return "notices"
    if any(word in path for word in ("admission", "fee", "scholarship")):
        return "admissions-and-fees"
    if any(word in path for word in ("course", "department", "academic")):
        return "academics"
    return "general"


def _ensure_chunks(document_id: int, database_path: Path) -> int:
    with get_connection(database_path) as connection:
        version = get_latest_document_version(connection, document_id)
        if version is None or version["extraction_status"] != "extracted":
            return 0
        existing = list_document_version_chunks(connection, int(version["id"]))
    if existing:
        return 0
    return len(chunk_document_version(int(version["id"]), database_path=database_path))


class _SnapshotDocTemplate(SimpleDocTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_pages: list[tuple[str, int]] = []

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == "SourceTitle":
            self.source_pages.append((flowable.getPlainText(), self.page))


def export_knowledge_pdf(output_path: Path = PDF_OUTPUT) -> dict[str, object]:
    """Export the current RAG corpus and append the original scanned PDFs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    database_path = get_database_path()
    with get_connection(database_path) as connection:
        rows = connection.execute(
            """
            SELECT d.id, d.title, d.source_url, d.document_type, d.category,
                   d.file_path, d.fetched_at, s.name AS source_name,
                   v.extracted_text, v.extraction_status
            FROM documents AS d
            JOIN document_versions AS v ON v.id = (
                SELECT MAX(latest.id) FROM document_versions AS latest
                WHERE latest.document_id = d.id
            )
            JOIN sources AS s ON s.id = d.source_id
            ORDER BY COALESCE(d.category, 'general'), lower(d.title), d.source_url
            """
        ).fetchall()
        records = [dict(row) for row in rows]

    selected_ids, cutoff = _current_student_document_ids(database_path)
    records = [row for row in records if int(row["id"]) in selected_ids]

    now = datetime.now(timezone.utc)
    scan_records = [
        row for row in records
        if row["extraction_status"] == "no_text"
        or (
            row["document_type"] == "pdf"
            and not _usable_pdf_text(row["extracted_text"] or "")
        )
    ]
    extracted_count = len(records) - len(scan_records)
    styles = _pdf_styles()
    doc = _SnapshotDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=19 * mm,
        rightMargin=19 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title="B.K. Birla College - Recent Student Updates Knowledge Base",
        author="Lumen AI",
    )
    story = [
        Spacer(1, 20 * mm),
        Paragraph("LUMEN AI", styles["Brand"]),
        Spacer(1, 7 * mm),
        Paragraph("B.K. Birla College", styles["CoverTitle"]),
        Paragraph("Recent Student Updates Knowledge Base", styles["CoverSubtitle"]),
        Spacer(1, 8 * mm),
        Paragraph(
            "A focused snapshot of recent student-facing notices, exam updates, "
            "academic items and fee information. Each entry retains its official source URL.",
            styles["Body"],
        ),
        Spacer(1, 7 * mm),
        _summary_table(records, extracted_count, scan_records, now, styles),
        Spacer(1, 8 * mm),
        Paragraph("Coverage and use", styles["SectionTitle"]),
        Paragraph(
            "This archive is generated from the official site's public sitemaps and "
            f"filtered to student-facing records published or uploaded since {cutoff.isoformat()}. "
            "Current notice and timetable landing pages are retained. The chatbot "
            "indexes the same records individually so citations point to their official page or attachment.",
            styles["Body"],
        ),
        Paragraph(
            f"{len(scan_records)} image-based or text-poor PDF attachments are included "
            "in the appendix as their original PDF pages. Their contents need OCR "
            "before the chatbot can answer questions that depend on text inside them.",
            styles["Body"],
        ),
        PageBreak(),
        Paragraph("Source catalogue", styles["SectionTitle"]),
        Paragraph(
            "Entries are ordered by topic and title. Use the clickable official URL "
            "to open the current source; the PDF records the fetched snapshot below.",
            styles["Body"],
        ),
        Spacer(1, 4 * mm),
    ]

    for row in records:
        story.extend(_record_flowables(row, styles))
    story.append(PageBreak())
    story.append(Paragraph("Scanned PDF appendices", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "The original page images follow. Their source URLs are listed in the "
            "catalogue above.",
            styles["Body"],
        )
    )
    for row in scan_records:
        story.append(Paragraph(html.escape(row["title"]), styles["SourceTitle"]))
        story.append(_url_paragraph(row["source_url"], styles["URL"]))
        story.append(Spacer(1, 3 * mm))

    doc.build(story, onFirstPage=_draw_footer(now), onLaterPages=_draw_footer(now))
    _append_scanned_pdfs(output_path, records, doc.source_pages)
    return {
        "output": str(output_path),
        "documents": len(records),
        "text_extractable": extracted_count,
        "scanned_pdfs_appended": len(scan_records),
        "pages": pymupdf.open(output_path).page_count,
        "created_at": now.isoformat(),
    }


def _pdf_styles():
    font_dir = Path("C:/Windows/Fonts")
    if (font_dir / "arial.ttf").exists() and (font_dir / "arialbd.ttf").exists():
        pdfmetrics.registerFont(TTFont("LumenArial", str(font_dir / "arial.ttf")))
        pdfmetrics.registerFont(TTFont("LumenArialBold", str(font_dir / "arialbd.ttf")))
        font, bold = "LumenArial", "LumenArialBold"
    else:
        font, bold = "Helvetica", "Helvetica-Bold"
    base = getSampleStyleSheet()
    return {
        "Brand": ParagraphStyle("Brand", parent=base["Normal"], fontName=bold, fontSize=11, leading=14, textColor=colors.HexColor("#4ccac5"), alignment=TA_CENTER, spaceAfter=4),
        "CoverTitle": ParagraphStyle("CoverTitle", parent=base["Title"], fontName=bold, fontSize=29, leading=35, textColor=colors.HexColor("#14253b"), alignment=TA_CENTER, spaceAfter=5),
        "CoverSubtitle": ParagraphStyle("CoverSubtitle", parent=base["Normal"], fontName=font, fontSize=13, leading=18, textColor=colors.HexColor("#4a657c"), alignment=TA_CENTER),
        "SectionTitle": ParagraphStyle("SectionTitle", parent=base["Heading1"], fontName=bold, fontSize=17, leading=22, textColor=colors.HexColor("#14304d"), spaceBefore=6, spaceAfter=8, keepWithNext=True),
        "SourceTitle": ParagraphStyle("SourceTitle", parent=base["Heading2"], fontName=bold, fontSize=11, leading=14, textColor=colors.HexColor("#163958"), spaceBefore=8, spaceAfter=4, keepWithNext=True),
        "Body": ParagraphStyle("Body", parent=base["BodyText"], fontName=font, fontSize=8.5, leading=11.3, textColor=colors.HexColor("#263544"), spaceAfter=3, splitLongWords=True),
        "URL": ParagraphStyle("URL", parent=base["BodyText"], fontName=font, fontSize=7.5, leading=10, textColor=colors.HexColor("#147d8c"), spaceAfter=3, splitLongWords=True),
        "Meta": ParagraphStyle("Meta", parent=base["BodyText"], fontName=font, fontSize=7.5, leading=10, textColor=colors.HexColor("#526274"), spaceAfter=2),
        "ArchiveNote": ParagraphStyle("ArchiveNote", parent=base["BodyText"], fontName=font, fontSize=8, leading=11, textColor=colors.HexColor("#806027"), backColor=colors.HexColor("#fbf5e9"), borderColor=colors.HexColor("#edddb9"), borderWidth=0.5, borderPadding=6, spaceBefore=4, spaceAfter=6),
    }


def _summary_table(records, extracted_count, scan_records, created_at, styles):
    data = [
        ["SNAPSHOT", created_at.strftime("%d %b %Y %H:%M UTC")],
        ["OFFICIAL SOURCES", "https://bkbck.edu.in/"],
        ["SOURCE RECORDS", str(len(records))],
        ["TEXT-EXTRACTABLE", str(extracted_count)],
        ["SCANNED PDF APPENDICES", str(len(scan_records))],
    ]
    table = Table(data, colWidths=[53 * mm, 105 * mm], hAlign="CENTER")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f2f6f8")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#51687c")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#1b3045")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d8e1e6")),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return table


def _record_flowables(row, styles):
    title = _safe_pdf_text(row["title"] or "Untitled college source")
    flowables = [Paragraph(html.escape(title), styles["SourceTitle"])]
    flowables.append(
        Paragraph(
            html.escape(
                f"Source group: {row['source_name']} | Type: {row['document_type']} | "
                f"Fetched: {row['fetched_at']} | Extraction: {row['extraction_status']}"
            ),
            styles["Meta"],
        )
    )
    flowables.append(_url_paragraph(row["source_url"], styles["URL"]))
    text = _safe_pdf_text(row["extracted_text"] or "")
    if row["source_url"] == "https://bkbck.edu.in/examination/exam-timetable/":
        text = (
            "This current official page lists Winter Session 2026 examination "
            "timetable attachments. The attachment list includes a recent entry "
            "dated 26 September 2026. Schedule details are in the linked PDFs; "
            "some are image scans, so exact subject-level exam dates are not "
            "searchable in the chatbot text. Open the official page to view the attachments."
        )
    if not text.strip() or (
        row["document_type"] == "pdf" and not _usable_pdf_text(text)
    ):
        flowables.append(
            Paragraph(
                "No selectable text was found in this attachment. The original "
                "scanned PDF pages are included in the appendix.",
                styles["ArchiveNote"],
            )
        )
    else:
        for block in _text_blocks(text):
            flowables.append(Paragraph(html.escape(block).replace("\n", "<br/>"), styles["Body"]))
    flowables.append(Spacer(1, 2 * mm))
    return flowables


def _url_paragraph(url: str, style):
    safe_url = html.escape(url, quote=True)
    return Paragraph(f'<link href="{safe_url}" color="#147d8c">{safe_url}</link>', style)


def _text_blocks(text: str, max_words: int = 180):
    clean = re.sub(r"[\t\r]+", " ", text.replace("\f", "\n"))
    for line in clean.splitlines():
        words = line.split()
        while words:
            block = words[:max_words]
            yield " ".join(block)
            words = words[max_words:]


def _safe_pdf_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        " " if unicodedata.category(char) in {"Cc", "Cf", "Co", "Cs"} else char
        for char in normalized
    )


def _draw_footer(created_at):
    def draw(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#dbe4e9"))
        canvas.setLineWidth(0.45)
        canvas.line(19 * mm, 14 * mm, A4[0] - 19 * mm, 14 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#6b7b89"))
        canvas.drawString(19 * mm, 9 * mm, f"Lumen AI | Snapshot {created_at:%d %b %Y UTC}")
        canvas.drawRightString(A4[0] - 19 * mm, 9 * mm, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()
    return draw


def _append_scanned_pdfs(output_path, records, source_pages):
    scanned = []
    seen_paths: set[Path] = set()
    for row in records:
        if row["extraction_status"] != "no_text" and not (
            row["document_type"] == "pdf"
            and not _usable_pdf_text(row["extracted_text"] or "")
        ):
            continue
        path = Path(row["file_path"])
        if path.exists() and path not in seen_paths:
            scanned.append((row["title"], path))
            seen_paths.add(path)

    archive = pymupdf.open(output_path)
    outline = [[1, "Official source catalogue", 1]]
    outline.extend([[2, title[:180], page] for title, page in source_pages])
    for title, path in scanned:
        try:
            original = pymupdf.open(path)
            first_page = archive.page_count + 1
            archive.insert_pdf(original, links=True, annots=True)
            outline.append([1, f"Scanned PDF: {title[:150]}", first_page])
            original.close()
        except (OSError, RuntimeError, ValueError):
            continue
    archive.set_toc(outline)
    archive.set_metadata({
        "title": "B.K. Birla College Official Site Knowledge Base",
        "author": "Lumen AI",
        "subject": "Public official college pages, notices, courses and linked documents",
        "keywords": "B.K. Birla College, notices, academics, examinations, fees",
    })
    temporary_path = output_path.with_name(output_path.stem + ".combined.pdf")
    archive.save(temporary_path, garbage=3, deflate=True)
    archive.close()
    os.replace(temporary_path, output_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-only", action="store_true", help="Rebuild the curated index and PDF from the local corpus without fetching the site.")
    parser.add_argument("--retry-oversized", action="store_true", help="Retry the three oversized official PDFs and rebuild the archive.")
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    curated_summary: dict[str, object] | None = None
    if args.retry_oversized:
        print("Large PDF retry:", json.dumps(retry_oversized_attachments()), flush=True)
    elif not args.export_only:
        sync_official_site()
    else:
        selected_ids, cutoff = _current_student_document_ids(get_database_path())
        index = build_vector_index(
            database_path=get_database_path(),
            document_ids=selected_ids,
            chunk_filter=_current_student_chunk,
        )
        print(
            f"Curated index: {index.vector_count} vectors from {len(selected_ids)} "
            f"sources since {cutoff.isoformat()}",
            flush=True,
        )
        curated_summary = {
            "student_data_cutoff": cutoff.isoformat(),
            "selected_documents": len(selected_ids),
            "index_vectors": index.vector_count,
            "index_model": index.model_name,
        }
    summary = export_knowledge_pdf()
    if curated_summary is not None:
        report_path = PROJECT_ROOT / "data" / "processed" / "official_site_snapshot.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        report.update(curated_summary)
        report["knowledge_pdf"] = summary
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PDF archive:", json.dumps(summary))


if __name__ == "__main__":
    main()
