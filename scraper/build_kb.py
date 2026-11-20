"""Turn raw crawled pages into the retrieval-ready knowledge base (JSONL chunks)."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from .config import KB_PATH, RAW_DIR

log = logging.getLogger("scraper.build_kb")

TARGET_CHARS = 900  # soft size of a chunk
MAX_CHARS = 1400  # hard ceiling
MIN_CHARS = 70  # drop anything smaller
OVERLAP_CHARS = 160  # context carried into the next chunk
BOILERPLATE_RATIO = 0.30  # a line on >=30% of a site's pages is site chrome
BOILERPLATE_MIN_PAGES = 8


def _load_raw(raw_dir: Path) -> list[dict]:
    docs = []
    for path in sorted(raw_dir.rglob("*.json")):
        try:
            docs.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            log.warning("Skipping unreadable file %s", path)
    return docs


def _boilerplate_lines(docs: list[dict]) -> dict[str, set[str]]:
    """Per site: lines that repeat on many pages (menus, footers, banners)."""
    by_site: dict[str, list[dict]] = {}
    for d in docs:
        by_site.setdefault(d["site"], []).append(d)
    result: dict[str, set[str]] = {}
    for site, site_docs in by_site.items():
        if len(site_docs) < BOILERPLATE_MIN_PAGES:
            result[site] = set()
            continue
        counts: Counter[str] = Counter()
        for d in site_docs:
            counts.update(set(d["text"].splitlines()))
        threshold = max(BOILERPLATE_MIN_PAGES, int(len(site_docs) * BOILERPLATE_RATIO))
        result[site] = {line for line, n in counts.items() if n >= threshold}
        log.info("%s: %d boilerplate lines removed", site, len(result[site]))
    return result


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\w)(\+?\d[\d\s().-]{7,}\d)(?!\w)")
_LICENSE_LINE = re.compile(r"\b(FAL License|REGA Ad License|Ad License No|Licen[sc]e No)\b", re.I)


def _is_repetitive(line: str) -> bool:
    """'Related listings' blobs repeat the same few words dozens of times."""
    tokens = line.split()
    return len(tokens) > 40 and len(set(tokens)) / len(tokens) < 0.3


def _scrub_listing_lines(lines: list[str]) -> list[str]:
    """Data minimisation for individual property ads: no phone numbers, emails or
    broker licence boilerplate. Company-level contact details on other pages are kept."""
    out = []
    for line in lines:
        if _EMAIL.search(line) or _PHONE.search(line) or _LICENSE_LINE.search(line):
            continue
        if re.fullmatch(r"(Call|Call Now|WhatsApp|Chat|Share|Save|Report)", line, re.I):
            continue
        out.append(line)
    return out


def _clean_structured(text: str) -> str:
    """Drop stringified Python lists (e.g. social-profile URL arrays) from JSON-LD summaries."""
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\s*,\s*\|", " |", text)  # "Riyadh, | next" -> "Riyadh | next"
    return re.sub(r"\s*\|\s*\|\s*", " | ", text).strip(" |,")


def _split_long(line: str) -> list[str]:
    if len(line) <= MAX_CHARS:
        return [line]
    parts, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", line):
        if len(current) + len(sentence) + 1 > TARGET_CHARS and current:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    # A sentence with no punctuation can still be huge: hard split.
    out = []
    for p in parts:
        out.extend(p[i : i + MAX_CHARS] for i in range(0, len(p), MAX_CHARS))
    return out


def chunk_lines(lines: list[str]) -> list[tuple[str, str]]:
    """Return ``(heading, text)`` chunks. Splits prefer heading boundaries."""
    chunks: list[tuple[str, str]] = []
    heading = ""
    buf: list[str] = []
    buf_len = 0
    buf_heading = ""

    def flush(carry: bool) -> None:
        nonlocal buf, buf_len, buf_heading
        text = "\n".join(buf).strip()
        if len(text) >= MIN_CHARS:
            chunks.append((buf_heading, text))
        tail = ""
        if carry and buf:
            tail = buf[-1] if len(buf[-1]) <= OVERLAP_CHARS and not buf[-1].startswith("## ") else ""
        buf = [tail] if tail else []
        buf_len = len(tail)
        buf_heading = heading

    for raw in lines:
        if raw.startswith("## "):
            heading = raw[3:].strip()
            # Start a new chunk at a heading once the current one has substance.
            if buf_len >= TARGET_CHARS * 0.45:
                flush(carry=False)
            if not buf:
                buf_heading = heading
            buf.append(raw)
            buf_len += len(raw)
            continue
        for piece in _split_long(raw):
            if buf and buf_len + len(piece) > TARGET_CHARS:
                flush(carry=True)
                if not buf_heading:
                    buf_heading = heading
            if not buf:
                buf_heading = heading
            buf.append(piece)
            buf_len += len(piece) + 1
    flush(carry=False)
    return chunks


_BRAND_SUFFIX = re.compile(r"\s*[|\-–—]\s*(DarGlobal|Dar Global|Wasalt|وصلت)[^|]*$", re.I)


def _slug_title(url: str) -> str:
    slug = url.rstrip("/").rsplit("/", 1)[-1] or url.split("//", 1)[-1]
    slug = re.sub(r"[-_]\d{5,}$", "", slug)  # trailing listing ids
    return re.sub(r"[-_]+", " ", slug).strip().title()


def _titles(docs: list[dict]) -> dict[str, str]:
    """Pick a meaningful title per URL. Some pages share one generic <title>
    (e.g. every Wasalt policy page) - for those fall back to the H1, then the URL slug."""
    cleaned = {d["url"]: _BRAND_SUFFIX.sub("", d.get("title", "")).strip() for d in docs}
    counts = Counter((d["site"], cleaned[d["url"]]) for d in docs)
    result = {}
    for d in docs:
        title = cleaned[d["url"]]
        if not title or counts[(d["site"], title)] >= 3:
            title = d.get("h1") or _slug_title(d["url"])
        result[d["url"]] = title
    return result


def build(raw_dir: Path = RAW_DIR, out_path: Path = KB_PATH) -> dict:
    docs = _load_raw(raw_dir)
    if not docs:
        raise SystemExit(f"No raw pages found in {raw_dir}. Run the crawl first.")
    boilerplate = _boilerplate_lines(docs)
    titles = _titles(docs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    per_site: Counter[str] = Counter()
    pages_per_site: Counter[str] = Counter()
    total = 0

    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for doc in docs:
            site = doc["site"]
            drop = boilerplate.get(site, set())
            lines = [ln for ln in doc["text"].splitlines() if ln not in drop and not _is_repetitive(ln)]
            if "/property/" in doc["url"]:
                lines = _scrub_listing_lines(lines)
            title = titles[doc["url"]]

            pieces: list[tuple[str, str]] = []
            overview = [title.rstrip(".!?:; ") + "."]
            if doc.get("description"):
                overview.append(doc["description"])
            overview.extend(_clean_structured(s) for s in doc.get("structured", [])[:6])
            if len(" ".join(overview)) >= MIN_CHARS:
                pieces.append(("Overview", "\n".join(overview)))
            pieces.extend(chunk_lines(lines))

            n = 0
            for heading, text in pieces:
                digest = hashlib.sha1(f"{site}|{text}".encode()).hexdigest()
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                chunk = {
                    "id": f"{site}-{digest[:10]}",
                    "site": site,
                    "site_label": doc["site_label"],
                    "url": doc["url"],
                    "title": title,
                    "heading": heading,
                    "text": re.sub(r"^## ", "", text, flags=re.M),
                    "scraped_at": doc["scraped_at"],
                }
                fh.write(json.dumps(chunk, ensure_ascii=False, separators=(",", ":")) + "\n")
                n += 1
            if n:
                pages_per_site[site] += 1
            per_site[site] += n
            total += n

    meta = {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "chunks": total,
        "chunks_per_site": dict(per_site),
        "pages_per_site": dict(pages_per_site),
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Knowledge base written: %s (%d chunks)", out_path, total)
    return meta
