"""HTML -> clean, structured text.

The output is deliberately simple: headings are rendered as ``## Heading`` lines
and everything else is plain text, one visual block per line. The chunker in
``build_kb`` relies on this convention.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import parse_qsl, urldefrag, urlencode, urljoin, urlparse

from bs4 import BeautifulSoup, Comment

from .config import SKIP_EXTENSIONS, TRACKING_PARAMS

# Tags that never contain user-visible article content.
_DROP_TAGS = ("script", "style", "noscript", "svg", "iframe", "template", "canvas",
              "nav", "footer", "form", "button", "select", "option", "dialog")
_NOISE_ATTR = re.compile(
    r"cookie|consent|onetrust|gdpr|popup|modal|newsletter|breadcrumb|skip-link", re.I
)
_BLOCK_TAGS = ("p", "div", "section", "article", "li", "ul", "ol", "tr", "table",
               "h1", "h2", "h3", "h4", "h5", "h6", "br", "dt", "dd", "figcaption",
               "blockquote", "main", "address")
_WS = re.compile(r"[ \t\r\f\v\u00a0\u200b]+")


def normalize_url(url: str) -> str | None:
    """Canonicalise a URL: no fragment, no tracking params, no trailing slash."""
    url, _ = urldefrag(url.strip())
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    if parsed.path.lower().endswith(SKIP_EXTENSIONS):
        return None
    query = urlencode(
        sorted((k, v) for k, v in parse_qsl(parsed.query) if not TRACKING_PARAMS.match(k))
    )
    path = parsed.path.rstrip("/") or ""
    host = parsed.netloc.lower().removeprefix("www.")
    return parsed._replace(netloc=host, path=path, query=query, fragment="").geturl()


def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    out: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        norm = normalize_url(urljoin(base_url, href))
        if norm:
            out.append(norm)
    return list(dict.fromkeys(out))


def _meta(soup: BeautifulSoup, *names: str) -> str:
    for name in names:
        tag = soup.find("meta", attrs={"name": name}) or soup.find(
            "meta", attrs={"property": name}
        )
        if tag and tag.get("content"):
            value = _WS.sub(" ", tag["content"]).strip()
            if value.lower() not in ("null", "none", "undefined", "n/a", ""):  # CMS placeholders
                return value
    return ""


def _flatten_jsonld(node: Any, out: list[str], depth: int = 0) -> None:
    """Pull human-meaningful strings out of schema.org JSON-LD."""
    if depth > 6:
        return
    if isinstance(node, list):
        for item in node:
            _flatten_jsonld(item, out, depth + 1)
        return
    if not isinstance(node, dict):
        return
    kind = node.get("@type")
    kinds = kind if isinstance(kind, list) else [kind]
    if "FAQPage" in kinds:
        for q in node.get("mainEntity", []) or []:
            if isinstance(q, dict):
                answer = (q.get("acceptedAnswer") or {}).get("text", "")
                out.append(f"Q: {q.get('name', '')}\nA: {_strip_tags(answer)}")
        return
    if any(k in kinds for k in ("Organization", "Corporation", "RealEstateAgent",
                                "Product", "Article", "NewsArticle", "BlogPosting",
                                "Residence", "Apartment", "House", "Offer",
                                "RealEstateListing", "Place", "LocalBusiness")):
        bits = []
        for key in ("name", "headline", "description", "slogan"):
            if isinstance(node.get(key), str) and node[key].strip():
                bits.append(f"{key}: {_strip_tags(node[key])}")
        offers = node.get("offers")
        if isinstance(offers, dict) and offers.get("price"):
            bits.append(f"price: {offers.get('priceCurrency', '')} {offers['price']}".strip())
        addr = node.get("address")
        if isinstance(addr, dict):
            parts = [v for k, v in addr.items() if not k.startswith("@") and isinstance(v, str) and v]
            if parts:
                bits.append("address: " + ", ".join(parts))
        if node.get("telephone"):
            bits.append(f"telephone: {node['telephone']}")
        if node.get("email"):
            bits.append(f"email: {node['email']}")
        if bits:
            out.append(" | ".join(bits))
    for value in node.values():
        if isinstance(value, (dict, list)):
            _flatten_jsonld(value, out, depth + 1)


def _strip_tags(text: str) -> str:
    return _WS.sub(" ", BeautifulSoup(text or "", "lxml").get_text(" ")).strip()


def extract_page(html: str, url: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")

    title = _WS.sub(" ", soup.title.get_text()).strip() if soup.title else ""
    description = _meta(soup, "description", "og:description")
    og_title = _meta(soup, "og:title")

    # Structured data first (before we strip <script>).
    structured: list[str] = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            _flatten_jsonld(json.loads(script.string or ""), structured)
        except (json.JSONDecodeError, TypeError):
            continue

    # A page-level <header> is site chrome, but a <header> inside an article/section
    # usually carries the article title, so keep those.
    for header in soup.find_all("header"):
        if header.find_parent(["article", "main", "section"]) is None:
            header.decompose()
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    for tag in soup.find_all(attrs={"role": re.compile("navigation|banner|dialog", re.I)}):
        tag.decompose()
    for tag in soup.find_all(True):
        if getattr(tag, "decomposed", False) or tag.attrs is None:
            continue
        ident = " ".join([tag.get("id", "") or "", " ".join(tag.get("class", []) or [])])
        if ident.strip() and _NOISE_ATTR.search(ident):
            tag.decompose()

    # Headings -> markdown-ish markers.
    for h in soup.find_all(re.compile(r"^h[1-4]$")):
        text = _WS.sub(" ", h.get_text(" ")).strip()
        h.clear()
        h.string = f"\n## {text}\n" if text else ""

    for tag in soup.find_all(_BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")

    body = soup.body or soup
    raw_lines = body.get_text(" ").splitlines()

    lines: list[str] = []
    for line in raw_lines:
        line = _WS.sub(" ", line).strip()
        if len(line) < 2 or re.fullmatch(r"[\W_]+", line):
            continue
        if lines and lines[-1] == line:
            continue
        lines.append(line)

    headings = [ln[3:] for ln in lines if ln.startswith("## ")]
    h1 = headings[0] if headings else ""

    return {
        "url": url,
        "title": og_title or title,
        "h1": h1,
        "description": description,
        "structured": list(dict.fromkeys(structured)),
        "text": "\n".join(lines),
    }
