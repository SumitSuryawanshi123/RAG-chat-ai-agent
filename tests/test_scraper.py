import json

from scraper.build_kb import MAX_CHARS, build, chunk_lines
from scraper.extract import extract_links, extract_page, normalize_url

HTML = """
<html><head><title>Villas | DarGlobal</title>
<meta name="description" content="Luxury villas">
<script type="application/ld+json">{"@type":"FAQPage","mainEntity":[{"name":"Is it freehold?",
"acceptedAnswer":{"text":"<p>Yes, freehold.</p>"}}]}</script></head>
<body>
<nav><a href="/menu">MENU ITEM</a></nav>
<header><a href="/">Top bar</a></header>
<main>
  <h1>Luxury Villas</h1>
  <p>Beachfront villas in <b>Oman</b>.</p>
  <div class="cookie-banner">We use cookies</div>
  <ul><li>Three bedrooms</li><li>Private pool</li></ul>
  <a href="/projects/aida?utm_source=x#top">AIDA</a>
  <a href="/brochure.pdf">pdf</a>
</main>
<footer>Footer junk</footer>
<script>var x = 1;</script>
</body></html>
"""


def test_extract_page_cleans_and_structures():
    doc = extract_page(HTML, "https://darglobal.co.uk/villas")
    assert doc["title"] == "Villas | DarGlobal"
    assert doc["description"] == "Luxury villas"
    assert "## Luxury Villas" in doc["text"]
    assert "Beachfront villas in Oman" in doc["text"]
    assert "Private pool" in doc["text"]
    for junk in ("MENU ITEM", "Footer junk", "We use cookies", "var x"):
        assert junk not in doc["text"]
    assert any("Is it freehold?" in s and "Yes, freehold." in s for s in doc["structured"])


def test_placeholder_description_is_ignored():
    html = '<html><head><title>T</title><meta name="description" content="null"></head><body><p>x</p></body></html>'
    assert extract_page(html, "https://x.test/")["description"] == ""


def test_normalize_url_strips_tracking_and_fragments():
    assert normalize_url("https://www.Example.com/a/?utm_source=x&b=2#frag") == "https://example.com/a?b=2"
    assert normalize_url("https://example.com/file.pdf") is None
    assert normalize_url("mailto:a@b.c") is None


def test_extract_links_skips_assets():
    links = extract_links(HTML, "https://darglobal.co.uk/villas")
    assert "https://darglobal.co.uk/projects/aida" in links
    assert not any(link.endswith(".pdf") for link in links)


def test_chunker_respects_size_and_keeps_headings():
    lines = ["## Section A"] + [f"Sentence number {i} about luxury villas in Oman." * 3 for i in range(30)]
    lines += ["## Section B", "Short paragraph that is long enough to survive the minimum size filter for chunks."]
    chunks = chunk_lines(lines)
    assert len(chunks) > 2
    assert all(len(text) <= MAX_CHARS + 200 for _, text in chunks)
    assert chunks[0][0] == "Section A"
    assert chunks[-1][0] == "Section B"


def test_chunker_splits_giant_single_line():
    chunks = chunk_lines(["word " * 1000])
    assert len(chunks) >= 3


def test_structured_data_cleaner_drops_stringified_lists():
    from scraper.build_kb import _clean_structured

    dirty = "name: Wasalt | address: Saudi Arabia, Riyadh, ['https://fb.com/w', 'https://x.com/w'] | telephone: 8001"
    assert _clean_structured(dirty) == "name: Wasalt | address: Saudi Arabia, Riyadh | telephone: 8001"
    assert _clean_structured("name: DarGlobal | description: Luxury") == "name: DarGlobal | description: Luxury"


def test_listing_pages_are_scrubbed_of_contact_details_and_repetition(tmp_path):
    raw = tmp_path / "raw" / "wasalt"
    raw.mkdir(parents=True)
    related = " ".join(["Apartment 50 SQM with 1 Bedroom"] * 12)
    doc = {
        "url": "https://wasalt.sa/en/property/rent/apartment-1", "title": "Apartment 50 SQM For Rent",
        "h1": "", "description": "Apartment in Riyadh", "structured": [], "site": "wasalt",
        "site_label": "Wasalt", "scraped_at": "2026-11-20T00:00:00+00:00",
        "text": "## Property Description\nApartment for rent in Jareer, Riyadh. Price: 2800 SAR per year, one bedroom.\n"
                "Call +966 50 123 4567 for viewing\nbroker@example.com\nFAL License No: 1200015014\nCall\n" + related,
    }
    (raw / "1.json").write_text(json.dumps(doc), encoding="utf-8")
    out = tmp_path / "kb.jsonl"
    build(tmp_path / "raw", out)
    text = "\n".join(json.loads(line)["text"] for line in out.read_text(encoding="utf-8").splitlines())
    assert "Price: 2800 SAR" in text
    for leaked in ("966", "@example.com", "FAL License", "Apartment 50 SQM with 1 Bedroom Apartment"):
        assert leaked not in text


def test_build_removes_boilerplate_and_dedupes(tmp_path):
    raw = tmp_path / "raw" / "darglobal"
    raw.mkdir(parents=True)
    for i in range(10):
        doc = {
            "url": f"https://darglobal.co.uk/p{i}", "title": f"Page {i} | DarGlobal", "h1": "", "description": "",
            "structured": [], "site": "darglobal", "site_label": "DarGlobal", "scraped_at": "2026-11-20T00:00:00+00:00",
            "text": "REPEATED MENU LINE\n## Heading\n" + f"Unique content for page {i}. " * 8,
        }
        (raw / f"{i}.json").write_text(json.dumps(doc), encoding="utf-8")
    out = tmp_path / "kb.jsonl"
    meta = build(tmp_path / "raw", out)
    chunks = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert meta["chunks"] == len(chunks) > 0
    assert not any("REPEATED MENU LINE" in c["text"] for c in chunks)
    assert all(c["title"].startswith("Page ") and "DarGlobal" not in c["title"] for c in chunks)
    assert len({c["id"] for c in chunks}) == len(chunks)
