from chatbot.retrieval import KnowledgeBase, tokenize


def test_tokenize_normalises_brand_names_and_plurals():
    assert tokenize("Dar Global villas") == tokenize("DarGlobal villa")
    assert "the" not in tokenize("the villas")


def test_search_finds_relevant_chunk(kb):
    hits = kb.search("Which brand partners with DarGlobal on Painite villas?")
    assert hits and hits[0].chunk.id == "dg-2"


def test_search_respects_site_hint(kb):
    hits = kb.search("how do auctions work on Wasalt")
    assert hits[0].chunk.site == "wasalt"
    assert hits[0].chunk.id == "ws-1"


def test_search_returns_empty_for_unknown_terms(kb):
    assert kb.search("quantum chromodynamics lagrangian") == []


def test_search_limits_chunks_per_page(kb_path):
    import json

    from tests.conftest import SAMPLE_CHUNKS

    extra = [dict(SAMPLE_CHUNKS[0], id=f"x{i}", text=f"AIDA villa golf Oman detail {i}") for i in range(5)]
    path = kb_path.parent / "many.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in [*SAMPLE_CHUNKS, *extra]), encoding="utf-8")
    hits = KnowledgeBase.from_jsonl(path).search("AIDA golf Oman villa", k=6, max_per_page=2)
    urls = [h.chunk.url for h in hits]
    assert max(urls.count(u) for u in set(urls)) <= 2


def test_stats(kb):
    stats = kb.stats()
    assert stats["chunks"] == 4
    assert stats["chunks_per_site"] == {"darglobal": 2, "wasalt": 2}
    assert stats["last_scraped"].startswith("2026-11-20T11")


def test_missing_file_has_helpful_error(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="scraper"):
        KnowledgeBase.from_jsonl(tmp_path / "nope.jsonl")
