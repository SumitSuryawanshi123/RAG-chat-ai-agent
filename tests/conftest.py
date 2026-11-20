import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_CHUNKS = [
    {
        "id": "dg-1", "site": "darglobal", "site_label": "DarGlobal",
        "url": "https://darglobal.co.uk/aida-trump-villas", "title": "AIDA Trump Villas",
        "heading": "Overview",
        "text": "AIDA is a luxury golf community in Oman. The Trump villas offer sea views, "
                "a championship golf course and branded residences in Muscat.",
        "scraped_at": "2026-11-20T10:00:00+00:00",
    },
    {
        "id": "dg-2", "site": "darglobal", "site_label": "DarGlobal",
        "url": "https://darglobal.co.uk/partners/lamborghini", "title": "Lamborghini partnership",
        "heading": "Our partner",
        "text": "DarGlobal partners with Automobili Lamborghini on the Painite villas, "
                "a design-inspired collection of ultra-luxury homes.",
        "scraped_at": "2026-11-20T10:00:00+00:00",
    },
    {
        "id": "ws-1", "site": "wasalt", "site_label": "Wasalt",
        "url": "https://auction.wasalt.sa/en", "title": "Wasalt Auctions",
        "heading": "How auctions work",
        "text": "Wasalt organises online and hybrid real estate auctions in Saudi Arabia. "
                "Register, deposit via the e-wallet, then place bids on verified properties.",
        "scraped_at": "2026-11-20T11:00:00+00:00",
    },
    {
        "id": "ws-2", "site": "wasalt", "site_label": "Wasalt",
        "url": "https://wasalt.sa/en/apartments-for-sale-in-riyadh", "title": "Apartments for sale in Riyadh",
        "heading": "Listings",
        "text": "Browse verified apartments for sale in Riyadh with prices in SAR, bedrooms and area in sqm.",
        "scraped_at": "2026-11-20T11:00:00+00:00",
    },
]


@pytest.fixture
def kb_path(tmp_path) -> Path:
    path = tmp_path / "kb.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in SAMPLE_CHUNKS), encoding="utf-8")
    return path


@pytest.fixture
def kb(kb_path):
    from chatbot.retrieval import KnowledgeBase

    return KnowledgeBase.from_jsonl(kb_path)
