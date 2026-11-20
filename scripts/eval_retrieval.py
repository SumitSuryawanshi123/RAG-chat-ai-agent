"""Retrieval sanity benchmark: does the right evidence appear in the top-k passages?

Each case is (question, expected substring). A case passes if the substring appears
(case-insensitively) in the url, title or text of any of the top-k hits. This checks
retrieval only - no LLM or API key is involved.

    python scripts/eval_retrieval.py            # prints a table, exit 1 if recall < 80%
    python scripts/eval_retrieval.py -v         # also print the top hits per question
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chatbot.config import get_settings  # noqa: E402
from chatbot.retrieval import KnowledgeBase  # noqa: E402

CASES = [
    # DarGlobal
    ("Who is the CEO of DarGlobal?", "Ziad El Chaar"),
    ("Is DarGlobal listed on a stock exchange?", "London Stock Exchange"),
    ("Tell me about Trump International Hotel & Tower Dubai", "trump-international"),
    ("What is the Da Vinci Tower by Pagani?", "pagani"),
    ("Which projects does DarGlobal have in Oman?", "oman"),
    ("What is AIDA?", "aida"),
    ("DarGlobal luxury villas in Spain", "spain"),
    ("Does DarGlobal have a Lamborghini partnership?", "lamborghini"),
    ("Elie Saab residences", "elie"),
    ("What is DarGlobal developing in Qatar?", "qatar"),
    ("Dar Global Saudi Arabia Jeddah development", "jeddah"),
    ("Does DarGlobal offer golf communities?", "golf"),
    # Wasalt
    ("How many registered users does Wasalt have?", "705k"),
    ("What is Wasalt Business for brokers?", "CRM"),
    ("How do Wasalt auctions work?", "auction"),
    ("Can foreigners buy property in Saudi Arabia?", "foreigners"),
    ("What is the Ejar system?", "ejar"),
    ("apartments for sale in Riyadh", "riyadh"),
    ("villas for sale in Jeddah", "jeddah"),
    ("Where can I buy land in Makkah?", "makkah"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-k", type=int, default=get_settings().top_k)
    args = parser.parse_args()

    kb = KnowledgeBase.from_jsonl(get_settings().kb_path)
    passed = 0
    for question, expected in CASES:
        hits = kb.search(question, k=args.k)
        needle = expected.lower()
        rank = next(
            (i for i, h in enumerate(hits, 1)
             if needle in f"{h.chunk.url} {h.chunk.title} {h.chunk.text}".lower()),
            None,
        )
        passed += rank is not None
        print(f"{'PASS' if rank else 'FAIL'}  rank={rank or '-':<2}  {question}")
        if args.verbose or not rank:
            for h in hits[:3]:
                print(f"        {h.score:6.1f}  {h.chunk.site:<9} {h.chunk.title[:48]:<48} {h.chunk.url[-48:]}")
    recall = passed / len(CASES)
    print(f"\nrecall@{args.k}: {passed}/{len(CASES)} = {recall:.0%}")
    return 0 if recall >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
