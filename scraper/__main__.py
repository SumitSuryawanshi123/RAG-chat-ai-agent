"""CLI:  python -m scraper {crawl,build,all} [options]"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .build_kb import build
from .config import SITES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scraper", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("crawl", "all"):
        p = sub.add_parser(name, help="crawl the sites" + (" and build the KB" if name == "all" else ""))
        p.add_argument("--site", choices=["all", *SITES], default="all")
        p.add_argument("--refresh", action="store_true", help="re-fetch pages already cached in data/raw")
        p.add_argument("--limit", type=int, help="max pages per site config (smoke tests)")
        p.add_argument("--headed", action="store_true", help="show the browser window")
    sub.add_parser("build", help="rebuild data/knowledge_base.jsonl from data/raw")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    if args.command in ("crawl", "all"):
        from .crawler import Crawler  # imported lazily so `build` works without Playwright

        keys = list(SITES) if args.site == "all" else [args.site]
        configs = [cfg for k in keys for cfg in SITES[k]]
        Crawler(refresh=args.refresh, headless=not args.headed, limit=args.limit).run(configs)
    if args.command in ("build", "all"):
        print(json.dumps(build(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
