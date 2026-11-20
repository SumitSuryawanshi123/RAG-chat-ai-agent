"""Crawl configuration for the two target websites.

Everything that is site specific lives here so the crawler itself stays generic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
KB_PATH = ROOT / "data" / "knowledge_base.jsonl"

# A stock desktop-browser UA: the targets' bot protection rejects UAs containing
# "bot"/"crawler" outright. We still behave politely (robots.txt, rate limiting).
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Query parameters that never change page content.
TRACKING_PARAMS = re.compile(
    r"^(utm_.*|sfmc_.*|gclid|fbclid|ltclid|trk|dealercode|ref|source)$", re.I
)

# Never fetch these (binary / non-HTML assets).
SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".mp4",
    ".mov", ".zip", ".xml", ".gz", ".css", ".js", ".json", ".woff", ".woff2",
)


@dataclass(frozen=True)
class SitemapSource:
    """A sitemap to mine for URLs, optionally filtered and capped."""

    url: str
    include: str | None = None  # regex a URL must match
    max_urls: int | None = None  # cap after filtering
    sample_evenly: bool = False  # take an evenly spaced sample instead of the head


@dataclass
class SiteConfig:
    key: str  # short id used in the knowledge base ("darglobal" / "wasalt")
    label: str  # human friendly name
    hosts: tuple[str, ...]  # hosts we are allowed to crawl
    robots_url: str
    seeds: list[str]
    sitemaps: list[SitemapSource] = field(default_factory=list)
    # Links discovered while crawling are followed only if they match this.
    follow: str | None = None
    exclude: str = r"$^"  # regex of URLs to drop (default matches nothing)
    max_pages: int = 300
    delay_seconds: float = 1.2
    max_depth: int = 1
    # Cloudflare challenges every navigation after the first within one browser
    # session, so for such sites each page gets a fresh, stateless context.
    fresh_context_per_page: bool = False

    def allows_host(self, host: str) -> bool:
        host = host.lower()
        return any(host == h or host.endswith("." + h) for h in self.hosts)


DARGLOBAL = SiteConfig(
    key="darglobal",
    label="DarGlobal",
    hosts=("darglobal.co.uk",),
    robots_url="https://darglobal.co.uk/robots.txt",
    seeds=[
        "https://darglobal.co.uk/",
        "https://darglobal.co.uk/about",
        "https://darglobal.co.uk/projects",
        "https://darglobal.co.uk/investor",
        "https://darglobal.co.uk/why-invest",
        "https://darglobal.co.uk/faq",
        "https://darglobal.co.uk/get-in-touch",
        "https://darglobal.co.uk/hospitality",
        "https://darglobal.co.uk/commercial",
        "https://darglobal.co.uk/luxury-golf-communities",
        "https://darglobal.co.uk/exclusive-membership",
        "https://darglobal.co.uk/one-of-one",
        "https://darglobal.co.uk/press",
    ],
    sitemaps=[
        # Every top-level page, project page, partner page, press release and insight...
        SitemapSource(
            url="https://darglobal.co.uk/sitemap.xml",
            include=r"^https://darglobal\.co\.uk(/[^/]+)?/?$"
            r"|/(projects|partners|one-of-one|press|insights|rayana|amaya|trump-plaza)/",
        ),
        # ...plus a bounded sample of the (very large) blog.
        SitemapSource(
            url="https://darglobal.co.uk/sitemap.xml",
            include=r"/blog/",
            max_urls=40,
            sample_evenly=True,
        ),
    ],
    follow=r"^https://darglobal\.co\.uk/(projects|partners|press|insights|one-of-one)(/|$)",
    exclude=(
        r"/(thank-you|success|pay-online|new-home-page|landing-page|landingpage|"
        r"internship-application|bookYourUnit|broker-contest-terms-conditions|"
        r"win-a-trip-to-dubai|sohar-islamic)(/|$)"
    ),
    max_pages=240,
    delay_seconds=1.0,
)

WASALT = SiteConfig(
    key="wasalt",
    label="Wasalt",
    hosts=("wasalt.sa", "wasalt.com"),
    robots_url="https://wasalt.sa/robots.txt",
    seeds=[
        "https://wasalt.sa/en",
        "https://wasalt.sa/en/corporate",
        "https://wasalt.sa/en/s/wasalt-business",
        "https://wasalt.sa/en/s/masterplan",
        "https://wasalt.sa/en/s/terms-and-conditions",
        "https://wasalt.sa/en/s/terms-of-use",
        "https://wasalt.sa/en/s/privacy-policy",
        "https://wasalt.sa/en/s/terms-of-advertising",
        "https://wasalt.sa/en/s/terms-n-conditions-for-auction",
        "https://wasalt.sa/en/s/terms-n-conditions-for-ewallet",
    ],
    sitemaps=[
        # A representative, evenly spaced sample of real property listings.
        SitemapSource(
            url="https://cdn.wasalt.sa/sitemap/product_sitemap_en_sa.xml.gz",
            include=r"/en/property/",
            max_urls=40,
            sample_evenly=True,
        ),
        # A sample of area/category landing pages (price ranges, counts, listings).
        SitemapSource(
            url="https://cdn.wasalt.sa/sitemap/category_sitemap_en_sa.xml.gz",
            include=r"/en/(properties|apartments|villas|lands)-for-(sale|rent)-in-[a-z-]+$",
            max_urls=40,
            sample_evenly=True,
        ),
    ],
    # Home page links to the main city/category pages; blog & help live on subdomains.
    follow=(
        r"^https://wasalt\.sa/en/((properties|apartments|villas|lands)-for-(sale|rent)-in-[a-z-]+"
        r"|s/[a-z-]+|corporate|about[a-z-]*|projects?/[^?]+)$"
    ),
    exclude=r"/(login|register|search)(/|\?|$)|/ar(/|$)",
    max_pages=120,
    delay_seconds=1.2,
    fresh_context_per_page=True,
)

# Extra seed pages on Wasalt sub-domains (blog, auctions). Crawled via a small
# secondary config so the rules above stay strict.
WASALT_BLOG = SiteConfig(
    key="wasalt",
    label="Wasalt",
    hosts=("blog.wasalt.sa", "auction.wasalt.sa"),
    robots_url="https://blog.wasalt.sa/robots.txt",
    seeds=[
        "https://blog.wasalt.sa/en/",
        "https://auction.wasalt.sa/en",
    ],
    # URLs are normalised without trailing slashes, so match a single path segment.
    follow=r"^https://blog\.wasalt\.sa/en/[^/?#]+$",
    exclude=r"/(tag|author|category|page|feed|wp-json|wp-admin)(/|$)|\?",
    max_pages=25,
    delay_seconds=1.5,
    fresh_context_per_page=True,
)

SITES: dict[str, list[SiteConfig]] = {
    "darglobal": [DARGLOBAL],
    "wasalt": [WASALT, WASALT_BLOG],
}
