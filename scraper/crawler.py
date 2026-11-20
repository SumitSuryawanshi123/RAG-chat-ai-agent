"""Polite Playwright crawler.

Why a real browser? Both targets sit behind bot protection (Incapsula for
DarGlobal, Cloudflare for Wasalt) and render a lot client-side, so plain
``requests`` calls only ever get a challenge page.

Politeness: robots.txt is honoured (including wildcard rules and crawl-delay),
requests are rate limited per site, images/fonts/media are never downloaded and
only publicly reachable pages are fetched (no logins, no forms submitted).
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import logging
import re
import time
from collections import deque
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from playwright.sync_api import BrowserContext, Page, sync_playwright
from playwright.sync_api import Error as PlaywrightError
from protego import Protego

from .config import RAW_DIR, USER_AGENT, SiteConfig, SitemapSource
from .extract import extract_links, extract_page, normalize_url

log = logging.getLogger("scraper.crawler")

_CHALLENGE_TITLES = ("just a moment", "attention required", "access denied", "pardon our interruption")
_BLOCKED_RESOURCES = {"image", "media", "font"}
MIN_TEXT_CHARS = 80
MAX_CONSECUTIVE_FAILURES = 8


def _slug(url: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9]+", "-", url.split("://", 1)[-1]).strip("-").lower()[:70]
    return f"{stem}-{hashlib.sha1(url.encode()).hexdigest()[:8]}"


def _decode_body(body: bytes) -> str:
    if body[:2] == b"\x1f\x8b":  # raw gzip (e.g. *.xml.gz sitemaps)
        body = gzip.decompress(body)
    return body.decode("utf-8", "replace")


class Crawler:
    def __init__(self, *, refresh: bool = False, headless: bool = True, limit: int | None = None):
        self.refresh = refresh
        self.headless = headless
        self.limit = limit
        self.stats = {"fetched": 0, "cached": 0, "skipped": 0, "failed": 0}
        self._browser = None

    # ------------------------------------------------------------------ public
    def run(self, configs: list[SiteConfig]) -> dict[str, int]:
        with sync_playwright() as pw:
            browser = self._browser = pw.chromium.launch(headless=self.headless)
            try:
                for cfg in configs:
                    self._crawl_site(browser, cfg)
            finally:
                browser.close()
        log.info("Crawl finished: %s", self.stats)
        return self.stats

    # --------------------------------------------------------------- internals
    def _new_context(self, browser) -> BrowserContext:
        ctx = browser.new_context(
            user_agent=USER_AGENT,
            locale="en-US",
            viewport={"width": 1366, "height": 900},
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )
        ctx.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in _BLOCKED_RESOURCES
            else route.continue_(),
        )
        ctx.set_default_navigation_timeout(45_000)
        return ctx

    def _crawl_site(self, browser, cfg: SiteConfig) -> None:
        log.info("=== %s (%s) ===", cfg.label, ", ".join(cfg.hosts))
        ctx = self._new_context(browser)
        shared_page = ctx.new_page()

        @contextmanager
        def session():
            """Yield a page. Some sites challenge every navigation after the first
            in a browser session, so those get a fresh, stateless context per page."""
            if not cfg.fresh_context_per_page:
                yield shared_page
                return
            page_ctx = self._new_context(browser)
            try:
                yield page_ctx.new_page()
            finally:
                page_ctx.close()

        try:
            # Warm-up: lets the bot-protection layer set its cookies on the context.
            with session() as warm:
                self._goto(warm, cfg.seeds[0])
            robots = self._load_robots(ctx, cfg)
            delay = max(cfg.delay_seconds, (robots.crawl_delay(USER_AGENT) or 0) if robots else 0)

            queue: deque[tuple[str, int]] = deque()
            seen: set[str] = set()

            def enqueue(url: str, depth: int) -> bool:
                norm = normalize_url(url)
                if not norm or norm in seen:
                    return False
                host = re.sub(r"^https?://([^/]+).*$", r"\1", norm)
                if not cfg.allows_host(host) or re.search(cfg.exclude, norm):
                    return False
                if robots and not robots.can_fetch(norm, USER_AGENT):
                    log.info("robots.txt disallows %s", norm)
                    self.stats["skipped"] += 1
                    return False
                seen.add(norm)
                queue.append((norm, depth))
                return True

            for seed in cfg.seeds:
                enqueue(seed, 0)
            for source in cfg.sitemaps:
                for url in self._sitemap_urls(ctx, source):
                    enqueue(url, 0)
            log.info("Queued %d start URLs for %s", len(queue), cfg.label)

            done = 0
            consecutive_failures = 0
            max_pages = min(cfg.max_pages, self.limit) if self.limit else cfg.max_pages
            while queue and done < max_pages:
                url, depth = queue.popleft()
                record, from_cache = self._fetch_or_load(session, url, cfg)
                if record is None:
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        log.error(
                            "%d failures in a row on %s - the site is refusing us; aborting this site.",
                            consecutive_failures, cfg.label,
                        )
                        break
                    time.sleep(delay)  # stay polite even when requests fail
                    continue
                consecutive_failures = 0
                done += 1
                if not from_cache:
                    time.sleep(delay)
                if depth < cfg.max_depth and cfg.follow:
                    for link in record.get("links", []):
                        if re.search(cfg.follow, link):
                            enqueue(link, depth + 1)
                if done % 10 == 0:
                    log.info("%s: %d pages done, %d queued", cfg.label, done, len(queue))
            log.info("%s: stored %d pages", cfg.label, done)
        finally:
            ctx.close()

    def _api_get(self, ctx: BrowserContext, url: str, timeout: int = 60_000) -> tuple[int, bytes]:
        """HTTP GET for non-HTML resources (robots.txt, sitemaps) -> (status, body).

        Runs in a *separate* throw-away context so the request client's cookies never
        leak into the browsing session. Cookies only flow browser -> API client
        (DarGlobal's Incapsula layer wants them even for the sitemap).
        """
        api = self._browser.new_context(user_agent=USER_AGENT)
        try:
            api.add_cookies(ctx.cookies())
            resp = api.request.get(url, timeout=timeout)
            return resp.status, resp.body()  # read before the context is disposed
        finally:
            api.close()

    def _load_robots(self, ctx: BrowserContext, cfg: SiteConfig) -> Protego | None:
        try:
            status, body = self._api_get(ctx, cfg.robots_url, timeout=30_000)
            text = body.decode("utf-8", "replace")
            if status == 200 and "<html" not in text[:200].lower():
                log.info("Loaded robots.txt from %s", cfg.robots_url)
                return Protego.parse(text)
            log.warning("robots.txt at %s returned %s - assuming allowed", cfg.robots_url, status)
        except PlaywrightError as exc:
            log.warning("Could not fetch robots.txt (%s) - assuming allowed", exc)
        return None

    def _sitemap_urls(self, ctx: BrowserContext, source: SitemapSource) -> list[str]:
        try:
            status, raw = self._api_get(ctx, source.url)
            if status != 200:
                log.warning("Sitemap %s -> HTTP %s", source.url, status)
                return []
            body = _decode_body(raw)
        except (PlaywrightError, OSError) as exc:
            log.warning("Sitemap %s failed: %s", source.url, exc)
            return []
        urls = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", body)
        if source.include:
            urls = [u for u in urls if re.search(source.include, u)]
        if source.max_urls and len(urls) > source.max_urls:
            if source.sample_evenly:
                step = len(urls) / source.max_urls
                urls = [urls[int(i * step)] for i in range(source.max_urls)]
            else:
                urls = urls[: source.max_urls]
        log.info("Sitemap %s -> %d URLs", source.url, len(urls))
        return urls

    def _fetch_or_load(self, session, url: str, cfg: SiteConfig) -> tuple[dict | None, bool]:
        path = RAW_DIR / cfg.key / f"{_slug(url)}.json"
        if path.exists() and not self.refresh:
            self.stats["cached"] += 1
            return json.loads(path.read_text(encoding="utf-8")), True

        with session() as page:
            html = self._goto(page, url, retries=2)
            final_url = page.url
        if html is None:
            self.stats["failed"] += 1
            return None, False

        record = extract_page(html, final_url)
        if len(record["text"]) < MIN_TEXT_CHARS:
            log.warning("Thin page (%d chars), skipping: %s", len(record["text"]), url)
            self.stats["skipped"] += 1
            return None, False

        record.update(
            site=cfg.key,
            site_label=cfg.label,
            requested_url=url,
            scraped_at=datetime.now(UTC).isoformat(timespec="seconds"),
            links=[link for link in extract_links(html, final_url)
                   if cfg.allows_host(re.sub(r"^https?://([^/]+).*$", r"\1", link))],
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        self.stats["fetched"] += 1
        log.info("OK  %-6d chars  %s", len(record["text"]), url)
        return record, False

    def _goto(self, page: Page, url: str, retries: int = 1) -> str | None:
        for attempt in range(1, retries + 2):
            try:
                resp = page.goto(url, wait_until="domcontentloaded")
                status = resp.status if resp is not None else 200
                if status in (404, 410):
                    log.warning("HTTP %s for %s", status, url)
                    return None
                # Bot-protection challenges are served with 403/503 and clear themselves
                # via JavaScript, so wait for that before judging the response.
                if not self._wait_for_real_page(page):
                    raise PlaywrightError(f"bot challenge not cleared (HTTP {status})")
                if status >= 400 and self._looks_like_error_page(page):
                    log.warning("HTTP %s for %s", status, url)
                    return None
                # Chatty analytics scripts keep the network busy; timing out is fine.
                with contextlib.suppress(PlaywrightError):
                    page.wait_for_load_state("networkidle", timeout=5_000)
                self._scroll(page)
                return page.content()
            except PlaywrightError as exc:
                log.warning("Attempt %d/%d failed for %s: %s", attempt, retries + 1, url, str(exc)[:160])
                time.sleep(2 * attempt)
        return None

    @staticmethod
    def _wait_for_real_page(page: Page, timeout: float = 15.0) -> bool:
        """Wait out JS challenges (Cloudflare / Incapsula). True once a real page shows."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                title = (page.title() or "").lower()
                html_head = page.content()[:1500]
            except PlaywrightError:  # navigation in flight while the challenge redirects
                page.wait_for_timeout(500)
                continue
            challenged = any(t in title for t in _CHALLENGE_TITLES) or "_Incapsula_Resource" in html_head
            if not challenged:
                return True
            page.wait_for_timeout(1_000)
        log.warning("Bot challenge did not clear for %s", page.url)
        return False

    @staticmethod
    def _looks_like_error_page(page: Page) -> bool:
        title = (page.title() or "").lower()
        return any(t in title for t in ("404", "not found", "error", "forbidden"))

    @staticmethod
    def _scroll(page: Page, steps: int = 5) -> None:
        """Scroll so lazy-loaded sections render before we read the DOM."""
        try:
            for _ in range(steps):
                page.mouse.wheel(0, 1600)
                page.wait_for_timeout(250)
            page.evaluate("window.scrollTo(0, 0)")
        except PlaywrightError:
            pass


def count_raw_pages(raw_dir: Path = RAW_DIR) -> dict[str, int]:
    return {d.name: len(list(d.glob("*.json"))) for d in raw_dir.iterdir() if d.is_dir()} if raw_dir.exists() else {}
