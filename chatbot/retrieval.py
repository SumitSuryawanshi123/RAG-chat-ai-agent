"""Dependency-free BM25 retrieval over the scraped knowledge base.

Design notes
------------
* Pure Python on purpose: the app is deployed on Vercel serverless functions where
  heavy ML dependencies (torch, sentence-transformers, FAISS...) blow the bundle
  size limit and cold-start budget. For a few thousand short chunks BM25 is fast,
  deterministic, needs no embedding API and works very well for this kind of
  entity-heavy corpus (project names, cities, prices).
* The index is built in memory at start-up in a few hundred milliseconds.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("chatbot.retrieval")

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could", "do", "does", "did", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or", "our", "please", "so", "tell", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "to", "us", "was", "we", "were", "what", "when", "where", "which", "who", "whom", "why", "will", "with", "would", "you", "your", "about", "give", "show", "list", "also", "any", "some", "more"]
)
# Normalise brand spellings so "Dar Global", "dar-global" and "DarGlobal" all match.
_ALIASES = (
    (re.compile(r"dar[\s\-]+global", re.I), "darglobal"),
    (re.compile(r"wasalt\.(sa|com)", re.I), "wasalt"),
    (re.compile(r"\bw\s?asalt\b", re.I), "wasalt"),
    (re.compile(r"\bproperties\b", re.I), "property"),
)
_SITE_HINTS = {
    "darglobal": ("darglobal", "trump", "aida", "pagani", "lamborghini", "elie saab", "missoni",
                  "fendi", "aston martin", "d-villas", "rayana", "amaya", "london-listed", "lse"),
    "wasalt": ("wasalt", "auction", "barter", "riyadh", "jeddah", "dammam", "makkah", "madinah",
               "saudi listings", "ejar"),
}
K1 = 1.4
B = 0.72

# Page-level priors. Long legal pages (privacy policy, T&Cs) are keyword-dense and
# would otherwise outrank real answers for general questions.
_LEGAL_URL = re.compile(r"terms|privacy|polic(y|ies)|conditions|cookie|disclaimer|slavery", re.I)
_LEGAL_QUERY = frozenset({"term", "condition", "privacy", "policy", "legal", "cookie", "gdpr", "disclaimer"})
_ABOUT_QUERY = re.compile(
    r"\b(what|who)\s+(is|are|was)\b|\bwhat'?s\b|\babout\b|\boverview\b|\bintroduc|\bdescribe\b|\bfounder\b|\bceo\b", re.I
)
_HUB_PATHS = frozenset({"", "/about", "/corporate", "/investor", "/why-invest", "/hospitality", "/commercial"})
LEGAL_PENALTY = 0.5
HUB_BOOST = 1.5
MIN_RELATIVE_SCORE = 0.4  # keep hits scoring at least 40% of the best hit
TITLE_BOOST = 0.8


def _is_hub(url: str) -> bool:
    path = re.sub(r"^https?://[^/]+", "", url).split("?")[0].rstrip("/")
    path = re.sub(r"^/en(?=/|$)", "", path)
    return path in _HUB_PATHS


def _stem(token: str) -> str:
    """A very small suffix stripper - enough to unify plural/verb forms."""
    if len(token) <= 3 or not token.isascii():
        return token
    for suffix in ("ingly", "ing", "edly", "ed", "ies", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            stem = token[: -len(suffix)]
            return stem + "y" if suffix == "ies" else stem
    return token


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    for pattern, repl in _ALIASES:
        text = pattern.sub(repl, text)
    out = []
    for tok in _TOKEN.findall(text.lower()):
        if drop_stopwords and tok in _STOPWORDS:
            continue
        out.append(_stem(tok))
    return out


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    site: str
    site_label: str
    url: str
    title: str
    heading: str
    text: str
    scraped_at: str


@dataclass(frozen=True, slots=True)
class Hit:
    chunk: Chunk
    score: float


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._doc_len: list[int] = []
        self._doc_tokens: list[frozenset[str]] = []
        self._title_tokens = [frozenset(tokenize(c.title)) for c in chunks]
        self._legal = [bool(_LEGAL_URL.search(c.url)) for c in chunks]
        self._hub = [_is_hub(c.url) for c in chunks]
        self._build()

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_jsonl(cls, path: Path) -> KnowledgeBase:
        if not path.exists():
            raise FileNotFoundError(
                f"Knowledge base not found at {path}. Run `python -m scraper all` to build it."
            )
        chunks = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    chunks.append(Chunk(**json.loads(line)))
        log.info("Loaded %d chunks from %s", len(chunks), path)
        return cls(chunks)

    def _build(self) -> None:
        for idx, c in enumerate(self.chunks):
            # Title/heading carry the most signal: weight them by repetition.
            tokens = tokenize(f"{c.title} {c.title} {c.heading} {c.heading} {c.text}")
            self._doc_len.append(len(tokens))
            self._doc_tokens.append(frozenset(tokens))
            for term, tf in Counter(tokens).items():
                self._postings[term].append((idx, tf))
        n = max(len(self.chunks), 1)
        self._avg_len = sum(self._doc_len) / n
        self._idf = {
            term: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for term, p in self._postings.items()
        }

    # ------------------------------------------------------------------- search
    def stats(self) -> dict:
        per_site = Counter(c.site for c in self.chunks)
        pages = {s: len({c.url for c in self.chunks if c.site == s}) for s in per_site}
        latest = max((c.scraped_at for c in self.chunks), default="")
        return {"chunks": len(self.chunks), "chunks_per_site": dict(per_site),
                "pages_per_site": pages, "last_scraped": latest}

    @staticmethod
    def _site_boosts(query: str) -> dict[str, float]:
        q = query.lower()
        boosts = {}
        for site, hints in _SITE_HINTS.items():
            if any(h in q for h in hints):
                boosts[site] = 1.35
        # If both are named (or none), don't favour either.
        return boosts if len(boosts) == 1 else {}

    def search(self, query: str, k: int = 6, *, max_per_page: int = 2) -> list[Hit]:
        terms = tokenize(query)
        if not terms or not self.chunks:
            return []
        q_counts = Counter(terms)
        scores: dict[int, float] = defaultdict(float)
        for term, qtf in q_counts.items():
            idf = self._idf.get(term)
            if idf is None:
                continue
            for idx, tf in self._postings[term]:
                norm = tf * (K1 + 1) / (tf + K1 * (1 - B + B * self._doc_len[idx] / self._avg_len))
                scores[idx] += idf * norm * (1 + 0.15 * (qtf - 1))

        if not scores:
            return []

        # Reward chunks that contain the query's consecutive word pairs (phrase match).
        bigrams = {f"{a} {b}" for a, b in zip(terms, terms[1:], strict=False)}
        boosts = self._site_boosts(query)
        wants_legal = bool(_LEGAL_QUERY & q_counts.keys())
        is_about = bool(_ABOUT_QUERY.search(query))
        for idx in list(scores):
            c = self.chunks[idx]
            if self._legal[idx] and not wants_legal:
                scores[idx] *= LEGAL_PENALTY
            if is_about and self._hub[idx]:
                scores[idx] *= HUB_BOOST
            if bigrams:
                hay = " ".join(tokenize(f"{c.title} {c.heading} {c.text}", drop_stopwords=True))
                scores[idx] *= 1 + 0.12 * sum(1 for bg in bigrams if bg in hay)
            scores[idx] *= boosts.get(c.site, 1.0)
            # Coverage: fraction of distinct query terms present.
            covered = sum(1 for t in q_counts if t in self._doc_tokens[idx]) / len(q_counts)
            scores[idx] *= 0.6 + 0.4 * covered
            # Users usually name the thing they want: reward pages whose *title* covers the query.
            title_cov = sum(1 for t in q_counts if t in self._title_tokens[idx]) / len(q_counts)
            if title_cov >= 0.5:
                scores[idx] *= 1 + TITLE_BOOST * title_cov

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        cutoff = ranked[0][1] * MIN_RELATIVE_SCORE  # drop weak tail matches: they only add noise
        hits: list[Hit] = []
        per_page: Counter[str] = Counter()
        for idx, score in ranked:
            if score < cutoff:
                break
            c = self.chunks[idx]
            if per_page[c.url] >= max_per_page:
                continue  # keep the context diverse
            per_page[c.url] += 1
            hits.append(Hit(c, score))
            if len(hits) >= k:
                break
        return hits
