"""Wikipedia pages with Ye et al.'s base injection inserted before </body>.

Cleaning matches the paper's `_strip_attrs_and_prune_head_keep_title`.
Injection text is vendor `experiments/agent-injections/config/injection.yaml`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import ssl
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import yaml
from bs4 import BeautifulSoup, Comment, NavigableString

try:
    import certifi
    _CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _CTX = ssl.create_default_context()

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
VENDOR_INJECTION = (
    ROOT / "vendor" / "role-confusion" / "experiments" / "agent-injections"
    / "config" / "injection.yaml"
)
BODY_CLOSE = re.compile(r"</body\s*>", re.IGNORECASE)


def strip_attrs_and_prune(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()
    if soup.head is not None and soup.head.title is not None:
        title = soup.head.title.extract()
        soup.head.clear()
        soup.head.append(title)
    for tag in soup.find_all(True):
        tag.attrs = {}
    for s in soup.find_all(string=lambda s: isinstance(s, NavigableString) and not s.strip()):
        s.extract()
    for t in soup.find_all(["script", "style", "footer", "nav"]):
        t.decompose()
    return str(soup)


def _get(url: str, ua: dict[str, str]) -> str | None:
    try:
        req = urllib.request.Request(url, headers=ua)
        with urllib.request.urlopen(req, timeout=30, context=_CTX) as r:
            return r.read().decode("utf-8", "ignore")
    except Exception:
        return None


def _random_titles(n: int, ua: dict[str, str]) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    while len(titles) < n:
        url = ("https://en.wikipedia.org/w/api.php?action=query&list=random"
               "&rnnamespace=0&rnlimit=50&format=json")
        raw = _get(url, ua)
        if not raw:
            time.sleep(1)
            continue
        for it in json.loads(raw)["query"]["random"]:
            if it["title"] not in seen:
                seen.add(it["title"])
                titles.append(it["title"])
        time.sleep(0.3)
    return titles


def _fetch_clean(title: str, ua: dict[str, str], min_kb: float, max_kb: float) -> dict | None:
    key = urllib.parse.quote(title.replace(" ", "_"), safe="")
    html = _get(f"https://en.wikipedia.org/wiki/{key}", ua)
    if html is None:
        return None
    cleaned = strip_attrs_and_prune(html)
    size_kb = len(cleaned.encode()) / 1024
    if not (min_kb <= size_kb <= max_kb) or "</body>" not in cleaned.lower():
        return None
    return {"title": title, "html": cleaned, "size_kb": round(size_kb, 1)}


def fetch_pages(n: int, min_kb: float, max_kb: float) -> list[dict]:
    ua = {"User-Agent": "Mozilla/5.0 (research replication; role-confusion-eval)"}
    pages: list[dict] = []
    while len(pages) < n:
        titles = _random_titles((n - len(pages)) * 8 + 20, ua)
        with ThreadPoolExecutor(max_workers=8) as ex:
            for res in ex.map(lambda t: _fetch_clean(t, ua, min_kb, max_kb), titles):
                if res is None:
                    continue
                res["idx"] = len(pages)
                pages.append(res)
                print(f"  {len(pages)}/{n}: {res['title']} ({res['size_kb']} KB)")
                if len(pages) >= n:
                    break
    return pages[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--min-kb", type=float, default=5.0)
    ap.add_argument("--max-kb", type=float, default=10.0)
    args = ap.parse_args()
    injection = yaml.safe_load(VENDOR_INJECTION.read_text())["injection_text"].strip()
    rows = []
    for p in fetch_pages(args.n, args.min_kb, args.max_kb):
        rows.append({**p, "id": f"{p['idx']:03d}__clean", "variant": "clean"})
        injected, n = BODY_CLOSE.subn(f"{injection}</body>", p["html"], count=1)
        if n == 0:
            raise ValueError(f"no </body> in {p['title']}")
        rows.append({**p, "id": f"{p['idx']:03d}__base-injection",
                     "variant": "base-injection", "html": injected})
    out = HERE / "pages.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows ({args.n} articles × 2) -> {out}")


if __name__ == "__main__":
    main()
