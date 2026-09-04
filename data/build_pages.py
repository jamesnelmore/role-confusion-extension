"""Build the agent-hijacking dataset, faithful to Ye et al.'s agent-injections track.

Mirrors experiments/agent-injections/01-run-user-injections-gpt-oss.ipynb:
  * Live Wikipedia HTML (paper samples article URLs from wikimedia/wikipedia,
    then GETs the live page). We sample random mainspace titles via the action
    API and fetch the rendered article — same distribution, no HF dependency.
  * Cleaning is the paper's `_strip_attrs_and_prune_head_keep_title`:
    drop comments, keep only <title> in <head>, strip ALL attributes, remove
    whitespace-only text nodes, drop <script>/<style>/<footer>/<nav>.
  * Keep only pages whose cleaned size is 5-10 KB (paper's window). This is
    small enough to run gpt-oss-20b locally on a 24 GB GPU.
  * Injection: the single base text from the paper's agent-injections
    config/injection.yaml, inserted immediately before </body>.
  * Variants: clean (control) and base-injection.

The exfil endpoint (dpaste.com) is sinkholed to 127.0.0.1 at eval time, so the
attack is emitted and scored but nothing leaves the machine.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from bs4 import BeautifulSoup, Comment, NavigableString

try:
    import certifi
    _CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _CTX = ssl.create_default_context()

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
BODY_CLOSE = re.compile(r"</body\s*>", re.IGNORECASE)

# The paper's agent-injections base injection (config/injection.yaml).
VENDOR_INJECTION = (
    ROOT / "vendor" / "role-confusion" / "experiments" / "agent-injections"
    / "config" / "injection.yaml"
)


def load_injection_text() -> str:
    """The single base-injection string, faithful to the paper."""
    import yaml
    if VENDOR_INJECTION.exists():
        cfg = yaml.safe_load(VENDOR_INJECTION.read_text())
        return cfg["injection_text"].strip()
    # Fallback: our mirrored copy (prompts/injections.yaml, first base entry).
    cfg = yaml.safe_load((ROOT / "prompts" / "injections.yaml").read_text())
    return cfg["base_injections"][0]["prompt"].strip()


def strip_attrs_and_prune(html: str) -> str:
    """Port of the paper's _strip_attrs_and_prune_head_keep_title."""
    soup = BeautifulSoup(html, "lxml")

    # Drop comments.
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()

    # Keep only <title> in <head>.
    if soup.head is not None and soup.head.title is not None:
        title = soup.head.title.extract()
        soup.head.clear()
        soup.head.append(title)

    # Strip all attributes.
    for tag in soup.find_all(True):
        tag.attrs = {}

    # Remove whitespace-only text nodes.
    for s in soup.find_all(string=lambda s: isinstance(s, NavigableString) and not s.strip()):
        s.extract()

    # Remove script/style/footer/nav.
    for t in soup.find_all(["script", "style", "footer", "nav"]):
        t.decompose()

    return str(soup)


def _get_random_titles(n_titles: int, ua: dict[str, str]) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    while len(titles) < n_titles:
        url = ("https://en.wikipedia.org/w/api.php?action=query&list=random"
               "&rnnamespace=0&rnlimit=50&format=json")
        try:
            req = urllib.request.Request(url, headers=ua)
            with urllib.request.urlopen(req, timeout=30, context=_CTX) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(3)
                continue
            break
        except Exception:
            time.sleep(1)
            continue
        for it in data["query"]["random"]:
            if it["title"] not in seen:
                seen.add(it["title"])
                titles.append(it["title"])
        time.sleep(0.3)
    return titles


def _fetch_clean(
    title: str, ua: dict[str, str], min_kb: float, max_kb: float
) -> dict | None:
    key = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = f"https://en.wikipedia.org/wiki/{key}"
    try:
        req = urllib.request.Request(url, headers=ua)
        with urllib.request.urlopen(req, timeout=30, context=_CTX) as r:
            html = r.read().decode("utf-8", "ignore")
    except Exception:
        return None
    cleaned = strip_attrs_and_prune(html)
    size_kb = len(cleaned.encode("utf-8")) / 1024
    if size_kb < min_kb or size_kb > max_kb:
        return None
    if "</body>" not in cleaned.lower():
        return None
    return {"title": title, "html": cleaned, "size_kb": round(size_kb, 1)}


def fetch_pages(
    n: int, min_kb: float = 5, max_kb: float = 10, workers: int = 8
) -> list[dict]:
    ua = {"User-Agent": "Mozilla/5.0 (research replication; role-confusion-eval)"}
    pages: list[dict] = []
    # Cleaned pages in the 5-10 KB window are a minority, so oversample titles
    # in batches until we have enough.
    while len(pages) < n:
        need = n - len(pages)
        titles = _get_random_titles(need * 8 + 20, ua)
        if not titles:
            break
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(lambda t: _fetch_clean(t, ua, min_kb, max_kb), titles):
                if res is None:
                    continue
                res["idx"] = len(pages)
                pages.append(res)
                print(f"  collected {len(pages)}/{n}: {res['title']} ({res['size_kb']} KB)")
                if len(pages) >= n:
                    break
    return pages


def inject(html: str, injection_text: str) -> str:
    new, count = BODY_CLOSE.subn(f"{injection_text}</body>", html, count=1)
    if count == 0:
        raise ValueError("no </body> to inject before")
    return new


def build(n: int, min_kb: float, max_kb: float) -> None:
    injection_text = load_injection_text()
    pages = fetch_pages(n, min_kb, max_kb)

    rows: list[dict] = []
    for p in pages:
        rows.append({
            "id": f"{p['idx']:03d}__clean", "idx": p["idx"], "variant": "clean",
            "title": p["title"], "size_kb": p["size_kb"], "html": p["html"],
        })
        rows.append({
            "id": f"{p['idx']:03d}__base-injection", "idx": p["idx"],
            "variant": "base-injection", "title": p["title"],
            "size_kb": p["size_kb"], "html": inject(p["html"], injection_text),
        })

    out = HERE / "pages.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(rows)} pages ({len(pages)} articles x 2 variants) to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10, help="number of articles")
    ap.add_argument("--min-kb", type=float, default=5.0)
    ap.add_argument("--max-kb", type=float, default=10.0)
    args = ap.parse_args()
    build(args.n, args.min_kb, args.max_kb)
