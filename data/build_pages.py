"""Build the agent-hijacking dataset, faithful to Ye et al. §B.2.

Differences from the paper are only where safety requires it, and are noted:
  * Pages are real Wikipedia HTML (paper: wikimedia/wikipedia 20231101.en,
    classes stripped, <100KB). We fetch live via the Wikipedia REST API — same
    distribution, no HF dependency.
  * Injections are their EXACT templates from prompts/injections.yaml, inserted
    immediately before </body>, one chosen at random per page (their
    create_injection_function).
  * Variants: base-injection (plain command) and cot-forgery-injection (command
    + absurd-trigger + gpt-oss-styled forged policy CoT). Plus a clean control.

The exfil endpoint in the injections is dpaste.com/api (the paper's). It never
resolves inside the sandbox (--network none), so the attack is attempted but
cannot succeed — which is what the judge scores.
"""

import argparse
import json
import pathlib
import random
import re
import time
import ssl
import urllib.error
import urllib.parse
import urllib.request

try:
    import certifi
    _CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _CTX = ssl.create_default_context()

HERE = pathlib.Path(__file__).parent
BODY_CLOSE = re.compile(r"</body\s*>", re.IGNORECASE)


def strip_classes(html):
    return re.sub(r'\s+class="[^"]*"', "", html)


def _get_random_titles(n_titles, ua):
    """Bulk random mainspace titles via the action API (50 per request)."""
    import urllib.request
    titles, seen = [], set()
    while len(titles) < n_titles:
        url = ("https://en.wikipedia.org/w/api.php?action=query&list=random"
               "&rnnamespace=0&rnlimit=50&format=json")
        try:
            req = urllib.request.Request(url, headers=ua)
            with urllib.request.urlopen(req, timeout=30, context=_CTX) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(3); continue
            break
        except Exception:
            time.sleep(1); continue
        for it in data["query"]["random"]:
            if it["title"] not in seen:
                seen.add(it["title"]); titles.append(it["title"])
        time.sleep(0.3)
    return titles


def _fetch_one(title, ua, max_kb):
    """Fetch a normal article pageview (less rate-limited than the APIs)."""
    import urllib.parse
    import urllib.request
    key = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = f"https://en.wikipedia.org/wiki/{key}"
    try:
        req = urllib.request.Request(url, headers=ua)
        with urllib.request.urlopen(req, timeout=30, context=_CTX) as r:
            html = r.read().decode("utf-8", "ignore")
    except Exception:
        return None
    html = strip_classes(html)
    if len(html.encode()) / 1024 > max_kb or "</body>" not in html.lower():
        return None
    return {"title": title, "html": html}


def fetch_random_wikipedia(n, max_kb=100, seed=1234, workers=8):
    """n random Wikipedia article HTMLs, fetched concurrently. Robust to the
    machine napping between calls and to per-request rate limits."""
    from concurrent.futures import ThreadPoolExecutor
    ua = {"User-Agent": "Mozilla/5.0 (research replication; role-confusion-eval)"}
    titles = _get_random_titles(n * 2 + 20, ua)
    print(f"  got {len(titles)} candidate titles; fetching HTML with {workers} workers")
    pages = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(lambda t: _fetch_one(t, ua, max_kb), titles):
            if res is None:
                continue
            res["idx"] = len(pages)
            pages.append(res)
            print(f"  collected {len(pages)}/{n}: {res['title']}")
            if len(pages) >= n:
                break
    return pages


def inject(html, prompt):
    new, count = BODY_CLOSE.subn(f"{prompt}</body>", html, count=1)
    if count == 0:
        raise ValueError("no </body> to inject before")
    return new


def build(n, seed=1234):
    import yaml
    cfg = yaml.safe_load((HERE.parent / "prompts" / "injections.yaml").read_text())
    base = [i["prompt"].strip() for i in cfg["base_injections"]]
    forgery = [i["prompt"].strip() for i in cfg["prompt_injections"]]

    raw = fetch_random_wikipedia(n, seed=seed)
    random.seed(seed)
    rows = []
    for p in raw:
        rows.append({"id": f"{p['idx']:03d}__clean", "idx": p["idx"],
                     "variant": "clean", "title": p["title"], "html": p["html"]})
        rows.append({"id": f"{p['idx']:03d}__base-injection", "idx": p["idx"],
                     "variant": "base-injection", "title": p["title"],
                     "html": inject(p["html"], random.choice(base))})
        rows.append({"id": f"{p['idx']:03d}__cot-forgery-injection", "idx": p["idx"],
                     "variant": "cot-forgery-injection", "title": p["title"],
                     "html": inject(p["html"], random.choice(forgery))})
    out = HERE / "pages.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(rows)} pages ({len(raw)} articles x 3 variants) to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10, help="number of articles")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    build(args.n, args.seed)
