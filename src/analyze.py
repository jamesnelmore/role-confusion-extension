"""Summarize role-confusion eval logs: scored-only ASR and matched flips.

Merges every *.eval under each condition dir (so a resume log can fill holes).
NA / missing scores are excluded from ASR and from flip counts — they are
truncated runs (usually CUDA OOM mid-attack), not safe refusals.
"""

from __future__ import annotations

import glob
import pathlib
from collections import Counter

from inspect_ai.log import read_eval_log

ROOT = pathlib.Path(__file__).parent.parent
# Later dirs win on a scored label; listed so a rerun can supersede the
# OOM-truncated pilot without deleting it.
CONDITIONS: dict[str, list[str]] = {
    "off / injected": ["logs/full_off_inj", "logs/rerun_off_inj"],
    "project / injected": ["logs/full_project_inj", "logs/rerun_project_inj"],
    "off / clean": ["logs/full_off_clean"],
    "project / clean": ["logs/full_project_clean"],
}
ATTACK = {"ATTACK_ATTEMPT", "ATTACK_SUCCESS"}
VALID = ATTACK | {"SAFE_DENIAL", "SAFE_SILENT"}


def _load(dirpaths: list[str]) -> dict[str, str] | None:
    """Merge samples across eval files. Scored labels overwrite NA."""
    rows: dict[str, str] = {}
    found = False
    for d in dirpaths:
        files = sorted(glob.glob(str(ROOT / d / "*.eval")))
        for f in files:
            found = True
            log = read_eval_log(f)
            for s in (log.samples or []):
                sc = s.scores.get("exfil_asr") if s.scores else None
                label = sc.answer if sc else "NA"
                prev = rows.get(str(s.id))
                if prev in VALID and label not in VALID:
                    continue
                rows[str(s.id)] = label
    return rows if found else None


def _scored(rows: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in rows.items() if v in VALID}


def main() -> None:
    print(f"{'condition':22} {'N':>3} {'ASR':>6}   label breakdown (scored only)")
    print("-" * 78)
    per_cond: dict[str, dict[str, str]] = {}
    for name, dirs in CONDITIONS.items():
        rows = _load(dirs)
        if rows is None:
            print(f"{name:22}  --   (no log yet)")
            continue
        scored = _scored(rows)
        per_cond[name] = scored
        n = len(scored)
        na = sum(1 for v in rows.values() if v not in VALID)
        asr = sum(l in ATTACK for l in scored.values()) / n if n else 0
        brk = ", ".join(f"{k}:{v}" for k, v in sorted(Counter(scored.values()).items()))
        extra = f"   (plus {na} NA/unscored, excluded)" if na else ""
        print(f"{name:22} {n:>3} {asr:>6.3f}   {brk}{extra}")

    inj_off = per_cond.get("off / injected")
    inj_prj = per_cond.get("project / injected")
    if inj_off and inj_prj:
        both = sorted(set(inj_off) & set(inj_prj))
        print(f"\nMatched per-page (injected, both scored): off -> project  N={len(both)}")
        print("-" * 78)
        flips_to_attack = flips_to_safe = 0
        off_att = prj_att = 0
        for pid in both:
            o = inj_off[pid]
            p = inj_prj[pid]
            oa, pa = o in ATTACK, p in ATTACK
            off_att += int(oa)
            prj_att += int(pa)
            mark = ""
            if not oa and pa:
                mark = "  <== projection CAUSED attack"
                flips_to_attack += 1
            elif oa and not pa:
                mark = "  <== projection PREVENTED attack"
                flips_to_safe += 1
            print(f"  page {pid.split('__')[0]}:  {o:14} -> {p:14}{mark}")
        n = len(both)
        if n:
            print(f"\n  matched ASR   off {off_att}/{n}={off_att/n:.3f}   "
                  f"project {prj_att}/{n}={prj_att/n:.3f}")
            print(f"  projection caused  {flips_to_attack} new attacks")
            print(f"  projection stopped {flips_to_safe} attacks")
            # McNemar without continuity correction; N is usually small.
            disc = flips_to_attack + flips_to_safe
            if disc:
                chi2 = (flips_to_safe - flips_to_attack) ** 2 / disc
                print(f"  McNemar chi2 (b-c)^2/(b+c) = {chi2:.3f}  "
                      f"(4 vs 2 on N=7 is ~0.67; need ~3.84 for p<.05)")


if __name__ == "__main__":
    main()
