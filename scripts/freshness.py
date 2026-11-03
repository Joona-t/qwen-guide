#!/usr/bin/env python3
"""Freshness and scope check for the guide. Run: python3 -I scripts/freshness.py  (exit 0 = PASS)

Two kinds of stale text reached the live page (BUGS 2026-11-03): framing that said the guide was
"for Qwen3.5" while it also covers the 27B models and two GPUs, and facts that were settled or
contradicted after they were written. This fails the build on those patterns and reports every
dated claim with its age, so a re-check is due before a date is copied forward.

Rules:
  STALE   patterns that must never come back (each with the reason and what to write instead)
  DATED   every "checked / as of / current on <date>" claim; fails when older than MAX_AGE_DAYS
"""
import datetime, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FILES = [ROOT / "source/qwen-guide.md", ROOT / "scripts/build.py"]
MAX_AGE_DAYS = 45
TODAY = datetime.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else datetime.date.today()

STALE = [
    (r"(?i)written \*\*\w+ 2026\*\* for Qwen3\.5", "scopes the whole guide to Qwen3.5; it also covers Qwen3.6/3.8-27B and two GPUs"),
    (r"guide to Qwen3\.5 on your own machine", "hero scoped to Qwen3.5 only"),
    (r"useful line for a personal machine", "presents Qwen3.5 as the only line; name the tier (8-16 GB)"),
    (r"(?i)verified for Qwen3\.6-27B only", "Qwen3.8-27B has the identical config (research.md §B)"),
    (r"(?i)if (its|Qwen3\.8 keeps the same) layout", "the 3.8 layout is verified (research.md §B)"),
    (r"(?i)(quantized models|they) think (for )?longer", "contradicted by arXiv 2504.04823 (no longer outputs)"),
    (r"qwen-fine-tuning-guide", "the page lives at /qwen-guide/"),
    (r"Fine-tune Qwen locally, back to top", "the guide has four parts, not just fine-tuning"),
]
FULL = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
MONTHS = {m: i for i, m in enumerate(FULL, 1)} | {m[:3]: i for i, m in enumerate(FULL, 1)}
# historical citations that name a document's own date, not a freshness claim (exact substrings)
HISTORICAL = ["As of the note's 2026-04-22", "as of May 2026"]
NAMES = "|".join(sorted(MONTHS, key=len, reverse=True))
DATE = r"(\d{4}-\d{2}-\d{2}|\d{1,2} (?:%s) \d{4}|(?:%s) \d{4})" % (NAMES, NAMES)
DATED = re.compile(r"(?i)(checked|as of|current on|current in|re-checked|at the time of writing|snapshot, dated|unverified on|was current|version [\d.]+ in)\b[^.\n|]{0,40}?" + DATE)

def to_date(s):
    if re.match(r"\d{4}-", s): return datetime.date.fromisoformat(s)
    parts = s.split()
    if len(parts) == 3: return datetime.date(int(parts[2]), MONTHS[parts[1].capitalize()], int(parts[0]))
    return datetime.date(int(parts[1]), MONTHS[parts[0].capitalize()], 28)   # "October 2026" counts from the month's end

def main():
    fails, dated = [], []
    for f in FILES:
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for pat, why in STALE:
                if f.name == "freshness.py": continue
                if re.search(pat, line): fails.append(f"{f.name}:{n}: STALE ({why}): {line.strip()[:110]}")
            for m in DATED.finditer(line):
                if any(h in line for h in HISTORICAL) and any(h in m.group(0) for h in HISTORICAL): continue
                age = (TODAY - to_date(m.group(2))).days
                dated.append((age, f"{f.name}:{n}"))
                if age > MAX_AGE_DAYS: fails.append(f"{f.name}:{n}: DATED claim is {age} days old (> {MAX_AGE_DAYS}); re-check the source and re-date: {m.group(0)}")
    ages = sorted(a for a, _ in dated)
    print(f"dated claims: {len(ages)}; oldest {ages[-1] if ages else 0} days; median {ages[len(ages)//2] if ages else 0} days")
    print(f"stale-pattern hits: {sum('STALE' in x for x in fails)}")
    for x in fails: print(x)
    print("PASS" if not fails else f"FAIL ({len(fails)})")
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
