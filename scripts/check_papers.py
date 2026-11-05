#!/usr/bin/env python3
"""Check the reading list against arXiv. Run: python3 -I scripts/check_papers.py  (exit 0 = PASS; needs network)

Every link in the "papers" section must use the paper's exact title as its link text. A wrong ID
from memory nearly reached the page (BUGS 2026-11-05), and this catches that class of mistake.
Kept out of verify.mjs because that check is offline. Uses arXiv's free API, one request every 3 s.
"""
import pathlib, re, sys, time, urllib.request, xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parent.parent
ATOM = "{http://www.w3.org/2005/Atom}"

def norm(s): return re.sub(r"[^a-z0-9]", "", s.lower())

def main():
    md = (ROOT / "source/qwen-guide.md").read_text(encoding="utf-8")
    section = md.split('<!-- section id="papers"')[1].split("<!-- section ")[0]
    want = {i: t for t, i in re.findall(r"\[([^\]]+)\]\(https://arxiv\.org/abs/([0-9.]+)\)", section)}
    ids, fails = list(want), []
    for k in range(0, len(ids), 20):
        url = "https://export.arxiv.org/api/query?max_results=20&id_list=" + ",".join(ids[k:k + 20])
        req = urllib.request.Request(url, headers={"User-Agent": "qwen-guide-check/1.0"})
        root = ET.fromstring(urllib.request.urlopen(req, timeout=40).read())
        got = {e.find(ATOM + "id").text.split("/abs/")[-1].split("v")[0]: " ".join(e.find(ATOM + "title").text.split()) for e in root.findall(ATOM + "entry")}
        for i in ids[k:k + 20]:
            if i not in got: fails.append(f"{i}: not found on arXiv")
            elif norm(got[i]) != norm(want[i]): fails.append(f"{i}: guide says '{want[i]}', arXiv says '{got[i]}'")
        time.sleep(3.1)
    print(f"papers checked: {len(ids)}")
    for f in fails: print(f)
    print("PASS" if not fails else f"FAIL ({len(fails)})")
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
