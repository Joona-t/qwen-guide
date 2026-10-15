#!/usr/bin/env python3
"""Build index.html from source/qwen-fine-tuning-guide.md (the source of truth).

The Markdown stays readable on GitHub; HTML comments carry the build hints:
  <!-- section id="x" kicker="Part 1 · Foundations" -->   starts a <section>
  <!-- widget:name -->                                    inserts template/widgets/name.html
Run: python3 scripts/build.py          (writes index.html)
     python3 scripts/build.py --check  (exit 1 if index.html is stale)
"""
import html, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
T = ROOT / "template"

def inline(t):
    codes = []
    def keep(m):
        codes.append(html.escape(m.group(1), quote=False)); return f"\x00{len(codes)-1}\x00"
    t = re.sub(r"`([^`]+)`", keep, t)
    t = html.escape(t, quote=False)
    t = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", lambda m: f'<a href="{m.group(2)}" rel="noopener">{m.group(1)}</a>', t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", t)
    return re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{codes[int(m.group(1))]}</code>", t)

def callout(text):
    t = text.strip()
    kinds = {"⚠": ("gotcha", "⚠ Gotcha"), "◆": ("caveat", "◆ Caveat"), "📝": ("note", "📝 Note")}
    cls, tag = kinds.get(t[:1], ("note", "📝 How this guide was checked"))
    m = re.match(r"^(⚠|◆|📝)\s*\*\*(.+?)\*\*\s*(.*)$", t, re.S)
    if m:
        tag, body = f"{m.group(1)} {m.group(2).rstrip('.')}", m.group(3)
    else:
        body = t
    return f'<div class="callout callout--{cls}"><div class="ctag">{inline(tag)}</div><p>{inline(body)}</p></div>'

def code(lang, lines):
    return '<div class="code"><button class="copy">Copy</button><pre>' + html.escape("\n".join(lines), quote=False) + "</pre></div>"

def blocks(lines):
    out, i = [], 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("```"):
            j = i + 1
            while not lines[j].startswith("```"): j += 1
            out.append(code(l[3:], lines[i+1:j])); i = j + 1; continue
        m = re.match(r"^<!-- widget:(\w+) -->$", l.strip())
        if m:
            out.append((T / "widgets" / f"{m.group(1)}.html").read_text(encoding="utf-8").strip()); i += 1; continue
        if l.startswith("<!--"): i += 1; continue
        if l.startswith("### "): out.append(f"<h3>{inline(l[4:])}</h3>"); i += 1; continue
        if l.startswith("> "):
            buf = []
            while i < len(lines) and lines[i].startswith(">"): buf.append(lines[i][1:].strip()); i += 1
            out.append(callout(" ".join(buf))); continue
        if l.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")]); i += 1
            th = "".join(f"<th>{inline(c)}</th>" for c in rows[0])
            tb = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows[2:])
            out.append(f'<div class="tablewrap"><table><thead><tr>{th}</tr></thead><tbody>{tb}</tbody></table></div>'); continue
        lm = re.match(r"^(- |\d+\. )", l)
        if lm:
            ordered = lm.group(1) != "- "
            items = []
            while i < len(lines) and (re.match(r"^(- |\d+\. )", lines[i]) or (lines[i].startswith("  ") and items)):
                if re.match(r"^(- |\d+\. )", lines[i]):
                    items.append([re.sub(r"^(- |\d+\. )", "", lines[i])])
                else:
                    items[-1].append(lines[i][2:])
                i += 1
            lis = []
            for it in items:
                first, rest = it[0], it[1:]
                inner = inline(first) + ("\n" + blocks(rest) if rest else "")
                lis.append(f"<li>{inner}</li>")
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(lis) + f"</{tag}>"); continue
        if not l.strip(): i += 1; continue
        buf = []
        while i < len(lines) and lines[i].strip() and not re.match(r"^(```|### |> |\||- |\d+\. |<!--)", lines[i]):
            buf.append(lines[i].strip()); i += 1
        out.append(f"<p>{inline(' '.join(buf))}</p>")
    return "\n      ".join(out)

def build():
    md = (ROOT / "source/qwen-fine-tuning-guide.md").read_text(encoding="utf-8")
    title = re.match(r"^# (.+)$", md, re.M).group(1)
    parts = re.split(r"^<!-- section (.*?) -->\n", md, flags=re.M)
    intro = parts[0].split("\n", 1)[1]
    sections, toc = [], []
    intro_html = blocks(intro.split("\n"))
    sections.append(f'<section id="start">\n      <p class="kicker">Start here</p>\n      <h2>Before you begin</h2>\n      {intro_html}\n    </section>')
    toc.append('<a href="#start">Start here</a>')
    for attrs, body in zip(parts[1::2], parts[2::2]):
        a = dict(re.findall(r'(\w+)="([^"]*)"', attrs))
        h2, rest = body.split("\n", 1)
        h2 = re.sub(r"^## ", "", h2)
        sections.append(f'<section id="{a["id"]}">\n      <p class="kicker">{html.escape(a["kicker"])}</p>\n      <h2>{inline(h2)}</h2>\n      {blocks(rest.split(chr(10)))}\n    </section>')
        toc.append(f'<a href="#{a["id"]}">{html.escape(a.get("toc", h2))}</a>')
    mascot = (T / "mascot.datauri").read_text(encoding="utf-8").strip()
    desc = "A first-principles, offline guide to fine-tuning Qwen3.5 locally: LoRA on a Mac with mlx-lm or an NVIDIA PC with Unsloth, with interactive calculators, honest memory limits and an evaluation-first workflow. No paid API."
    head = (T / "head.html").read_text(encoding="utf-8").replace("{{TITLE}}", "Qwen Fine-Tuning — local LoRA guide").replace("{{DESCRIPTION}}", desc)
    head = head.replace("</style>", (T / "widgets.css").read_text(encoding="utf-8") + "</style>", 1)
    theme = (T / "theme-menu.html").read_text(encoding="utf-8")
    hero = f'''<header class="hero">
  <div class="hero-sparkles" aria-hidden="true"><span class="spark">✦</span><span class="spark">♥</span><span class="spark">✦</span></div>
  <div class="hero-inner">
    <div class="topbar">
      <div class="brand-mark">
        <img src="{mascot}" alt="Sparky, the LoveSpark wolf mascot">
        <span class="heart-dot" aria-hidden="true">♥</span>
        <div class="lockup"><b>LoveSpark</b><span>Local Qwen fine-tuning guide</span></div>
      </div>
      {theme}
    </div>
    <div class="badge-row">
      <span class="pill brand">Qwen3.5 × LoRA</span>
      <span class="pill ok">● 100% local, no paid API</span>
      <span class="pill">Mac (mlx-lm) or NVIDIA PC (Unsloth)</span>
      <span class="pill">checked Oct 2026</span>
    </div>
    <h1 class="title">{inline(title.split(":")[0])}.</h1>
    <p class="tagline">From one gradient step to a working LoRA adapter you can serve locally, on an <strong>Apple Silicon Mac</strong> or an <strong>NVIDIA PC</strong>.</p>
    <div class="verdicts">
      <div class="verdict warn"><div class="vlabel">① Rule</div><h3>Most problems don't need fine-tuning</h3><p>Prompt, retrieval and validators come first. Train only when a test set shows the base model failing.</p></div>
      <div class="verdict ok"><div class="vlabel">② Rule</div><h3>Small models train on small machines</h3><p>0.8B and 2B fit almost anywhere. 4B fits with care. 9B needs 24 GB-class hardware.</p></div>
      <div class="verdict sec"><div class="vlabel">③ Rule</div><h3>Evaluate or it didn't happen</h3><p>Falling training loss proves fit, not improvement. Write a held-out test before you train.</p></div>
    </div>
    <div class="hero-cta">
      <a class="btn primary" href="#start">Start here ↓</a>
      <a class="btn" href="#mac">Mac path</a>
      <a class="btn" href="#nvidia">NVIDIA path</a>
      <a class="btn" href="#trouble">Troubleshooting</a>
    </div>
  </div>
</header>'''
    nav = '<nav class="toc" aria-label="Table of contents">\n    <strong>Contents</strong>\n    ' + "\n    ".join(toc) + "\n  </nav>"
    footer = f'''<footer>
  <div class="foot-inner">
    <img src="{mascot}" alt="Sparky, the LoveSpark wolf mascot">
    <div>
      <strong>Qwen fine-tuning guide</strong>: one self-contained HTML file; open it anywhere, no server needed, nothing leaves your machine.<br>
      Built from <code>source/qwen-fine-tuning-guide.md</code> · facts checked October 2026 · MIT licensed · 100% local, no paid API · made for neurodivergent builders 🐺
    </div>
  </div>
</footer>'''
    script = "<script>\n" + (T / "base.js").read_text(encoding="utf-8") + (T / "widgets.js").read_text(encoding="utf-8") + "})();\n</script>"
    body = "\n\n    ".join(sections)
    return f'{head}<body>\n\n{hero}\n\n<div class="wrap">\n  {nav}\n\n  <main>\n    {body}\n  </main>\n</div>\n\n{footer}\n\n{script}\n</body>\n</html>\n'

if __name__ == "__main__":
    out = build()
    target = ROOT / "index.html"
    if "--check" in sys.argv:
        ok = target.exists() and target.read_text(encoding="utf-8") == out
        print("index.html is up to date" if ok else "index.html is STALE: run python3 scripts/build.py"); sys.exit(0 if ok else 1)
    target.write_text(out, encoding="utf-8")
    print(f"wrote index.html ({len(out)//1024} KB)")
