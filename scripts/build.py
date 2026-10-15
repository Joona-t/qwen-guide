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
    kinds = {"⚠": ("gotcha", "Watch out"), "◆": ("caveat", "Caveat"), "📝": ("note", "Note")}
    cls, tag = kinds.get(t[:1], ("note", "How this guide was checked"))
    m = re.match(r"^(⚠|◆|📝)\s*\*\*(.+?)\*\*\s*(.*)$", t, re.S)
    if m:
        tag, body = m.group(2).rstrip('.'), m.group(3)
    elif t[:1] in kinds:
        body = t[1:].strip()
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

THEMES = [("candy", "Candy", "#fbf3f5", "#a52f66"), ("retro", "Retro Pink", "#fdeef4", "#b8175a"),
          ("noir", "Dark Noir", "#17131b", "#f09cc4"), ("beige", "Beige Paper", "#f3eee4", "#9c4a2b")]
FONTS = [("serif", "Serif", "font-family:var(--serif)"), ("sans", "Sans", "font-family:var(--sans)"),
         ("dyslexic", "OpenDyslexic", "font-family:var(--dys)")]
RULES = [("Most problems don't need fine-tuning", "Prompt, retrieval and validators come first. Train only when a test set shows the base model failing."),
         ("Small models train on small machines", "0.8B and 2B fit almost anywhere. 4B fits with care. 9B needs 24 GB-class hardware."),
         ("Evaluate or it didn't happen", "Falling training loss proves fit, not improvement. Write a held-out test before you train.")]

def build():
    md = (ROOT / "source/qwen-fine-tuning-guide.md").read_text(encoding="utf-8")
    parts = re.split(r"^<!-- section (.*?) -->\n", md, flags=re.M)
    intro = parts[0].split("\n", 1)[1]
    sections, toc = [], []
    sections.append(f'<section id="start">\n      <p class="kicker"><span class="rule"></span>Start here</p>\n      <h2>Before you begin</h2>\n      {blocks(intro.split(chr(10)))}\n    </section>')
    toc.append(("start", "Start here"))
    for attrs, body in zip(parts[1::2], parts[2::2]):
        a = dict(re.findall(r'(\w+)="([^"]*)"', attrs))
        h2, rest = body.split("\n", 1)
        h2 = re.sub(r"^## ", "", h2)
        sections.append(f'<section id="{a["id"]}">\n      <p class="kicker"><span class="rule"></span>{html.escape(a["kicker"])}</p>\n      <h2>{inline(h2)}</h2>\n      {blocks(rest.split(chr(10)))}\n    </section>')
        toc.append((a["id"], a.get("toc", h2)))
    mascot = (T / "mascot.datauri").read_text(encoding="utf-8").strip()
    desc = "A first-principles, offline guide to fine-tuning Qwen3.5 locally: LoRA on a Mac with mlx-lm or an NVIDIA PC with Unsloth, with interactive calculators, honest memory limits and an evaluation-first workflow. No paid API."
    css = (T / "fonts.css").read_text(encoding="utf-8") + (T / "style.css").read_text(encoding="utf-8")
    themes = "".join(f'<button type="button" role="menuitemradio" aria-checked="false" data-theme="{k}"><span class="swatch" style="background:linear-gradient(135deg,{bg} 50%,{ac} 50%)"></span>{n}</button>' for k, n, bg, ac in THEMES)
    fonts = "".join(f'<button type="button" role="menuitemradio" aria-checked="false" data-font="{k}"><span class="fontpreview" style="{st}" aria-hidden="true">Aa</span>{n}</button>' for k, n, st in FONTS)
    rules = "".join(f"<li><h3>{inline(t)}</h3><p>{inline(d)}</p></li>" for t, d in RULES)
    body = "\n\n    ".join(sections)
    toc_items = "".join(f'<li><a href="#{i}">{html.escape(t)}</a></li>' for i, t in toc)
    return f"""<!doctype html>
<html lang="en" data-theme="candy" data-font="serif">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fine-tune Qwen locally · a LoRA guide for Mac and PC</title>
<meta name="description" content="{desc}">
<meta property="og:title" content="Fine-tune Qwen locally · a LoRA guide for Mac and PC">
<meta property="og:description" content="{desc}">
<meta property="og:url" content="https://lovespark.love/qwen-fine-tuning-guide/">
<meta name="theme-color" content="#fbf3f5">
<style>
{css}</style>
</head>
<body>
<a class="skip-link" href="#main">Skip to content</a>
<div id="progress" class="progress" aria-hidden="true"></div>

<header class="topbar">
  <a class="wordmark" href="#top" aria-label="Fine-tune Qwen locally, back to top">
    <img src="{mascot}" alt="Sparky, the LoveSpark wolf mascot">
    <span class="wordmark-text"><span class="wordmark-title">Qwen Fine-Tuning</span><span class="wordmark-sub">A LoveSpark guide · Oct 2026</span></span>
  </a>
  <nav class="topnav" aria-label="Main navigation">
    <a href="#mac">Mac path</a>
    <a href="#nvidia">NVIDIA path</a>
    <a href="#trouble">Troubleshooting</a>
    <div class="appearance">
      <button id="themeBtn" type="button" aria-haspopup="true" aria-expanded="false" aria-controls="themeMenu">Change Theme</button>
      <div id="themeMenu" role="menu" aria-label="Theme and reading font" hidden>
        <p class="menu-label">Theme</p>{themes}
        <hr>
        <p class="menu-label">Reading font</p>{fonts}
      </div>
    </div>
  </nav>
</header>

<section class="hero" id="top" aria-labelledby="hero-title">
  <div class="hero-copy">
    <p class="eyebrow"><span class="rule"></span>A local LoRA guide</p>
    <h1 id="hero-title">Fine-tune<br>Qwen locally</h1>
    <p class="hero-dates">Qwen3.5 0.8B – 9B · checked October 2026</p>
    <p class="hero-description">From one gradient step to a working LoRA adapter you can serve on your own machine, on an Apple Silicon Mac or an NVIDIA PC. No paid API, no accounts, nothing leaves your computer.</p>
    <div class="hero-actions">
      <a class="button" href="#start">Start reading <span aria-hidden="true">→</span></a>
      <a class="text-link" href="#mac">Mac path →</a>
      <a class="text-link" href="#nvidia">NVIDIA path →</a>
    </div>
  </div>
  <figure class="plaque" aria-label="Three rules this guide keeps coming back to">
    <ol>{rules}</ol>
  </figure>
</section>

<div class="layout">
  <nav class="toc" aria-label="Contents"><p class="eyebrow">Contents</p><ol>{toc_items}</ol></nav>
  <main id="main">
    <details class="toc-mobile"><summary>Contents</summary><ol>{toc_items}</ol></details>
    {body}
  </main>
</div>

<footer>
  <div class="foot-inner">
    <img src="{mascot}" alt="Sparky, the LoveSpark wolf mascot">
    <div>One self-contained page: no server, no tracking, nothing leaves your machine. Built from <code>source/qwen-fine-tuning-guide.md</code> · MIT licensed · companion to <a href="https://lovespark.love/" rel="noopener">LoveSpark</a>'s <a href="https://github.com/Joona-t/qwen-local-guide" rel="noopener">Qwen Local guide</a>.</div>
  </div>
</footer>

<script>
{(T / "base.js").read_text(encoding="utf-8")}{(T / "widgets.js").read_text(encoding="utf-8")}}})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    out = build()
    target = ROOT / "index.html"
    if "--check" in sys.argv:
        ok = target.exists() and target.read_text(encoding="utf-8") == out
        print("index.html is up to date" if ok else "index.html is STALE: run python3 scripts/build.py"); sys.exit(0 if ok else 1)
    target.write_text(out, encoding="utf-8")
    print(f"wrote index.html ({len(out)//1024} KB)")
