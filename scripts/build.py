#!/usr/bin/env python3
"""Build index.html from source/qwen-guide.md (the source of truth).

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
            wide = len(rows[0]) >= 5
            btn = '<button type="button" class="tbl-expand" aria-expanded="false">Expand table</button>' if wide else ''
            out.append(f'<div class="tablewrap{" tablewrap--wide" if wide else ""}">{btn}<table><thead><tr>{th}</tr></thead><tbody>{tb}</tbody></table></div>'); continue
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

THEMES = [("basalt", "Basalt", "#16171a", "#d9a23b"), ("bubblegum", "Bubblegum", "#fbf3f5", "#a52f66"), ("retro", "Retro Pink", "#fdeef4", "#b8175a"),
          ("noir", "Dark Noir", "#17131b", "#f09cc4"), ("beige", "Beige Paper", "#f3eee4", "#9c4a2b")]
FONTS = [("serif", "Serif", "font-family:var(--serif)"), ("sans", "Sans", "font-family:var(--sans)"),
         ("dyslexic", "OpenDyslexic", "font-family:var(--dys)")]
RULES = [("Run it before you change it", "Most \"the model can't do X\" problems are solved with a better prompt, retrieval or a different model, not with training or surgery."),
         ("Test on your own task", "A small set of real inputs with the answers you'd accept, written before you start, beats every leaderboard and every loss curve."),
         ("Own what you ship", "A local model is yours: its privacy, its mistakes, and anything you removed from it.")]

# Korean themes: the four Korean popup themes of the LoveSpark Retro Cursor extension
# (themes/lovespark-retro-cursor/popup.css): same palettes and backgrounds, mapped onto this page's tokens.
# Same table as the papers archive (lovespark.love/papers), with its card selectors mapped to this guide's widgets.
# "stops" are the colours text can sit on (background gradient stops, card fills); the build checks them all.
KOREAN = [
    ("seoul", "Seoul Y2K", "서울", dict(
        paper="#fdf9ff", deep="#ffe8f2", wash="#e9dcff", ink="#3a2a3a", soft="#4e3a50", muted="#6b4a66",
        line="#ecd6ec", strong="#c4a8e0", accent="#a8134f", swatch="#ff5baa",
        bg="linear-gradient(160deg, #dff1ff 0%, #ffe8f2 45%, #fff4d6 78%, #e9dcff 100%)",
        stops=["#dff1ff", "#ffe8f2", "#fff4d6", "#e9dcff"],
        extra="""{t} .calc, {t} .plaque, {t} .toc-group {{ border-radius: 10px; border-color: #c4a8e0; box-shadow: 3px 3px 0 #c4a8e0; }}
{t} .calc:hover, {t} .calc:focus-within {{ box-shadow: 4px 4px 0 #ff8fc6; }}
{t} .wbtn {{ border-radius: 6px; }}""")),
    ("gingham", "Gingham Sticker", "깅엄", dict(
        paper="#fffafc", deep="#ffe3ee", wash="#fff3f7", ink="#2a1824", soft="#3e2434", muted="#6e3a57",
        line="#f0c3d5", strong="#2a1824", accent="#b0124f", swatch="#f6a6c4",
        bg="linear-gradient(180deg, #fff3f7 0%, #fff3f7 100%)",
        stops=["#fff3f7", "#fffafc"],
        # the gingham itself stays off the reading area (text never sits on a pattern): a fixed top band and the footer
        extra="""{t} body::before {{ content: ""; position: fixed; inset: 0 0 auto 0; height: 10px; z-index: 30; pointer-events: none;
  background: repeating-linear-gradient(0deg, rgba(246,166,196,.6) 0 5px, transparent 5px 10px), repeating-linear-gradient(90deg, rgba(246,166,196,.6) 0 5px, transparent 5px 10px), #fff3f7; }}
{t} footer {{ background: repeating-linear-gradient(0deg, rgba(246,166,196,.42) 0 9px, transparent 9px 18px), repeating-linear-gradient(90deg, rgba(246,166,196,.42) 0 9px, transparent 9px 18px), #fff3f7; }}
{t} footer .foot-inner {{ background: #fffafc; border: 2px solid #2a1824; box-shadow: 3px 3px 0 #2a1824; border-radius: 14px; }}
{t} .calc, {t} .plaque, {t} .toc-group.current {{ background: #fffafc; border: 2px solid #2a1824; border-radius: 14px; box-shadow: 3px 3px 0 #2a1824; }}
{t} .calc:hover, {t} .calc:focus-within {{ box-shadow: 4px 4px 0 #ff4f9a; border-color: #2a1824; }}
{t} .wbtn {{ border: 2px solid #2a1824; border-radius: 10px; box-shadow: 2px 2px 0 #2a1824; }}""")),
    ("holo", "Holo Mochi", "모찌", dict(
        paper="#faf7ff", deep="#f3ebff", wash="#e9d5ff", ink="#2b1b4a", soft="#3d2c63", muted="#54447a",
        line="#e3d6f7", strong="#c7b2ee", accent="#6d35d8", swatch="#9b6cf0",
        bg="radial-gradient(60% 45% at 12% 8%, #ffd1ea 0%, transparent 70%), radial-gradient(55% 40% at 92% 18%, #cfe6ff 0%, transparent 70%), radial-gradient(60% 50% at 80% 88%, #c8f7e1 0%, transparent 72%), radial-gradient(55% 45% at 10% 80%, #ffd8c2 0%, transparent 70%), linear-gradient(#efe4ff, #efe4ff)",
        stops=["#ffd1ea", "#cfe6ff", "#c8f7e1", "#ffd8c2", "#efe4ff"],
        extra="""{t} body {{ background-attachment: fixed; }}
{t} .calc, {t} .plaque, {t} .toc-group {{ background: rgba(255,255,255,.62); border-color: rgba(255,255,255,.9); border-radius: 22px; box-shadow: 0 10px 28px rgba(140,100,230,.18), 0 2px 6px rgba(255,180,220,.3); }}
{t} .calc:hover, {t} .calc:focus-within {{ background: rgba(250,247,255,.92); border-color: #9b6cf0; box-shadow: 0 14px 36px rgba(140,100,230,.3); }}
{t} .wbtn {{ border-radius: 14px; }}""")),
    ("neon", "Neon Usagi", "네온", dict(
        paper="#120b18", deep="#1e1030", wash="#2a1740", ink="#f3eaf2", soft="#e4d5ea", muted="#c9b3dc",
        line="#2e1a40", strong="#7d2a8e", accent="#ff46d6", on="#120b18", swatch="#ff46d6", dark=True,
        bg="radial-gradient(80% 50% at 50% -10%, rgba(255,70,214,.22) 0%, transparent 70%), radial-gradient(70% 45% at 100% 110%, rgba(110,17,145,.45) 0%, transparent 70%)",
        stops=["#120b18", "#3e1a40", "#3a0f50", "#2a1740"],   # base, top pink glow, bottom violet glow, card fill
        extra="""{t} body {{ background-attachment: fixed; }}
{t} .calc, {t} .plaque, {t} .toc-group {{ background: rgba(42,23,64,.88); border-color: rgba(255,70,214,.45); border-radius: 16px; }}
{t} .calc:hover, {t} .calc:focus-within {{ border-color: #ff46d6; box-shadow: 0 0 0 1px rgba(255,70,214,.55), 0 0 14px rgba(255,70,214,.45), 0 0 34px rgba(255,70,214,.2); }}
{t} .wbtn {{ border-radius: 12px; border-color: rgba(255,70,214,.35); }}""")),
]

def _lum(h):
    c = [int(h[i:i+2], 16) / 255 for i in (1, 3, 5)]
    c = [v / 12.92 if v <= .03928 else ((v + .055) / 1.055) ** 2.4 for v in c]
    return .2126 * c[0] + .7152 * c[1] + .0722 * c[2]
def contrast(a, b):
    A, B = _lum(a), _lum(b); return (max(A, B) + .05) / (min(A, B) + .05)
def mix(a, b, t):
    return "#" + "".join(f"{round(int(a[i:i+2], 16) * t + int(b[i:i+2], 16) * (1 - t)):02x}" for i in (1, 3, 5))

def korean_css():
    """CSS for the Korean themes; the build fails if any text pair drops below 4.8:1 (the template's floor)."""
    out = "\n/* ---------- Korean themes (generated by scripts/build.py from KOREAN) ---------- */\n"
    out += """/* the theme menu holds 9 themes: no mid-syllable Hangul breaks, capped to the window and scrollable */
#themeMenu { min-width: 272px; max-width: calc(100vw - 32px); max-height: calc(100vh - 110px); max-height: calc(100dvh - 110px); overflow-y: auto; overscroll-behavior: contain; }
#themeMenu button { white-space: nowrap; }
#themeMenu .kr-name { margin-left: auto; padding-left: 10px; font-size: 12px; color: var(--muted); word-break: keep-all; }
#themeMenu button[aria-checked="true"]:has(.kr-name)::after { margin-left: 8px; }
"""
    for key, name, _, t in KOREAN:
        bgs = [t["paper"], t["deep"], t["wash"], mix(t["deep"], t["paper"], .6)] + t["stops"]
        for fg in ("ink", "soft", "muted", "accent"):
            for col in bgs:
                r = contrast(t[fg], col)
                assert r >= 4.8, f"{name}: {fg} on {col} is {r:.2f}:1, below 4.8"
        on = t.get("on", "#ffffff")
        assert contrast(on, t["accent"]) >= 4.8, f"{name}: text on accent below 4.8"
        sel = f':root[data-theme="kr-{key}"]'
        out += (f'{sel} {{\n  --paper: {t["paper"]}; --paper-deep: {t["deep"]}; --wash: {t["wash"]};\n'
                f'  --ink: {t["ink"]}; --ink-soft: {t["soft"]}; --muted: {t["muted"]};\n  --line: {t["line"]}; --line-strong: {t["strong"]};\n'
                f'  --accent: {t["accent"]}; --on-accent: {on};\n  --ls-bg-gradient: {t["bg"]};\n  color-scheme: {"dark" if t.get("dark") else "light"};\n}}\n'
                + t["extra"].format(t=sel) + "\n")
    return out

ROMAN = {"1": "I", "2": "II", "3": "III", "4": "IV"}

def toc_html(toc):
    """Contents rail: "Start here", then one card per part (numeral + name) holding its sections."""
    start, groups = None, []
    for i, t in toc:
        if i == "part": groups.append((t, []))
        elif not groups: start = (i, t)
        else: groups[-1][1].append((i, t))
    out = f'<a class="toc-start" href="#{start[0]}">{html.escape(start[1])}</a><ol class="toc-parts">' if start else '<ol class="toc-parts">'
    for title, items in groups:
        m = re.match(r"Part (\d+) · (.*)", title)
        num, label, name = (ROMAN.get(m[1], m[1]), f"Part {m[1]}", m[2]) if m else ("§", "Appendix", title)
        links = "".join(f'<li><a href="#{i}">{html.escape(t)}</a></li>' for i, t in items)
        out += (f'<li class="toc-group"><a class="toc-head" href="#{items[0][0]}"><span class="toc-num" aria-hidden="true">{num}</span>'
                f'<span class="toc-title"><small>{label}</small>{html.escape(name)}</span></a><ol>{links}</ol></li>')
    return out + "</ol>"

def build():
    md = (ROOT / "source/qwen-guide.md").read_text(encoding="utf-8")
    parts = re.split(r"^<!-- (section|part) (.*?) -->\n", md, flags=re.M)
    intro = parts[0].split("\n", 1)[1]
    sections, toc = [], []   # toc entries: ("part", title) or (id, label)
    sections.append(f'<section id="start">\n      <p class="kicker"><span class="rule"></span>Start here</p>\n      <h2>Before you begin</h2>\n      {blocks(intro.split(chr(10)))}\n    </section>')
    toc.append(("start", "Start here"))
    for kind, attrs, body in zip(parts[1::3], parts[2::3], parts[3::3]):
        a = dict(re.findall(r'(\w+)="([^"]*)"', attrs))
        if kind == "part":
            toc.append(("part", a["title"]))
            assert not body.strip(), "content between a part marker and its first section"
            continue
        h2, rest = body.split("\n", 1)
        h2 = re.sub(r"^## ", "", h2)
        sections.append(f'<section id="{a["id"]}">\n      <p class="kicker"><span class="rule"></span>{html.escape(a["kicker"])}</p>\n      <h2>{inline(h2)}</h2>\n      {blocks(rest.split(chr(10)))}\n    </section>')
        toc.append((a["id"], a.get("toc", h2)))
    mascot = (T / "mascot.datauri").read_text(encoding="utf-8").strip()
    ankh = (T / "ankh.datauri").read_text(encoding="utf-8").strip()   # Basalt wordmark, as on lovespark.love
    desc = "A first-principles, offline guide to local Qwen models in four parts: run one on a Mac, Linux or Windows; build retrieval and shared agent memory on it; fine-tune it with LoRA; understand abliteration, its costs and your responsibilities. With interactive calculators and verified sources. No paid API."
    css = (T / "fonts.css").read_text(encoding="utf-8") + (T / "style.css").read_text(encoding="utf-8") + korean_css()
    themes = "".join(f'<button type="button" role="menuitemradio" aria-checked="false" data-theme="{k}"><span class="swatch" style="background:linear-gradient(135deg,{bg} 50%,{ac} 50%)"></span>{n}</button>' for k, n, bg, ac in THEMES)
    korean = "".join(f'<button type="button" role="menuitemradio" aria-checked="false" data-theme="kr-{k}"><span class="swatch" style="background:linear-gradient(135deg,{t["paper"]} 50%,{t["swatch"]} 50%)"></span>{n}<span class="kr-name" lang="ko">{ko}</span></button>' for k, n, ko, t in KOREAN)
    fonts = "".join(f'<button type="button" role="menuitemradio" aria-checked="false" data-font="{k}"><span class="fontpreview" style="{st}" aria-hidden="true">Aa</span>{n}</button>' for k, n, st in FONTS)
    rules = "".join(f"<li><h3>{inline(t)}</h3><p>{inline(d)}</p></li>" for t, d in RULES)
    body = "\n\n    ".join(sections)
    toc_items = toc_html(toc)
    return f"""<!doctype html>
<html lang="en" data-theme="basalt" data-font="serif">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen, locally · run it, fine-tune it, understand abliteration</title>
<meta name="description" content="{desc}">
<meta property="og:title" content="Qwen, locally · run it, fine-tune it, understand abliteration">
<meta property="og:description" content="{desc}">
<meta property="og:url" content="https://lovespark.love/qwen-fine-tuning-guide/">
<meta name="theme-color" content="#16171a">
<style>
{css}</style>
</head>
<body>
<a class="skip-link" href="#main">Skip to content</a>
<div id="progress" class="progress" aria-hidden="true"></div>

<header class="topbar">
  <a class="wordmark" href="#top" aria-label="Fine-tune Qwen locally, back to top">
    <img src="{mascot}" alt="Sparky, the LoveSpark wolf mascot">
    <img class="wordmark-ankh" src="{ankh}" alt="" aria-hidden="true">
    <span class="wordmark-text"><span class="wordmark-title">Qwen, locally</span><span class="wordmark-sub">A LoveSpark guide · Oct 2026</span></span>
  </a>
  <nav class="topnav" aria-label="Main navigation">
    <a href="#run-basics">Run</a>
    <a href="#embed">Build</a>
    <a href="#basics">Fine-tune</a>
    <a href="#abl-what">Abliteration</a>
    <a href="#trouble">Reference</a>
    <div class="appearance">
      <button id="themeBtn" type="button" aria-haspopup="true" aria-expanded="false" aria-controls="themeMenu">Change Font &amp; Theme</button>
      <div id="themeMenu" role="menu" aria-label="Theme and reading font" hidden>
        <p class="menu-label">Theme</p>{themes}
        <p class="menu-label">Korean</p>{korean}
        <hr>
        <p class="menu-label">Reading font</p>{fonts}
      </div>
    </div>
  </nav>
</header>

<section class="hero" id="top" aria-labelledby="hero-title">
  <div class="hero-copy">
    <p class="eyebrow"><span class="rule"></span>A guide to local models</p>
    <h1 id="hero-title">Qwen,<br>locally</h1>
    <p class="hero-dates">Run it · build on it · fine-tune it · understand abliteration</p>
    <p class="hero-description">A first-principles guide to Qwen3.5 on your own machine, in four parts: choose and run a local model, build retrieval and shared memory for your agents on it, fine-tune it with LoRA, and understand what abliteration does and costs. Mac, Linux or Windows. No paid API, no accounts, nothing leaves your computer.</p>
    <div class="hero-actions">
      <a class="button" href="#start">Start reading <span aria-hidden="true">→</span></a>
      <a class="text-link" href="#run-basics">Part 1 · Run →</a>
      <a class="text-link" href="#embed">Part 2 · Build →</a>
      <a class="text-link" href="#basics">Part 3 · Fine-tune →</a>
      <a class="text-link" href="#abl-what">Part 4 · Abliteration →</a>
    </div>
  </div>
  <figure class="plaque" aria-label="Three rules this guide keeps coming back to">
    <ol>{rules}</ol>
  </figure>
</section>

<div class="layout">
  <nav class="toc" aria-label="Contents"><p class="eyebrow">Contents</p>{toc_items}</nav>
  <main id="main">
    <details class="toc-mobile"><summary>Contents</summary>{toc_items}</details>
    {body}
  </main>
</div>

<footer>
  <div class="foot-inner">
    <img src="{mascot}" alt="Sparky, the LoveSpark wolf mascot">
    <div>One self-contained page: no server, no tracking, nothing leaves your machine. Built from <code>source/qwen-guide.md</code> · MIT licensed · a <a href="https://lovespark.love/" rel="noopener">LoveSpark</a> guide · <a href="https://github.com/Joona-t/qwen-guide" rel="noopener">source on GitHub</a>.</div>
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
