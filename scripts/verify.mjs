// Offline checks for the built guide. Run: node scripts/verify.mjs  (exit 0 = PASS)
import { readFileSync } from 'node:fs';
import { strict as assert } from 'node:assert';
import { execFileSync } from 'node:child_process';
import vm from 'node:vm';

const root = new URL('../', import.meta.url);
const html = readFileSync(new URL('index.html', root), 'utf8');
const md = readFileSync(new URL('source/qwen-fine-tuning-guide.md', root), 'utf8');
let checks = 0;
const ok = (c, msg) => { assert.ok(c, msg); checks++; };
const near = (a, b, t = 1e-9) => ok(Math.abs(a - b) < t, `${a} != ${b}`);

// 1. index.html is exactly what the build produces from the Markdown
execFileSync('python3', ['-I', 'scripts/build.py', '--check'], { cwd: root, stdio: 'pipe' }); checks++;

// 2. widget math, run from the page's own script
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script); checks++;
const mathSrc = script.slice(script.indexOf('/* ---- Fine-tuning widgets'), script.indexOf("if($('calc-step'))"));
ok(mathSrc.length > 200, 'math block not found');
const ctx = { window: {}, document: { getElementById: () => null } };
vm.createContext(ctx);
vm.runInContext(mathSrc, ctx);
const M = ctx.window.FineTuneMath;
ok(M, 'FineTuneMath not exposed');
near(M.sigmoid(-2), 0.11920292202211755);
near(M.loss(-2, 1), 2.1269280110429727);
near(-2 - 0.5 * (M.sigmoid(-2) - 1), -1.5596014610110587);
for (const y of [0, 1]) for (const w of [-20, -2, 0, 2, 20]) {
  const e = 1e-5;
  near((M.loss(w + e, y) - M.loss(w - e, y)) / (2 * e), M.sigmoid(w) - y, 1e-8);   // gradient = p − y
  ok(M.loss(w - 0.5 * (M.sigmoid(w) - y), y) <= M.loss(w, y) + 1e-12, 'a step must not raise loss');
}
for (const y of [0, 1]) for (const w of [-1000, 1000]) ok(Number.isFinite(M.loss(w, y)), 'loss overflow');
const a = M.adapter(4096, 4096, 16, 1);
ok(a.trainable === 131072 && a.base === 16777216, 'LoRA count'); near(a.ratio, 0.78125);
near(M.weights(4.66, 16).gb, 9.32); near(M.weights(4.66, 4).gb, 2.33);
near(M.weights(9, 16).gib, 16.763806343078613);
ok(JSON.stringify(M.QUIZ) === '["b","b","a","b"]', 'quiz key');

// 3. page integrity
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]);
ok(ids.length === new Set(ids).size, 'duplicate ids');
for (const [, id] of html.matchAll(/href="#([^"]+)"/g)) ok(ids.includes(id), `missing anchor #${id}`);
for (const n of ['gradient', 'lora', 'memory', 'quiz']) ok(!md.includes(`widget:${n}`) || html.includes(`id="${n === 'gradient' ? 'calc-step' : n === 'lora' ? 'calc-lora' : n === 'memory' ? 'calc-mem' : 'quizForm'}"`), `widget ${n} not rendered`);
ok(!/<(?:script|img|iframe)\b[^>]*\bsrc\s*=\s*["']https?:/i.test(html), 'external script/img');
ok(!/<link\b[^>]*\bhref\s*=\s*["']https?:/i.test(html), 'external stylesheet');
ok(!/\b(fetch|XMLHttpRequest|WebSocket|sendBeacon)\s*\(/.test(script), 'network API in script');

// 4. no local paths ship
const personal = /\/Users\/[A-Za-z]|\/home\/[A-Za-z]|C:\\Users\\/;   // no local home paths in published files
for (const [name, text] of [['index.html', html.replace(/data:[^"')]+/g, '')], ['source', md]]) ok(!personal.test(text), `personal reference in ${name}: ${text.match(personal)}`);

console.log(JSON.stringify({ result: 'PASS', checks }, null, 1));
