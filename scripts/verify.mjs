// Offline checks for the built guide. Run: node scripts/verify.mjs  (exit 0 = PASS)
import { readFileSync } from 'node:fs';
import { strict as assert } from 'node:assert';
import { execFileSync } from 'node:child_process';
import vm from 'node:vm';

const root = new URL('../', import.meta.url);
const html = readFileSync(new URL('index.html', root), 'utf8');
const md = readFileSync(new URL('source/qwen-guide.md', root), 'utf8');
let checks = 0;
const ok = (c, msg) => { assert.ok(c, msg); checks++; };
const near = (a, b, t = 1e-9) => ok(Math.abs(a - b) < t, `${a} != ${b}`);

// 1. index.html is exactly what the build produces from the Markdown
execFileSync('python3', ['-I', 'scripts/build.py', '--check'], { cwd: root, stdio: 'pipe' }); checks++;
// 1b. no stale scope or settled-UNCERTAIN text, no dated claim older than 45 days (scripts/freshness.py)
execFileSync('python3', ['-I', 'scripts/freshness.py'], { cwd: root, stdio: 'pipe' }); checks++;

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
// handwired: program and network agree on every whole a in 0..10; between 4 and 5 the gate ramps (documented in the widget caveat)
for (let a = 0; a <= 10; a++) ok(M.wiredProgram(a) === M.wiredNetwork(a), `handwired disagrees at a=${a}`);
for (let a = 0; a <= 10; a += 0.25) ok(M.wiredGate(a) >= 0 && M.wiredGate(a) <= 1, 'gate outside [0,1]');
near(M.wiredGate(4.5), 0.5); near(M.wiredNetwork(4.5), 11); ok(M.wiredProgram(4.5) === 10, 'program at 4.5');
// ported calculators (from the retired qwen-local-guide)
ok(Math.round(M.decode(120, 6.6)) === 18, 'decode M4 9B'); ok(Math.round(M.decode(120, 3.4)) === 35, 'decode M4 4B'); ok(Math.round(M.decode(153, 6.6)) === 23, 'decode M6 9B (matches Part 1 prose)');
near(M.budget(6.6, true, false, false).workload, 8.8); ok(M.budget(6.6, true, false, false).verdict === 'ok', 'budget 9B+RAG short');
near(M.budget(6.6, true, true, false).workload, 9.8); ok(M.budget(6.6, true, true, false).verdict === 'tight', 'budget 9B+RAG long');
near(M.budget(6.6, true, true, true).kv, 1.5); ok(M.budget(9, true, true, false).verdict === 'over', 'budget 9GB+RAG long');
ok(M.BUDGET.os + M.BUDGET.usableHigh === M.BUDGET.total, 'budget bands add up');
ok(Math.round(M.chain(0.95, 8) * 100) === 66, 'compounding: 0.95^8 ≈ 66%');
near(M.wiredNetwork(4.25), 11.5); near(M.wiredNetwork(4.75), 10.5);
// fit + decode ceiling (research.md §2)
{ const r = M.fitCalc({ kind: 'nv', mem: 24, cards: 1, model: 19, kvKiB: 64, ctx: 8192, kvType: 'q8_0' });
  ok(r.verdict === 'fits', '3090 + 19 GB at 8K q8_0 fits'); near(r.usable, 23); near(r.kv, 8192 * 64 * 1024 * 0.5 / 1e9); }
ok(M.fitCalc({ kind: 'mac', mem: 16, cards: 1, model: 19, kvKiB: null, ctx: 8192, kvType: 'f16' }).verdict === 'no', 'Mac 16 GB + 19 GB does not fit');
near(M.fitCalc({ kind: 'mac', mem: 16, cards: 1, model: 6.6, kvKiB: null, ctx: 8192, kvType: 'f16' }).usable, 12);
ok(M.fitCalc({ kind: 'nv', mem: 24, cards: 1, model: 19, kvKiB: null, ctx: 32768, kvType: 'f16' }).kv === null, 'no KV constant -> kv null');
near(M.fitCalc({ kind: 'nv', mem: 48, cards: 2, model: 19, kvKiB: null, ctx: 8192, kvType: 'f16' }).usable, 46);
ok(M.fitCalc({ kind: 'nv', mem: 24, cards: 1, model: 22.1, kvKiB: 20, ctx: 8192, kvType: 'f16' }).verdict === 'tight', '35B-A3B Q4 on 3090 is tight');
near(M.decodeCeiling(936, 19), 936 / 19); ok(Math.abs(M.decodeCeiling(936, 19) - 49.3) < 0.05, 'decode 3090 19 GB ≈ 49.3');
for (const a of [-3, 0, 3.9, 4, 5, 5.1, 10, 1e6]) ok(M.wiredProgram(a) === M.wiredNetwork(a), `handwired disagrees at a=${a}`);
// quant widget (research.md §B, §C, §E): bpw, ceiling and context against hand-computed values
{ const m = M.QMODELS.q27, q4 = m.quants.find(q => q[0] === 'Q4_K_M')[1];
  ok(q4 === 19.0 && m.kvKiB === 64 && m.params === 27.78, 'Qwen3.8-27B constants');
  const r = M.quantCalc({ params: m.params, active: m.active, kvKiB: m.kvKiB, size: q4, kind: 'nv', mem: 24, bw: 936, cards: 1, kvType: 'f16' });
  near(r.bpw, 19 * 8 / 27.78); ok(Math.abs(r.bpw - 5.47) < 0.01, 'Q4_K_M ≈ 5.47 bpw');
  near(r.ceiling, 936 / 19); near(r.usable, 23); ok(r.ctx === Math.floor(4e9 / 65536), '3090 Q4_K_M f16 context = 4 GB / 64 KiB');
  const h = M.quantCalc({ params: m.params, active: m.active, kvKiB: m.kvKiB, size: q4, kind: 'nv', mem: 24, bw: 936, cards: 1, kvType: 'q8_0' });
  ok(h.ctx === Math.floor(4e9 / 32768), 'q8_0 doubles the context');
  const bf = M.quantCalc({ params: m.params, active: m.active, kvKiB: m.kvKiB, size: 53.8, kind: 'nv', mem: 48, bw: 936, cards: 2, kvType: 'f16' });
  ok(!bf.fits && bf.ctx === 0, 'BF16 27B does not fit on two 3090s');
  const q8 = M.quantCalc({ params: m.params, active: m.active, kvKiB: m.kvKiB, size: 28.6, kind: 'nv', mem: 48, bw: 936, cards: 2, kvType: 'f16' });
  ok(q8.ctx === 262144, 'Q8_0 on two 3090s is capped at the native window'); ok(Math.round(q8.ceiling) === 33, 'Q8_0 ceiling ≈ 33'); }
{ const m = M.QMODELS.q122, r = M.quantCalc({ params: m.params, active: m.active, kvKiB: m.kvKiB, size: 44.7, kind: 'nv', mem: 48, bw: 936, cards: 2, kvType: 'f16' });
  ok(r.fits && Math.abs(r.left - 1.3) < 1e-9, '122B IQ3_XXS leaves 1.3 GB on two 3090s'); near(r.ceiling, 936 / (44.7 * 10 / 125.09));
  ok(!M.quantCalc({ params: m.params, active: m.active, kvKiB: m.kvKiB, size: 76.5, kind: 'nv', mem: 48, bw: 936, cards: 2, kvType: 'f16' }).fits, '122B Q4_K_M needs RAM offload'); }
for (const [k, m] of Object.entries(M.QMODELS)) { ok(m.quants.some(q => q[0] === 'Q4_K_M'), `${k} has a Q4_K_M default`); for (let i = 1; i < m.quants.length; i++) ok(m.quants[i][1] > m.quants[i-1][1], `${k} quants sorted by size`); }

// 3. page integrity
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]);
ok(ids.length === new Set(ids).size, 'duplicate ids');
for (const [, id] of html.matchAll(/href="#([^"]+)"/g)) ok(ids.includes(id), `missing anchor #${id}`);
// fit: unknown KV with < 5 GB headroom is 'tight' (Gemma 4 31B on a 3090), >= 5 GB is 'fits'
ok(M.fitCalc({kind:'gpu', mem:24, cards:1, model:18.3, kvKiB:null, ctx:16384, kvType:'q8_0'}).verdict === 'tight', 'fit: unknown-KV tight');
ok(M.fitCalc({kind:'gpu', mem:24, cards:1, model:14, kvKiB:null, ctx:16384, kvType:'q8_0'}).verdict === 'fits', 'fit: unknown-KV fits');
const WIDGET_IDS = { decode: 'calc-decode', budget: 'calc-budget', reliability: 'calc-rel', gradient: 'calc-step', handwired: 'calc-wired', lora: 'calc-lora', memory: 'calc-mem', quiz: 'quizForm', fit: 'calc-fit', quant: 'calc-quant' };
for (const [n, id] of Object.entries(WIDGET_IDS)) ok(!md.includes(`widget:${n}`) || html.includes(`id="${id}"`), `widget ${n} not rendered`);
ok(!/<(?:script|img|iframe)\b[^>]*\bsrc\s*=\s*["']https?:/i.test(html), 'external script/img');
ok(!/<link\b[^>]*\bhref\s*=\s*["']https?:/i.test(html), 'external stylesheet');
ok(!/\b(fetch|XMLHttpRequest|WebSocket|sendBeacon)\s*\(/.test(script), 'network API in script');

// 4. no local paths ship
const personal = /\/Users\/[A-Za-z]|\/home\/[A-Za-z]|C:\\Users\\/;   // no local home paths in published files
for (const [name, text] of [['index.html', html.replace(/data:[^"')]+/g, '')], ['source', md]]) ok(!personal.test(text), `personal reference in ${name}: ${text.match(personal)}`);

console.log(JSON.stringify({ result: 'PASS', checks }, null, 1));
