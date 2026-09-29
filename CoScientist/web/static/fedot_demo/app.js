/* Vendored FEDOT.MAS renderer, adapted from PR #368 at 6e18bcf4.
 * Observation-only: init() does not wire the stand's run/import/key controls.
 * The bridge at the bottom consumes the session/run journal via FedotRuns.
 * See README.md here before updating the vendored assets.
 */
/* FEDOT.MAS GUI — прототип. Ванильный JS, без сборки и внешних зависимостей. */
(() => {
"use strict";

const $ = (id) => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";

/* ──────────────────── Доступ: токен из ссылки и ключ провайдера ────────────────────
 * В публичном режиме сервер не хранит ключ провайдера — его вводит сам пользователь,
 * а токен доступа приезжает в ссылке (?t=…). Обёртка над fetch подставляет токен во
 * все запросы разом: так ни один из девяти вызовов нельзя забыть.
 */
const LS_TOKEN = "fedotmas-token";
const LS_KEY = "fedotmas-key";

/* ─────────────────────────── Иконки ролей ─────────────────────────── */
const ICONS = {
  hub:    ["M12 3v5M12 16v5M4.5 7.5l3.5 3M16 13.5l3.5 3M19.5 7.5l-3.5 3M8 13.5l-3.5 3", "M12 12m-3 0a3 3 0 106 0a3 3 0 10-6 0"],
  search: ["M11 4a7 7 0 100 14 7 7 0 000-14z", "M16.5 16.5L21 21"],
  chart:  ["M4 20V10M10 20V4M16 20v-7M22 20H2"],
  shield: ["M12 3l7 3v6c0 4.2-2.9 7.7-7 9-4.1-1.3-7-4.8-7-9V6l7-3z", "M9 12l2.2 2.2L15.5 10"],
  doc:    ["M6 3h8l4 4v14H6z", "M14 3v4h4M9 12h6M9 16h6"],
  db:     ["M12 3c4.4 0 8 1.3 8 3s-3.6 3-8 3-8-1.3-8-3 3.6-3 8-3z", "M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6", "M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"],
  coin:   ["M12 3v18", "M16 7H10a2.5 2.5 0 000 5h4a2.5 2.5 0 010 5H8"],
  wrench: ["M15 3a5 5 0 00-4.6 7L3 17.4 6.6 21l7.4-7.4A5 5 0 0021 9l-3 3-3-3 3-3a5 5 0 00-3-3z"],
  user:   ["M12 4a3.5 3.5 0 100 7 3.5 3.5 0 000-7z", "M5 20c0-3.6 3.1-6 7-6s7 2.4 7 6"]
};

const ROLE_RULES = [
  [/router|coordinat|orchestr|dispatch|маршрут|координат|диспетчер/i, "hub", "coord"],
  [/critic|valid|review|complian|fraud|check|критик|валид|провер|контрол|комплаенс|мошен/i, "shield", "critic"],
  [/research|search|explor|vendor|news|registry|litig|поиск|исследов|разведк|новост|реестр/i, "search", "worker"],
  [/sql|schema|query|data|telemetry|телеметри|данн|запрос|витрин|схем/i, "db", "worker"],
  [/analy|scor|synthes|risk|insight|анализ|оценк|риск|синтез|скоринг|вывод/i, "chart", "worker"],
  [/writer|memo|report|plan|answer|doc|histor|extract|план|отчёт|отчет|записк|журнал|документ|истори|извлеч/i, "doc", "worker"],
  [/billing|invoice|payment|tariff|биллинг|счёт|счет|платеж|тариф|комисси/i, "coin", "worker"],
  [/support|tech|repair|maint|поддержк|техник|ремонт|обслуживан|наладк/i, "wrench", "worker"],
];

/* Цвет роли задаётся классом .role-* в CSS — иначе он не переживал бы смену темы. */
const ROLE_CLASS = { coord: "role-coord", worker: "role-worker", critic: "role-critic" };

function roleOf(name, isCoord) {
  if (isCoord) return { icon: "hub", kind: "coord" };
  for (const [re, icon, kind] of ROLE_RULES) if (re.test(name)) return { icon, kind };
  return { icon: "user", kind: "worker" };
}

/* ─────────────────────────── Состояние ─────────────────────────── */
const S = {
  preset: null,
  agents: new Map(),   // name -> {name, instruction, description, tools, output_key, isCoord, role}
  events: [],
  idx: 0,
  playing: false,
  speed: 1,
  files: [],           // прикреплённые файлы-источники: {name, path}
  timer: null,
  tokens: 0,
  seconds: 0,
  factor: 1,
  k: 1, cx: 0, cy: 0, bbox: null, needFit: true, userAdjusted: false, group: null,
  backend: null, live: true, abort: null, liveTimer: null, custom: [], hidden: [],
  models: { gen: "", run: "", single: "", judge: "" }, mcpCustom: [],   // мета-агент, исполнение, судья
  answer: null, baseline: null, judge: null, query: "",
};

/* ─────────────────────────── Утилиты ─────────────────────────── */
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
// esc() экранирует символы разметки, но не схему адреса: «javascript:…» прошло бы
// целиком и выполнилось по клику. Ссылки строим только на http(s).
const safeUrl = (u) => (/^https?:\/\//i.test(String(u || "")) ? String(u) : "");
// toLocaleString у строки возвращает её же: в импортированном файле на месте числа
// может оказаться разметка. Приводим к числу, непригодное показываем прочерком.
const nfmt = (n) => (Number.isFinite(Number(n)) ? Number(n).toLocaleString("ru-RU").replace(/,/g, " ") : "—");
const trunc = (s, n) => (s.length > n ? s.slice(0, n - 1) + "…" : s);

function fmtTime(sec) {
  if (sec < 60) return sec.toFixed(1).replace(".", ",") + " с";
  const m = Math.floor(sec / 60), s = Math.round(sec % 60);
  return `${m} мин ${String(s).padStart(2, "0")} с`;
}

function parseTarget(auto) {
  const m = /(\d+)\s*мин/.exec(auto), s = /([\d,.]+)\s*с(?!\w)/.exec(auto);
  return (m ? +m[1] * 60 : 0) + (s ? parseFloat(s[1].replace(",", ".")) || 0 : 0);
}

function el(tag, attrs = {}, ...kids) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  kids.forEach((k) => {
    if (k === null || k === undefined) return;
    n.appendChild(typeof k === "object" ? k : document.createTextNode(String(k)));
  });
  return n;
}

/* ─────────────────────────── Разбор конфигурации ─────────────────────────── */
function indexAgents(p) {
  const map = new Map();
  if (p.kind === "mas") {
    const c = p.config.coordinator;
    map.set(c.name, { ...c, isCoord: true, role: roleOf(c.name, true) });
    (p.config.workers || []).forEach((w) => map.set(w.name, { ...w, isCoord: false, role: roleOf(w.name, false) }));
  } else {
    (p.config.agents || []).forEach((a) => map.set(a.name, { ...a, isCoord: false, role: roleOf(a.name, false) }));
  }
  return map;
}

function allTools(p) {
  const used = new Map();
  for (const a of S.agents.values()) (a.tools || []).forEach((t) => {
    if (!used.has(t)) used.set(t, []);
    used.get(t).push(a.name);
  });
  return used;
}

function pipelineDepth(node) {
  if (!node) return 0;
  if (node.type === "agent") return 1;
  const kids = (node.children || []).map(pipelineDepth);
  if (!kids.length) return 1;
  return node.type === "parallel" ? Math.max(...kids) : kids.reduce((a, b) => a + b, 0);
}

/* ─────────────────────────── Раскладка графа ─────────────────────────── */
const NW = 204, NH = 60, HGAP = 54, VGAP = 24, JUNC = 38, PAD = 24, LOOPTOP = 26;

function measure(n) {
  if (n.type === "agent" || !(n.children || []).length)
    return { w: NW, h: NH, node: { ...n, type: "agent", agent_name: n.agent_name || "?" } };
  const kids = n.children.map(measure);
  if (n.type === "parallel") {
    return { w: Math.max(...kids.map((k) => k.w)) + 2 * JUNC,
             h: kids.reduce((s, k) => s + k.h, 0) + VGAP * (kids.length - 1), kids, node: n };
  }
  const seqW = kids.reduce((s, k) => s + k.w, 0) + HGAP * (kids.length - 1);
  const seqH = Math.max(...kids.map((k) => k.h));
  if (n.type === "loop") return { w: seqW + 2 * PAD, h: seqH + 2 * PAD + LOOPTOP, kids, node: n };
  return { w: seqW, h: seqH, kids, node: n };
}

function layoutMAW(pipeline, stageW) {
  const nodes = [], edges = [], groups = [], junctions = [];
  const m = measure(pipeline);

  function seq(kids, x, y, boxH) {
    let cx = x, prev = null, entry = null;
    kids.forEach((k) => {
      const r = place(k, cx, y + (boxH - k.h) / 2);
      if (prev) edges.push({ from: prev.exit, to: r.entry, fromName: prev.name, toName: r.name });
      else entry = r.entry;
      prev = r; cx += k.w + HGAP;
    });
    return { entry, exit: prev.exit, name: prev.name, first: kids.length ? entry : null };
  }

  function place(m, x, y) {
    const n = m.node;
    if (n.type === "agent") {
      nodes.push({ name: n.agent_name, x, y, w: NW, h: NH });
      return { entry: { x, y: y + NH / 2 }, exit: { x: x + NW, y: y + NH / 2 }, name: n.agent_name };
    }
    if (n.type === "parallel") {
      const sp = { x: x + JUNC / 2, y: y + m.h / 2 }, mp = { x: x + m.w - JUNC / 2, y: y + m.h / 2 };
      junctions.push(sp, mp);
      let cy = y;
      m.kids.forEach((k) => {
        const r = place(k, x + JUNC, cy);
        edges.push({ from: sp, to: r.entry, toName: r.name });
        edges.push({ from: r.exit, to: mp, fromName: r.name });
        cy += k.h + VGAP;
      });
      return { entry: sp, exit: mp, name: null };
    }
    if (n.type === "loop") {
      groups.push({ x, y, w: m.w, h: m.h, kind: "loop",
                    label: `цикл${n.max_iterations ? " ×" + n.max_iterations : ""}` });
      const r = seq(m.kids, x + PAD, y + PAD + LOOPTOP, m.h - 2 * PAD - LOOPTOP);
      edges.push({ from: r.exit, to: r.entry, loopback: true, top: y + 14 });
      return r;
    }
    return seq(m.kids, x, y, m.h);
  }

  // Ширина переноса подбирается под сцену: на узком экране цепочка складывается
  // в несколько рядов, иначе граф пришлось бы сильно уменьшать.
  const WRAP_W = Math.min(1050, Math.max(560, (stageW || 1150) / 1.15));
  if (pipeline.type === "sequential" && m.kids && m.kids.length > 1 && m.w > WRAP_W) {
    const rows = [];
    let cur = [], curW = 0;
    m.kids.forEach((k) => {
      const add = curW ? curW + HGAP + k.w : k.w;
      if (cur.length && add > WRAP_W) { rows.push(cur); cur = [k]; curW = k.w; }
      else { cur.push(k); curW = add; }
    });
    rows.push(cur);

    let y = 0, prevExit = null;
    rows.forEach((row) => {
      const rowH = Math.max(...row.map((k) => k.h));
      let x = 0, first = null, last = null;
      row.forEach((k) => {
        const r = place(k, x, y + (rowH - k.h) / 2);
        if (!first) first = r;
        if (last) edges.push({ from: last.exit, to: r.entry, fromName: last.name, toName: r.name });
        last = r; x += k.w + HGAP;
      });
      if (prevExit) edges.push({ from: prevExit, to: first.entry, wrap: y - 34, toName: first.name });
      prevExit = last.exit;
      y += rowH + 68;
    });
  } else {
    place(m, 0, 0);
  }
  return { nodes, edges, groups, junctions };
}

function layoutMAS(cfg) {
  const nodes = [], edges = [], groups = [], junctions = [];
  const ws = cfg.workers || [];
  if (!ws.length) {
    nodes.push({ name: cfg.coordinator.name, x: 0, y: 0, w: NW, h: NH, coord: true });
    return { nodes, edges, groups, junctions };
  }
  const per = ws.length > 3 ? Math.ceil(ws.length / 2) : ws.length;
  const rows = [];
  for (let i = 0; i < ws.length; i += per) rows.push(ws.slice(i, i + per));
  const rowW = (n) => n * NW + (n - 1) * 30;
  const maxW = Math.max(...rows.map((r) => rowW(r.length)));
  const cx = maxW / 2 - NW / 2;

  nodes.push({ name: cfg.coordinator.name, x: cx, y: 0, w: NW, h: NH, coord: true });
  rows.forEach((row, ri) => {
    const x0 = (maxW - rowW(row.length)) / 2;
    const y = NH + 92 + ri * (NH + 58);
    row.forEach((w, i) => {
      const x = x0 + i * (NW + 30);
      nodes.push({ name: w.name, x, y, w: NW, h: NH });
      edges.push({ from: { x: cx + NW / 2, y: NH }, to: { x: x + NW / 2, y }, vertical: true, toName: w.name });
    });
  });
  return { nodes, edges, groups, junctions };
}

/* ─────────────────────────── Отрисовка графа ─────────────────────────── */
function bezier(e) {
  const { from: a, to: b } = e;
  if (e.loopback) {
    const t = e.top, r = 10;
    return `M ${a.x} ${a.y} L ${a.x + 20 - r} ${a.y} Q ${a.x + 20} ${a.y} ${a.x + 20} ${a.y - r}` +
           ` L ${a.x + 20} ${t + r} Q ${a.x + 20} ${t} ${a.x + 20 - r} ${t}` +
           ` L ${b.x - 20 + r} ${t} Q ${b.x - 20} ${t} ${b.x - 20} ${t + r}` +
           ` L ${b.x - 20} ${b.y - r} Q ${b.x - 20} ${b.y} ${b.x - 20 + r} ${b.y} L ${b.x} ${b.y}`;
  }
  if (e.wrap !== undefined) {
    const my = e.wrap, r = 10, a2 = e.from, b2 = e.to;
    return `M ${a2.x} ${a2.y} L ${a2.x + 20 - r} ${a2.y} Q ${a2.x + 20} ${a2.y} ${a2.x + 20} ${a2.y + r}` +
           ` L ${a2.x + 20} ${my - r} Q ${a2.x + 20} ${my} ${a2.x + 20 - r} ${my}` +
           ` L ${b2.x - 20 + r} ${my} Q ${b2.x - 20} ${my} ${b2.x - 20} ${my + r}` +
           ` L ${b2.x - 20} ${b2.y - r} Q ${b2.x - 20} ${b2.y} ${b2.x - 20 + r} ${b2.y} L ${b2.x} ${b2.y}`;
  }
  if (e.vertical) {
    const dy = Math.max(28, (b.y - a.y) / 2);
    return `M ${a.x} ${a.y} C ${a.x} ${a.y + dy}, ${b.x} ${b.y - dy}, ${b.x} ${b.y}`;
  }
  const dx = Math.max(26, (b.x - a.x) / 2);
  return `M ${a.x} ${a.y} C ${a.x + dx} ${a.y}, ${b.x - dx} ${b.y}, ${b.x} ${b.y}`;
}

function drawNode(nd, i) {
  const a = S.agents.get(nd.name) || { name: nd.name, role: roleOf(nd.name, !!nd.coord), tools: [] };
  const pos = el("g", { transform: `translate(${nd.x},${nd.y})` });
  const g = el("g", { class: `node ${ROLE_CLASS[a.role.kind]} ${a.isCoord || nd.coord ? "coord" : ""}`,
                      "data-agent": nd.name, style: `animation-delay:${i * 55}ms` });

  g.appendChild(el("rect", { class: "ring", x: -4, y: -4, width: NW + 8, height: NH + 8, rx: 15 }));
  g.appendChild(el("rect", { class: "node-box", width: NW, height: NH, rx: 12 }));
  g.appendChild(el("rect", { class: "node-av-bg", x: 13, y: NH / 2 - 16, width: 32, height: 32, rx: 9 }));

  const glyph = el("g", { class: "node-av", transform: `translate(17,${NH / 2 - 12})` });
  ICONS[a.role.icon].forEach((d) => glyph.appendChild(el("path", { d })));
  g.appendChild(glyph);

  // Имена от мета-агента бывают длинными (особенно русские): кегль подбирается
  // под ширину карточки, а ниже пола 8px имя обрезается — полное видно в подсказке.
  const TEXT_W = NW - 53 - 12;
  const fit = (s, max, min) => {
    const size = Math.min(max, TEXT_W / (Math.max(s.length, 1) * 0.6));
    return size >= min
      ? { size, text: s }
      : { size: min, text: trunc(s, Math.floor(TEXT_W / (min * 0.6))) };
  };

  const name = fit(String(nd.name || ""), 11.5, 8);
  const nameEl = el("text", { class: "node-name", x: 53, y: NH / 2 - 3, style: `font-size:${name.size.toFixed(1)}px` }, name.text);
  nameEl.appendChild(el("title", {}, String(nd.name)));
  g.appendChild(nameEl);

  const sub = fit(a.output_key ? "→ " + a.output_key : trunc(a.description || "", 30), 10.5, 7.5);
  const subEl = el("text", { class: "node-sub", x: 53, y: NH / 2 + 15, style: `font-size:${sub.size.toFixed(1)}px` }, sub.text);
  subEl.appendChild(el("title", {}, a.output_key ? "→ " + a.output_key : String(a.description || "")));
  g.appendChild(subEl);

  (a.tools || []).forEach((t, k) => {
    const dot = el("circle", { class: "tool-dot", cx: NW - 14 - k * 11, cy: 15, r: 3.5 });
    dot.appendChild(el("title", {}, t));
    g.appendChild(dot);
  });

  pos.appendChild(g);
  return pos;
}

function renderGraph() {
  const svg = $("graph");
  svg.innerHTML = "";
  if (!S.preset) return;

  const stageW = svg.getBoundingClientRect().width;
  const L = S.preset.kind === "mas"
    ? layoutMAS(S.preset.config)
    : layoutMAW(S.preset.config.pipeline, stageW);

  const defs = el("defs");
  defs.innerHTML =
    '<filter id="glow" x="-40%" y="-40%" width="180%" height="180%">' +
    '<feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>' +
    '<marker id="arw" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">' +
    '<path d="M0 0 L10 5 L0 10z" fill="currentColor"/></marker>';
  defs.setAttribute("color", getComputedStyle(document.documentElement).getPropertyValue("--arrow").trim() || "#94A6BC");
  svg.appendChild(defs);

  const gGroups = el("g"), gEdges = el("g"), gNodes = el("g");
  svg.append(gGroups, gEdges, gNodes);

  L.groups.forEach((gr) => {
    gGroups.appendChild(el("rect", { class: `group-box ${gr.kind}`, x: gr.x, y: gr.y, width: gr.w, height: gr.h, rx: 14 }));
    // подпись «врезается» в рамку, как legend у fieldset
    gGroups.appendChild(el("rect", { class: "group-label-bg", x: gr.x + 12, y: gr.y - 9,
                                     width: gr.label.length * 7 + 10, height: 18, rx: 5 }));
    gGroups.appendChild(el("text", { class: `group-label ${gr.kind}`, x: gr.x + 17, y: gr.y + 4 }, gr.label));
  });

  L.edges.forEach((e) => {
    const p = el("path", { class: "edge" + (e.loopback ? " loopback" : ""), d: bezier(e), "marker-end": e.loopback ? "" : "url(#arw)" });
    if (e.toName) p.setAttribute("data-to", e.toName);
    if (e.fromName) p.setAttribute("data-from", e.fromName);
    gEdges.appendChild(p);
  });

  L.junctions.forEach((j) => gEdges.appendChild(el("circle", { class: "junction", cx: j.x, cy: j.y, r: 6 })));
  L.nodes.forEach((n, i) => gNodes.appendChild(drawNode(n, i)));

  // границы содержимого -> камера (масштаб + панорама)
  const xs = L.nodes.map((n) => n.x), ys = L.nodes.map((n) => n.y);
  let x0 = Math.min(...xs), y0 = Math.min(...ys);
  let x1 = Math.max(...L.nodes.map((n) => n.x + n.w)), y1 = Math.max(...L.nodes.map((n) => n.y + n.h));
  L.groups.forEach((g) => { x0 = Math.min(x0, g.x); y0 = Math.min(y0, g.y); x1 = Math.max(x1, g.x + g.w); y1 = Math.max(y1, g.y + g.h); });
  S.bbox = { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
  S.needFit = true;
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  fitView();
}

/* ─────────────────────────── Камера ─────────────────────────── */
function applyView() {
  const svg = $("graph"), r = svg.getBoundingClientRect();
  if (!r.width || !S.bbox) return;
  const w = r.width / S.k, h = r.height / S.k;
  svg.setAttribute("viewBox", `${S.cx - w / 2} ${S.cy - h / 2} ${w} ${h}`);
}

function fitView() {
  const svg = $("graph"), r = svg.getBoundingClientRect();
  if (!r.width || !S.bbox) return;   // контейнер ещё не измерен (фоновая вкладка) — довесит ResizeObserver
  S.needFit = false;
  S.userAdjusted = false;
  const pad = 56;
  const k = Math.min((r.width - pad) / S.bbox.w, (r.height - pad) / S.bbox.h);
  S.k = Math.min(Math.max(k, 0.62), 1.6);
  S.cx = S.bbox.x + S.bbox.w / 2;
  S.cy = S.bbox.y + S.bbox.h / 2;
  applyView();
}

function zoomBy(f) { S.userAdjusted = true; S.k = Math.min(2.4, Math.max(0.35, S.k * f)); applyView(); }

function initCamera() {
  const svg = $("graph");
  let drag = null;
  svg.addEventListener("pointerdown", (e) => {
    drag = { x: e.clientX, y: e.clientY, cx: S.cx, cy: S.cy };
    svg.setPointerCapture(e.pointerId); svg.classList.add("drag");
  });
  svg.addEventListener("pointermove", (e) => {
    if (!drag) return;
    S.userAdjusted = true;
    S.cx = drag.cx - (e.clientX - drag.x) / S.k;
    S.cy = drag.cy - (e.clientY - drag.y) / S.k;
    applyView();
  });
  const stop = (e) => { drag = null; svg.classList.remove("drag"); };
  svg.addEventListener("pointerup", stop);
  svg.addEventListener("pointercancel", stop);
  // Колесо зумит только с Ctrl/⌘ (и пинчем на трекпаде — он приходит с ctrlKey),
  // иначе прокрутка достаётся странице: в узкой раскладке иначе не пролистать.
  svg.addEventListener("wheel", (e) => {
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault();
    zoomBy(Math.exp(-e.deltaY * 0.0016));
  }, { passive: false });
  $("z-in").addEventListener("click", () => zoomBy(1.2));
  $("z-out").addEventListener("click", () => zoomBy(1 / 1.2));
  $("z-fit").addEventListener("click", fitView);
  window.addEventListener("resize", () => applyView());

  // Стадия может быть измерена нулём (страница открыта в фоновой вкладке) —
  // тогда масштаб подбирается, как только контейнер получит размер.
  if (window.ResizeObserver) {
    new ResizeObserver(() => {
      if (!$("graph").getBoundingClientRect().width) return;
      // после ручного зума/панорамы масштаб сохраняем, иначе — подгоняем под новый размер окна
      S.needFit || !S.userAdjusted ? fitView() : applyView();
    }).observe($("stage"));
  }
}

/* ─────────────────────────── Анимация потока ─────────────────────────── */
function flyToken(pathEl) {
  if (!pathEl.isConnected) return;
  const len = pathEl.getTotalLength();
  const dot = el("circle", { class: "token", r: 4 });
  pathEl.parentNode.appendChild(dot);
  const t0 = performance.now(), dur = 620 / S.speed;
  (function tick(t) {
    if (!pathEl.isConnected) { dot.remove(); return; }   // сцена перерисована — анимацию бросаем
    const k = Math.min(1, (t - t0) / dur);
    const pt = pathEl.getPointAtLength(len * k);
    dot.setAttribute("cx", pt.x); dot.setAttribute("cy", pt.y);
    dot.setAttribute("opacity", k > .85 ? (1 - k) / .15 : 1);
    if (k < 1) requestAnimationFrame(tick); else dot.remove();
  })(t0);
}

function nodeByName(name) {
  return [...$("graph").querySelectorAll(".node")].find((n) => n.dataset.agent === name) || null;
}

function edgesTo(name) {
  return [...$("graph").querySelectorAll(".edge")].filter((e) => e.dataset.to === name);
}

/** Агенты одной parallel-ветки остаются подсвеченными вместе: у их событий общий `group`. */
function activate(name, group) {
  const svg = $("graph");
  if (!group || group !== S.group) {
    svg.querySelectorAll(".node.active").forEach((n) => { n.classList.remove("active"); n.classList.add("done"); });
    svg.querySelectorAll(".edge.live").forEach((e) => e.classList.remove("live"));
  }
  S.group = group || null;
  const node = nodeByName(name);
  if (node) { node.classList.add("active"); node.classList.remove("done"); }
  edgesTo(name).forEach((e) => { e.classList.add("live"); flyToken(e); });
}

/* ─────────────────────────── Лента выполнения ─────────────────────────── */
function pushMessage(e) {
  const a = S.agents.get(e.agent) || { role: roleOf(e.agent, false) };
  const cls = ROLE_CLASS[a.role.kind];
  const badges = [e.tool, e.tool2].filter(Boolean);
  const wrap = document.createElement("div");
  wrap.className = `msg ${cls}` + (e.final ? " final" : "");
  wrap.innerHTML =
    `<div class="msg-head">
       <span class="msg-av">
         <svg viewBox="0 0 24 24" class="ic">${ICONS[a.role.icon].map((d) => `<path d="${d}"/>`).join("")}</svg>
       </span>
       <span class="msg-name">${esc(e.agent)}</span>
       <span class="msg-phase">${esc(e.phase)}${e.group ? " · параллельно" : ""}</span>
     </div>
     <div class="msg-text">${esc(e.text)}</div>` +
    (e.io ? `<details class="msg-io">
       <summary>вход и выход</summary>
       <div class="io-label">инструкция агента</div>
       <pre>${esc(e.io.instruction || "—")}</pre>
       <div class="io-label">на входе из состояния</div>
       <pre>${esc(e.io.incoming || "—")}</pre>
       <div class="io-label">результат агента${e.io.outputKey ? " → " + esc(e.io.outputKey) : ""}</div>
       <pre>${esc(e.io.output || "—")}</pre>
     </details>` : "") +
    badges.map((b, bi) => `<div class="msg-tool">
       <svg viewBox="0 0 24 24" class="ic"><path d="M15 3a5 5 0 00-4.6 7L3 17.4 6.6 21l7.4-7.4A5 5 0 0021 9l-3 3-3-3 3-3a5 5 0 00-3-3z"/></svg>
       ${esc(b)}${bi === 0 && e.toolNote ? " · " + esc(e.toolNote) : ""}</div>`).join("");
  const feed = $("feed");
  feed.appendChild(wrap);
  feed.scrollTop = feed.scrollHeight;
}

/* ─────────────────────────── Плеер ─────────────────────────── */
// Пауза между шагами при воспроизведении. Реальные шаги бывают по 20+ секунд —
// на счётчике времени они учитываются полностью, а ждать столько на показе нельзя.
const PACE_MIN = 900, PACE_MAX = 2800;
const pace = (ms) => Math.min(PACE_MAX, Math.max(PACE_MIN, ms));
function resetRun() {
  clearTimeout(S.timer);
  S.idx = 0; S.playing = false; S.tokens = 0; S.seconds = 0; S.group = null;
  $("feed").innerHTML = '<div class="empty">Журнал появится после запуска системы</div>';
  $("p-fill").style.width = "0%";
  $("m-tokens").textContent = "0";
  $("m-time").textContent = "0,0 с";
  $("graph").querySelectorAll(".node").forEach((n) => n.classList.remove("active", "done"));
  $("graph").querySelectorAll(".edge.live").forEach((e) => e.classList.remove("live"));
  setPlayIcon(false);
}

function setPlayIcon(playing) {
  const d = playing ? '<path d="M7 4h4v16H7zM13 4h4v16h-4z"/>' : '<path d="M7 4l12 8-12 8z"/>';
  $("p-play-icon").innerHTML = d;
  $("btn-run").innerHTML = `<svg viewBox="0 0 24 24" class="ic">${d}</svg>` + (playing ? "Пауза" : "Запустить");
}

function stepOnce() {
  if (S.idx >= S.events.length) { S.playing = false; setPlayIcon(false); return; }
  const e = S.events[S.idx++];
  if (S.idx === 1) $("feed").innerHTML = "";
  activate(e.agent, e.group);
  pushMessage(e);
  S.tokens += e.tokens || 0;
  S.seconds += (e.ms * S.factor) / 1000;
  $("m-tokens").textContent = nfmt(S.tokens);
  $("m-time").textContent = fmtTime(S.seconds);
  $("p-fill").style.width = (100 * S.idx / S.events.length).toFixed(1) + "%";
  if (S.idx >= S.events.length) {
    const last = $("graph").querySelector(".node.active");
    if (last) { last.classList.remove("active"); last.classList.add("done"); }
    S.playing = false; setPlayIcon(false);
    return;
  }
  if (S.playing) S.timer = setTimeout(stepOnce, pace(e.ms) / S.speed);
}

function play() {
  if (!S.preset || !S.events.length) return;
  showTab("feed");
  if (S.idx >= S.events.length) resetRun();
  S.playing = true; setPlayIcon(true);
  stepOnce();
}

function pause() { S.playing = false; clearTimeout(S.timer); setPlayIcon(false); }

/* ─────────────────────────── Панель инспектора ─────────────────────────── */
function highlightJSON(obj) {
  const raw = esc(JSON.stringify(obj, null, 2));
  return raw
    .replace(/&quot;([^&]*?)&quot;(\s*:)/g, '<span class="k">"$1"</span>$2')
    .replace(/:\s&quot;([\s\S]*?)&quot;(,?)$/gm, ': <span class="s">"$1"</span>$2')
    .replace(/:\s(-?\d+(?:\.\d+)?)(,?)$/gm, ': <span class="n">$1</span>$2')
    .replace(/:\s(true|false|null)(,?)$/gm, ': <span class="b">$1</span>$2');
}

function renderInspector() {
  const p = S.preset;

  // Агенты
  const cards = [...S.agents.values()].map((a) => {
    const cls = ROLE_CLASS[a.role.kind];
    const pills = [
      a.isCoord ? '<span class="pill">координатор</span>' : "",
      a.output_key ? `<span class="pill pill-out">${esc(a.output_key)}</span>` : "",
      ...(a.tools || []).map((t) => `<span class="pill pill-tool">${esc(t)}</span>`),
      a.model ? `<span class="pill">${esc(a.model)}</span>` : "",
    ].join("");
    return `<div class="acard ${cls}">
      <div class="acard-head">
        <span class="msg-av">
          <svg viewBox="0 0 24 24" class="ic">${ICONS[a.role.icon].map((d) => `<path d="${d}"/>`).join("")}</svg>
        </span>
        <span class="acard-name">${esc(a.name)}</span>
        <span class="acard-role">${a.isCoord ? "маршрутизация" : "исполнитель"}</span>
      </div>
      ${a.description ? `<div class="acard-instr" style="margin-bottom:7px">${esc(a.description)}</div>` : ""}
      <div class="acard-instr">${esc(trunc(a.instruction || "—", 420))}</div>
      <div class="acard-foot">${pills}</div>
    </div>`;
  });
  $("agent-cards").innerHTML = cards.join("");

  // Инструменты
  const used = allTools(p);
  $("tool-cards").innerHTML = used.size === 0
    ? '<div class="empty">Система не использует внешние инструменты</div>'
    : [...used.entries()].map(([t, agents]) => `<div class="tcard">
        <span class="tico"><svg viewBox="0 0 24 24" class="ic"><path d="M15 3a5 5 0 00-4.6 7L3 17.4 6.6 21l7.4-7.4A5 5 0 0021 9l-3 3-3-3 3-3a5 5 0 00-3-3z"/></svg></span>
        <div>
          <div class="tcard-name">${esc(t)}</div>
          <div class="tcard-desc">${esc(window.MCP_SERVERS[t] || "MCP-сервер")}</div>
          <div class="tcard-used">агенты: ${esc(agents.join(", "))}</div>
        </div></div>`).join("");

  // JSON
  $("json-label").textContent = p.kind === "mas" ? "MASConfig · config.json" : "MAWConfig · config.json";
  $("json").innerHTML = highlightJSON(p.config);
}

/** Лёгкий рендер Markdown: агенты отвечают заголовками, списками и таблицами. */
function mdToHtml(src) {
  const lines = esc(String(src || "")).split("\n");
  const out = [];
  let list = null, table = null;

  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  const closeTable = () => {
    if (table) {
      out.push("<table class=\"ans-table\"><thead><tr>" +
        table.head.map((h) => `<th>${inline(h)}</th>`).join("") + "</tr></thead><tbody>" +
        table.rows.map((r) => "<tr>" + r.map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>").join("") +
        "</tbody></table>");
      table = null;
    }
  };
  const inline = (s) => s
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<i>$2</i>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");

  for (const raw of lines) {
    const line = raw.trimEnd();
    const cells = line.trim().startsWith("|") && line.trim().endsWith("|")
      ? line.trim().slice(1, -1).split("|").map((c) => c.trim()) : null;

    if (cells) {                                   // таблица
      closeList();
      if (!table) table = { head: cells, rows: [] };
      else if (!cells.every((c) => /^:?-{2,}:?$/.test(c) || c === "")) table.rows.push(cells);
      continue;
    }
    closeTable();

    if (!line.trim()) { closeList(); continue; }
    if (/^#{1,6}\s/.test(line)) { closeList(); out.push(`<h5>${inline(line.replace(/^#+\s*/, ""))}</h5>`); continue; }
    if (/^(-{3,}|—{3,}|\*{3,})$/.test(line.trim())) { closeList(); out.push("<hr>"); continue; }

    const ul = line.match(/^\s*[-•*]\s+(.*)$/);
    const ol = line.match(/^\s*(\d+)[.)]\s+(.*)$/);
    if (ul || ol) {
      const want = ul ? "ul" : "ol";
      if (list !== want) { closeList(); out.push(`<${want}>`); list = want; }
      out.push(`<li>${inline((ul || ol)[ul ? 1 : 2])}</li>`);
      continue;
    }
    closeList();
    out.push(`<p>${inline(line)}</p>`);
  }
  closeList(); closeTable();
  return out.join("");
}

/* ─────────────────────────── Ответ системы ───────────────────────────
 * Для тех, кто смотрит демонстрацию, результат — это ответ на задачу, а не сам пайплайн.
 * Здесь же сравнение с одной моделью и вердикт судьи.
 */
/* ─────────────────── Короткий ответ ───────────────────
 * Агенты пишут обстоятельно: исходные данные, расчёт, источники. На демонстрации
 * сначала хотят увидеть сам ответ, а рассуждения — уже потом. Вытаскиваем вывод,
 * но только когда он явно помечен: угадывать «где тут ответ» по всему тексту
 * означало бы иногда показывать в рамке не то, а это хуже, чем не показывать.
 */
// Граница слова \b в JavaScript считает словом только ASCII: после «вывод» она не
// срабатывает, поэтому проверяем строку явно, а не одной регуляркой.
const CONCLUSION_WORDS = /^(вывод|итог|ответ|заключен|резюме|рекоменд)/i;

const stripMd = (line) => String(line)
  .replace(/^[#\s]*/, "").replace(/\*\*/g, "").replace(/^\d+[.)]\s*/, "").trim();

// Заголовок вида «Вывод:», «## Итоги», «**Выводы и рекомендации**», «3. Заключение».
// Ограничение длины отсекает обычные предложения, начинающиеся с «Ответ ...».
function isConclusionHeading(line) {
  const t = stripMd(line);
  if (!t || t.length > 60) return false;
  return CONCLUSION_WORDS.test(t.replace(/[:：]\s*$/, ""));
}

// Однострочный вид: «Ответ: 391».
function inlineConclusion(line) {
  const m = stripMd(line).match(/^(вывод|итог|ответ|заключение|резюме)[а-яё]*\s*[:：]\s*(.+)$/i);
  return m ? m[2].trim() : "";
}

function shortAnswer(text) {
  const lines = String(text || "").split("\n");
  // Идём с конца: итоговый вывод обычно последний, а по дороге встречаются
  // промежуточные «итого» внутри расчёта.
  for (let i = lines.length - 1; i >= 0; i--) {
    if (!isConclusionHeading(lines[i])) continue;
    const body = [];
    for (let j = i + 1; j < lines.length && body.length < 12; j++) {
      if (isConclusionHeading(lines[j])) break;
      if (/^\s*(-{3,}|={3,})\s*$/.test(lines[j])) break;
      body.push(lines[j]);
    }
    const out = body.join("\n").trim();
    if (out) return out;
  }
  for (let i = lines.length - 1; i >= 0; i--) {
    const one = inlineConclusion(lines[i]);
    if (one) return one;
  }
  return "";
}

function renderAnswer() {
  const host = $("answer");
  const blocks = [];

  if (S.answer) {
    // Рамка идёт ПОСЛЕ полного текста: сверху разбор и расчёт, внизу — сам ответ
    // на заданный вопрос. Полный текст при этом никуда не девается.
    const short = shortAnswer(S.answer.text);
    blocks.push(`<div class="ans-card system">
      <h4>Ответ мультиагентной системы</h4>
      <div class="ans-text md">${mdToHtml(S.answer.text)}</div>
      ${short ? `<div class="ans-short"><span class="ans-short-label">Ответ на вопрос</span>
         <div class="ans-text md">${mdToHtml(short)}</div></div>` : ""}
      ${S.answer.meta ? `<div class="ans-meta">${esc(S.answer.meta)}</div>` : ""}
    </div>`);
  }
  if (S.baseline) {
    blocks.push(`<div class="ans-card single">
      <h4>Ответ одной модели, без системы</h4>
      <div class="ans-text md">${mdToHtml(S.baseline.answer)}</div>
      <div class="ans-meta">${esc(S.baseline.model || "")} · ${esc(nfmt(S.baseline.tokens || 0))} токенов · ${esc(String(S.baseline.seconds ?? "—"))} с</div>
    </div>`);
  }
  if (S.judge) {
    const w = S.judge.winner === "system" ? "system" : S.judge.winner === "single" ? "single" : "tie";
    const label = w === "system" ? "Судья: лучше ответ системы"
      : w === "single" ? "Судья: лучше ответ одной модели" : "Судья: ничья";
    blocks.push(`<div class="ans-card judge">
      <h4>Оценка судьёй</h4>
      <div class="ans-winner ${w}">${esc(label)}</div>
      <div class="ans-text">${esc(S.judge.verdict)}</div>
      <div class="ans-meta">судья: ${esc(S.judge.model || "")}${S.judge.tokens ? " · " + esc(nfmt(S.judge.tokens)) + " токенов" : ""}</div>
    </div>`);
  }
  host.innerHTML = blocks.length ? blocks.join("")
    : '<div class="empty">Ответ системы появится после запуска</div>';

  // Кнопки живут, пока есть бэкенд: сравнение и оценку можно перезапустить и поверх записанных.
  const live = !!S.backend;
  const bs = $("btn-baseline"), bj = $("btn-judge");
  bs.disabled = !(live && S.answer);
  bj.disabled = !(live && S.answer && S.baseline);
  bs.textContent = S.baseline ? "Сравнить заново" : "Сравнить с одной моделью";
  bj.textContent = S.judge ? "Оценить заново" : "Оценить судьёй";
  bs.title = live ? "Решить ту же задачу одной моделью без системы"
                  : "Доступно в живом режиме: запустите gui/run.py";
  bj.title = live ? "Независимый судья сравнит оба ответа"
                  : "Доступно в живом режиме: запустите gui/run.py";
}

async function runBaseline() {
  if (!S.backend || !S.answer) return;
  const btn = $("btn-baseline");
  S.judge = null;                       // прошлый вердикт относится к старому сравнению
  btn.disabled = true; btn.textContent = "Одна модель отвечает…";
  try {
    const r = await fetch("api/baseline", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: S.query || $("query").value, model: S.models.single }),
    });
    const d = await readJson(r, "ответ одной модели");
    if (!d.ok) throw new Error(d.error);
    S.baseline = d;
  } catch (err) {
    S.baseline = { answer: "Не удалось получить ответ: " + err.message, model: "—", tokens: 0, seconds: "—" };
  } finally {
    btn.textContent = "Сравнить с одной моделью";
    if (S.preset) { S.preset.baseline = S.baseline; S.preset.judge = null; persistPreset(); }
    renderAnswer();
  }
}

// Прогресс судьи: он работает минутами и без обратной связи выглядит зависшим.
// Устроен так же, как прогресс генерации конфигурации.
function judgeProgress() {
  const box = $("judge-progress");
  const fill = $("judge-progress-fill");
  const log = $("judge-progress-log");
  const t0 = Date.now();
  let pct = 6;
  const timer = setInterval(() => {
    $("judge-progress-time").textContent = Math.round((Date.now() - t0) / 1000) + " с";
  }, 250);
  box.classList.remove("hidden");
  log.innerHTML = "";
  fill.style.width = pct + "%";
  return {
    step(text, percent) {
      $("judge-progress-step").textContent = text;
      if (percent != null) pct = percent;
      fill.style.width = pct + "%";
    },
    note(text) {
      const line = document.createElement("div");
      line.textContent = text;
      log.appendChild(line);
      log.scrollTop = log.scrollHeight;
    },
    bump(delta) { pct = Math.min(92, pct + delta); fill.style.width = pct + "%"; },
    stop() { clearInterval(timer); box.classList.add("hidden"); },
  };
}

async function runJudge() {
  if (!S.backend || !S.answer || !S.baseline) return;
  const btn = $("btn-judge");
  btn.disabled = true; btn.textContent = "Судья сравнивает…";
  const ui = judgeProgress();
  let tokens = 0;
  try {
    ui.step("Судья читает оба ответа", 10);
    let done = null;
    for await (const ev of sseEvents("api/judge_stream",
        { query: S.query || $("query").value, model: S.models.judge,
          system_answer: S.answer.text, single_answer: S.baseline.answer })) {
      if (ev.type === "done") { done = ev; break; }
      if (ev.type === "agent_start") {
        ui.step("Судья проверяет числа", 25);
      } else if (ev.type === "tool") {
        ui.note("пересчёт в песочнице: " + (ev.tool || "sandbox-light"));
        ui.bump(12);
      } else if (ev.type === "tokens" || ev.type === "text") {
        tokens += ev.tokens || 0;
        ui.bump(3);
      }
    }
    if (!done) throw new Error("поток прервался");
    if (!done.ok) throw new Error(done.error || "судья не вернул вердикт");
    ui.step("Вердикт вынесен", 100);
    S.judge = done;
  } catch (err) {
    S.judge = { verdict: "Не удалось получить оценку: " + err.message, winner: "tie", model: "—" };
  } finally {
    ui.stop();
    btn.disabled = false;
    btn.textContent = "Оценить заново";
    if (S.preset) { S.preset.judge = S.judge; persistPreset(); }
    renderAnswer();
  }
}

/* ─────────────────────────── Выбор MCP-серверов ─────────────────────────── */
// Каталог берётся из самого FEDOT.MAS: он находит серверы по [tool.fedotmas.mcp]
// в mcp-servers/. Свой сервер подключается по ссылке — HttpMCPServer поддержан штатно.
function renderMcpAdded() {
  const box = $("new-mcp-added");
  if (!box) return;
  box.innerHTML = S.mcpCustom.map((m, i) => `
    <span class="mcp-chip">${esc(m.name)} · ${esc(m.url)}
      <button type="button" data-i="${i}" title="Убрать">✕</button></span>`).join("");
  box.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    S.mcpCustom.splice(Number(b.dataset.i), 1); renderMcpAdded();
  }));
}


/* Локальных серверов всего десяток, и под произвольную задачу из зала в них может
 * не оказаться нужного. Поэтому тот же запрос уходит во внешний реестр Smithery —
 * это та база, где ищет и сам FEDOT.MAS (ветка feat/meta-agent-mcp-discovery). */
let mcpSearchTimer = null;
let mcpSearchSeq = 0;

async function searchMcpRegistry(query) {
  const box = $("new-mcp-remote");
  if (!box) return;
  const q = (query || "").trim();
  const count = $("new-mcp-count");
  if (q.length < 3) {
    box.innerHTML = '<div class="mcp-remote-head">Опишите, что должен уметь сервер —'
      + ' найдём в реестре и подключим по ссылке. Индекс англоязычный:'
      + ' «geocoding» находит Mapbox и OpenStreetMap, «геокодирование» — ничего.</div>';
    if (count) count.textContent = "";
    return;
  }
  const seq = ++mcpSearchSeq;
  box.innerHTML = '<div class="mcp-remote-head">Ищем в реестре Smithery…</div>';
  let d;
  try {
    const r = await fetch(`api/mcp_search?q=${encodeURIComponent(q)}`, { cache: "no-store" });
    d = await readJson(r, "поиск в реестре");
  } catch (err) {
    if (seq !== mcpSearchSeq) return;
    box.innerHTML = `<div class="mcp-remote-head">Реестр недоступен: ${esc(err.message)}</div>`;
    return;
  }
  if (seq !== mcpSearchSeq) return;          // пришёл ответ на устаревший запрос
  const list = d.servers || [];
  if (count) count.textContent = list.length ? `найдено ${list.length}` : "";
  // Развёрнутые серверы реестра отвечают 401 без ключа Smithery — предупреждаем
  // заранее, иначе подключение молча сломает прогон.
  const warn = d.needs_key && list.length
    ? '<div class="mcp-warn">Для подключения нужен ключ Smithery: задайте'
      + ' <code>SMITHERY_API_KEY</code> и перезапустите стенд. Без него сервер'
      + ' ответит агенту ошибкой доступа.</div>'
    : "";
  if (!list.length) {
    box.innerHTML = `<div class="mcp-remote-head">${
      d.ok ? "В реестре ничего не нашлось — попробуйте другие слова"
           : esc(d.error || "реестр не ответил")}</div>`;
    return;
  }
  box.innerHTML = warn
    + '<div class="mcp-remote-head">Готовые серверы — подключаются по ссылке,'
    + ' ставить ничего не нужно</div>'
    + list.map((m, i) => `
      <div class="mcp-remote-row">
        <div class="mcp-remote-text">
          <div class="mcp-remote-title">
            <b>${esc(m.title)}</b>${m.verified ? '<span class="mcp-badge">проверен</span>' : ""}
          </div>
          <span class="mcp-remote-desc">${esc(m.description || "")}</span>
        </div>
        <button type="button" class="btn btn-mini" data-i="${i}">Подключить</button>
      </div>`).join("");
  box.querySelectorAll("button[data-i]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const m = list[Number(btn.dataset.i)];
      if (S.mcpCustom.some((x) => x.url === m.url)) return;
      S.mcpCustom.push({ name: m.name, url: m.url, source: "smithery" });
      btn.textContent = "Подключён";
      btn.disabled = true;
      renderMcpAdded();
    });
  });
}

function initMcpPicker() {
  const search = $("new-mcp-search");
  // В FEDOT.MAS (ветка feat/meta-agent-mcp-discovery) в реестр отправляют текст самой
  // задачи. У нас задачи русские, а индекс Smithery англоязычный: на «кадастр
  // недвижимость» он выдаёт «Polícia Rodoviária Federal», на «geocoding» — Mapbox и
  // OpenStreetMap. Проверено запросами, поэтому постановку не подставляем.
  if (search) search.addEventListener("input", () => {
    clearTimeout(mcpSearchTimer);                      // ищем, когда перестали печатать
    mcpSearchTimer = setTimeout(() => searchMcpRegistry(search.value), 700);
  });
  const add = $("new-mcp-add");
  if (add) add.addEventListener("click", () => {
    const name = ($("new-mcp-name").value || "").trim();
    const url = ($("new-mcp-url").value || "").trim();
    if (!name || !/^https?:\/\//.test(url)) {
      $("new-note").textContent = "Для своего сервера нужны имя и адрес вида https://…";
      return;
    }
    S.mcpCustom.push({ name, url });
    $("new-mcp-name").value = ""; $("new-mcp-url").value = "";
    $("new-note").textContent = "";
    renderMcpAdded();
  });
}

/* ─────────────────────────── Разбор трудоёмкости ─────────────────────────── */
// Итоговая трудоёмкость — не отдельная догадка модели, а сумма часов по подзадачам:
// цифру, которую показывают на защите, должно быть чем объяснить построчно.
// Числа приходят и из импортированного файла, где на их месте может оказаться
// разметка. Приводим к числу: непригодное превращается в прочерк, а не в тег.
const num = (v) => (Number.isFinite(Number(v)) ? String(Number(v)).replace(".", ",") : "—");
// Вид системы нужен и как подпись, и как класс — оба места должны быть предсказуемы.
const kindLabel = (k) => (k === "mas" ? "MAS" : "MAW");
const hoursText = (h) => `${num(h)} чел.-ч`;

// «1,1 дня», «3 дня», «5 дней», «1 день» — при дробном числе русский всегда просит
// родительный падеж единственного числа, при целом работают обычные правила счёта.
function daysWord(d) {
  if (!Number.isInteger(d)) return "дня";
  const last = d % 10, two = d % 100;
  if (two >= 11 && two <= 14) return "дней";
  if (last === 1) return "день";
  if (last >= 2 && last <= 4) return "дня";
  return "дней";
}
const daysText = (d) => `${num(d)} чел. ${daysWord(d)}`;
const manualLabel = (b) => `≈ ${hoursText(b.total_hours)} (≈ ${daysText(b.total_days)})`;

function renderEffort(p) {
  const box = $("effort-box");
  if (!p) { box.innerHTML = '<div class="empty">Разбор появится после создания сценария</div>'; return; }
  const b = p.breakdown;
  if (!b || !b.subtasks || !b.subtasks.length) {
    box.innerHTML = `<div class="empty">${esc(p.manualNote || "Разбор не запрашивался.")}</div>`;
    return;
  }
  box.innerHTML = b.subtasks.map((t) => `
    <div class="effort-row">
      <b>${esc(t.name)}</b>
      <span class="h">${num(t.hours)} ч</span>
    </div>
    ${t.note ? `<div class="effort-note">${esc(t.note)}</div>` : ""}`).join("")
    + `<div class="effort-total"><span>Итого вручную</span>
         <span>≈ ${hoursText(b.total_hours)} (≈ ${daysText(b.total_days)})</span>
       </div>`
    + `<div class="effort-note">Сумма по ${b.subtasks.length} подзадачам.`
    + ` Дни — это часы, делённые на восьмичасовой рабочий день.`
    + ` Оценил ${esc(b.model || "")}.</div>`;
}


/* ─────────────────────────── Источники данных ─────────────────────────── */
function renderSources(p) {
  const list = p.sources || [];
  $("data-block").classList.toggle("hidden", !list.length);
  // У каждого источника должно быть видно происхождение: ссылка на открытые данные
  // либо явная пометка, что данные сгенерированы для демонстрации.
  const origin = (s) => safeUrl(s.url)
    ? `<a href="${esc(safeUrl(s.url))}" target="_blank" rel="noopener noreferrer">открытые данные ↗</a>`
    : `<span class="data-origin">${esc(s.origin || "сгенерировано")}</span>`;
  $("data-list").innerHTML = list.map((s) => `<div class="data-item">
      <svg viewBox="0 0 24 24" class="ic"><path d="M12 3c4.4 0 8 1.3 8 3s-3.6 3-8 3-8-1.3-8-3 3.6-3 8-3z"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6"/></svg>
      <span><b>${esc(s.title)}</b>${s.note ? esc(s.note) : ""}<br>${origin(s)}</span>
    </div>`).join("");
}

/* ─────────────────────────── Загрузка сценария ─────────────────────────── */
function loadPreset(p) {
  pause();
  S.preset = p;
  S.agents = indexAgents(p);
  S.events = p.trace;
  const traceMs = p.trace.reduce((s, e) => s + e.ms, 0);
  S.factor = traceMs ? parseTarget(p.auto) * 1000 / traceMs : 1;

  $("query").value = p.query;
  S.query = p.query;
  S.answer = p.answer ? { text: p.answer, meta: p.answerMeta || "" } : null;
  S.baseline = p.baseline || null;
  S.judge = p.judge || null;
  renderSources(p);
  renderAnswer();
  $("scenario-title").textContent = p.title;
  $("scenario-sub").textContent = "";   // описание сценария не показываем; строка нужна под ошибки
  const origin = p.id === "__live__" ? " · живой запуск"
    : p.real ? " · журнал реального запуска"
    : p.trace.length ? " · демонстрационный журнал" : "";
  $("kind-badge").textContent = (p.kind === "mas"
    ? "MASConfig · маршрутизация (ADK AutoFlow)"
    : "MAWConfig · пайплайн агентов") + origin;
  $("stat-manual").textContent = p.manual;
  const manualRow = $("stat-manual").closest(".stat-line");
  if (manualRow) {
    // Подсказку у сценариев с разбором собираем заново, а не берём сохранённую:
    // в старых записях осталась неверная фраза про «без допущения о длине дня».
    manualRow.title = p.breakdown && p.breakdown.subtasks
      ? `Сумма по ${p.breakdown.subtasks.length} подзадачам: ${hoursText(p.breakdown.total_hours)}.`
        + ` Дни — часы, делённые на восьмичасовой рабочий день.`
        + (p.breakdown.model ? ` Оценил ${p.breakdown.model}.` : "")
      : p.manualNote
        || "Экспертная оценка: сколько заняла бы разработка такой же системы вручную — "
           + "постановка, подбор инструментов, написание и отладка агентов. Не замерялась.";
  }
  $("stat-auto").innerHTML = `<b class="hl">${esc(p.auto)}</b>`;
  const genRow = $("stat-gen-row");           // строки может не быть в старой разметке
  if (genRow) {
    genRow.classList.toggle("hidden", !p.gen);
    $("stat-gen").textContent = p.gen || "—";
  }

  const nAgents = S.agents.size;
  const nSteps = p.kind === "mas" ? 2 : pipelineDepth(p.config.pipeline);
  $("m-agents").textContent = nAgents;
  $("m-steps").textContent = nSteps;
  $("m-tools").textContent = allTools(p).size;

  document.querySelectorAll(".preset").forEach((b) => b.classList.toggle("active", b.dataset.id === p.id));
  if (S.custom && !scenarioList().some((x) => x.id === p.id)) $("preset-count").textContent = scenarioList().length;
  $("btn-run").disabled = !S.backend;
  $("btn-run").title = S.backend
    ? "Исполнить систему по-настоящему"
    : "Бэкенд недоступен: запустите gui/run.py";
  $("btn-generate").disabled = !S.backend;
  renderGraph();
  renderInspector();
  renderEffort(p);
  resetRun();
  showRecordedRun(p);
}


// Показывает сохранённый журнал целиком сразу при выборе сценария: раньше вкладка
// «Ход выполнения» была пуста до нажатия «Запустить», хотя запись уже есть.
/* ─────────────── Перенос сценария между машинами ───────────────
 * В файл уходит весь объект сценария: схема системы, журнал прогона, ответ,
 * ответ одной модели, вердикт судьи, разбор трудоёмкости и источники. Этого
 * достаточно, чтобы на другой машине открыть его без запуска и без ключа.
 */
const SCENARIO_FORMAT = "fedotmas-scenario-1";

function exportScenario() {
  if (!S.preset) return;
  const payload = { format: SCENARIO_FORMAT, exported: new Date().toISOString(), preset: S.preset };
  const name = (S.preset.title || "сценарий").replace(/[^\wа-яёА-ЯЁ -]+/g, "").trim() || "сценарий";
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)],
                                           { type: "application/json" }));
  const a = document.createElement("a");
  a.href = url; a.download = `${name}.json`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function importScenario(text) {
  let data;
  try {
    data = JSON.parse(text);
  } catch (e) {
    alert("Это не похоже на файл сценария: " + e.message);
    return;
  }
  // Принимаем и обёртку с форматом, и голый объект сценария — так файл переживает
  // ручную правку, а старые выгрузки не становятся мусором.
  const preset = data && data.preset ? data.preset : data;
  if (!preset || typeof preset !== "object" || !preset.config) {
    alert("В файле нет конфигурации системы — импортировать нечего.");
    return;
  }
  // Файл пришёл извне: всё, на что интерфейс опирается при отрисовке, приводим к
  // ожидаемому виду. Иначе отсутствующее поле роняет renderPresetList или loadPreset,
  // а сценарий к тому моменту уже записан в localStorage — интерфейс не открывается
  // вовсе, пока руками не вычистить данные сайта.
  preset.title = String(preset.title || "Импортированный сценарий");
  preset.domain = String(preset.domain || "сценарий");
  preset.kind = preset.kind === "mas" ? "mas" : "maw";
  preset.trace = Array.isArray(preset.trace) ? preset.trace : [];
  preset.sources = Array.isArray(preset.sources) ? preset.sources : [];
  preset.genSteps = Array.isArray(preset.genSteps) ? preset.genSteps : [];
  if (preset.breakdown && !Array.isArray(preset.breakdown.subtasks)) delete preset.breakdown;
  // Конфигурация — единственное, без чего сценарий бессмыслен, и именно на ней
  // отрисовка падала: пустой pipeline, agents не массивом, workers строкой. Проверяем
  // до записи в хранилище, иначе битый файл оставался в списке и ронял интерфейс
  // при каждой загрузке страницы.
  const cfg = preset.config;
  const agentsOk = preset.kind === "mas"
    ? cfg.coordinator && typeof cfg.coordinator === "object" && Array.isArray(cfg.workers)
    : Array.isArray(cfg.agents) && cfg.agents.length && cfg.pipeline
      && typeof cfg.pipeline === "object";
  if (!agentsOk) {
    alert("Конфигурация в файле неполная: не хватает состава агентов или схемы запуска.");
    return;
  }
  // Идентификатор снимаем: addScenario выдаст новый. Свой оставлять нельзя —
  // при импорте того же файла дважды в списке оказались бы два сценария с одним id,
  // и правка одного меняла бы другой.
  delete preset.id;
  try {
    addScenario(preset);
    showTab("answer");
  } catch (e) {
    // Показать сценарий не вышло — убираем его из списка, иначе он останется
    // висеть и ронять отрисовку при каждом открытии.
    S.custom = S.custom.filter((p) => p !== preset);
    storeScenarios();
    renderPresetList();
    alert("Файл не удалось открыть как сценарий: " + e.message);
  }
}

function showRecordedRun(preset) {
  if (!preset.trace || !preset.trace.length) return;
  const feed = $("feed");
  feed.innerHTML = "";
  preset.trace.forEach((e) => pushMessage(e));
  feed.scrollTop = 0;                       // журнал показываем с начала, а не с конца

  S.idx = preset.trace.length;              // «Запустить» проиграет запись заново с нуля
  S.tokens = preset.trace.reduce((sum, e) => sum + (e.tokens || 0), 0);
  S.seconds = preset.trace.reduce((sum, e) => sum + e.ms, 0) * S.factor / 1000;
  $("m-tokens").textContent = nfmt(S.tokens);
  $("m-time").textContent = fmtTime(S.seconds);
  $("p-fill").style.width = "100%";
  $("graph").querySelectorAll(".node").forEach((n) => {
    n.classList.remove("active");
    n.classList.add("done");
  });
}


/* ─────────────────────────── Живой режим ───────────────────────────
 * Если рядом поднят run.py, интерфейс умеет не только проигрывать
 * сохранённые журналы, но и по-настоящему запускать FEDOT.MAS: мета-агент
 * проектирует систему, она исполняется, события приходят потоком (SSE).
 */

async function probeBackend() {
  try {
    const r = await fetch("api/status", { cache: "no-store" });
    if (!r.ok) return;
    S.backend = await r.json();
    resetOnServerRestart(S.backend.run_id);
    S.models = {
      gen: S.backend.model,
      run: S.backend.model,
      single: S.backend.model,
      judge: S.backend.judge_model || S.backend.model,
    };
    fillModelSelect("model-gen", "gen", S.backend.models, S.models.gen);
    fillModelSelect("model-run", "run", S.backend.models, S.models.run);
    fillModelSelect("model-single", "single", S.backend.models, S.models.single);
    fillModelSelect("model-judge", "judge", judgeChoices(), S.models.judge);
    renderKeyChip();
    restoreKey();
  } catch {
    /* бэкенд не отвечает — кнопки останутся заблокированными */
  }
  renderMode();
}

// У судьи набор шире: его модель задаётся отдельно и в общий список может не входить
function judgeChoices() {
  const list = (S.backend?.models || []).slice();
  const judge = S.backend?.judge_model;
  if (judge && !list.some((m) => m.id === judge)) list.unshift({ id: judge, label: judge });
  // В узкой колонке «google/gemini-2.5-pro» не помещается — вендора убираем, id остаётся значением
  return list.map((m) => ({ id: m.id, label: (m.label || m.id).split("/").pop() }));
}

function fillModelSelect(id, key, models, selected) {
  const sel = $(id);
  if (!sel) return;
  sel.innerHTML = (models || []).map((m) =>
    `<option value="${esc(m.id)}"${m.id === selected ? " selected" : ""}>${esc(m.label || m.id)}</option>`).join("");
  sel.addEventListener("change", () => { S.models[key] = sel.value; });
}

/* ──────────────────────── Ключ провайдера ────────────────────────
 * В публичном режиме сервер запускается без ключа: пока пользователь не введёт
 * свой, все обращения к моделям отвечают 428, и интерфейс открывает эту форму.
 */
function keyModal(show) {
  $("key-modal").classList.toggle("hidden", !show);
  // Код доступа обычно приезжает в ссылке, но страница-предупреждение туннеля
  // может её обрезать — тогда даём ввести код руками, иначе вход в тупике.
  const needToken = !!S.backend?.public && !accessToken();
  $("key-token-row").classList.toggle("hidden", !needToken);
  if (show) setTimeout(() => (needToken ? $("key-token") : $("key-input"))?.focus(), 30);
}

function renderKeyChip() {
  const chip = $("btn-key");
  if (!chip) return;
  const pub = !!S.backend?.public;
  chip.classList.toggle("hidden", !pub);
  const ok = !!S.backend?.user_key;
  chip.classList.toggle("key-ok", ok);
  $("btn-key-label").textContent = ok ? "Ключ принят" : "Ввести ключ";
  chip.title = ok ? "Ключ провайдера принят. Нажмите, чтобы сменить" : "Ввести ключ провайдера";
}

async function sendKey(key, remember) {
  const note = $("key-note");
  const btn = $("key-submit");
  note.textContent = "Проверяем ключ…";
  note.style.color = "var(--text-dim)";
  btn.disabled = true;
  try {
    const r = await fetch("api/key", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key }),
    });
    if (r.status === 401) {
      $("key-token-row").classList.remove("hidden");
      note.textContent = "Код доступа не подошёл: скопируйте ссылку целиком, вместе с частью после ?t=";
      note.style.color = "var(--warn)";
      return false;
    }
    const data = await readJson(r, "проверка ключа");
    if (!data.ok) {
      note.textContent = "Ключ не принят: " + (data.error || "провайдер отклонил запрос");
      note.style.color = "var(--warn)";
      return false;
    }
    try {
      if (remember) localStorage.setItem(LS_KEY, key);
      else localStorage.removeItem(LS_KEY);
    } catch {}
    if (S.backend) S.backend.user_key = true;
    note.textContent = "";
    renderKeyChip();
    keyModal(false);
    return true;
  } catch (e) {
    note.textContent = "Не удалось связаться с сервером: " + e.message;
    note.style.color = "var(--warn)";
    return false;
  } finally {
    btn.disabled = false;
  }
}

/* Ключ мог остаться в браузере с прошлого раза, а сервер тем временем
 * перезапустили — тогда отдаём его молча, без формы. */
async function restoreKey() {
  if (!S.backend?.public || S.backend.user_key) return;
  const saved = savedKey();
  if (accessToken() && saved && await sendKey(saved, true)) return;
  keyModal(true);
}

function initKeyForm() {
  $("key-submit").addEventListener("click", () => {
    const tok = $("key-token").value.trim();
    if (tok) setAccessToken(tok);
    const key = $("key-input").value.trim();
    if (!key) {
      $("key-note").textContent = "Введите ключ.";
      $("key-note").style.color = "var(--warn)";
      return;
    }
    sendKey(key, $("key-remember").checked);
  });
  $("key-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") $("key-submit").click();
  });
  $("key-close").addEventListener("click", () => keyModal(false));
  $("btn-key")?.addEventListener("click", () => {
    $("key-input").value = "";
    $("key-note").textContent = "";
    $("key-close").classList.toggle("hidden", !S.backend?.user_key);
    keyModal(true);
  });
}

/* Любой запрос может ответить 428 «нет ключа» — например, если сервер
 * перезапустили посреди показа. Тогда форма открывается сама. */
function noteKeyNeeded(status) {
  if (status !== 428) return false;
  if (S.backend) S.backend.user_key = false;
  renderKeyChip();
  keyModal(true);
  return true;
}

function renderMode() {
  const online = !!S.backend;
  // Отдельной плашки режима нет — режим всегда живой; о недоступном бэкенде
  // говорят заблокированные кнопки и подсказка на них.
  $("btn-generate").disabled = !online;
  $("btn-generate").title = online
    ? "Описать задачу и собрать под неё систему"
    : "Бэкенд недоступен: запустите gui/run.py и обновите страницу";
  // Запускать нечего, пока сценарий не создан
  $("btn-run").disabled = !S.preset || (!online && !S.preset.trace.length);
}

function liveMessage(kind, agent, text, tool) {
  pushMessage({ agent, phase: kind, text, tool, tokens: 0, ms: 0, final: kind === "готово" });
}

function liveActivate(name) {
  const node = nodeByName(name);
  if (node) { node.classList.remove("done"); node.classList.add("active"); }
  edgesTo(name).forEach((e) => { e.classList.add("live"); flyToken(e); });
}

function liveFinish(name) {
  const node = nodeByName(name);
  if (node) { node.classList.remove("active"); node.classList.add("done"); }
}

/** Записывает результат живого прогона в сценарий, чтобы его можно было проиграть позже. */
function persistPreset() {
  if (!S.preset || !S.preset.custom) return;
  const i = S.custom.findIndex((p) => p.id === S.preset.id);
  if (i >= 0) S.custom[i] = S.preset;
  storeScenarios();
  renderPresetList();
}

function stopLive() {
  if (S.abort) { S.abort.abort(); S.abort = null; }
  clearInterval(S.liveTimer);
  S.liveTimer = null;
  setPlayIcon(false);
}


/** Реальный запуск текущей конфигурации с потоковыми событиями. */
async function liveRun() {
  if (!S.preset) return;
  stopLive();
  showTab("feed");
  $("feed").innerHTML = "";
  $("graph").querySelectorAll(".node").forEach((n) => n.classList.remove("active", "done"));
  S.tokens = 0;
  const t0 = performance.now();
  const total = $("graph").querySelectorAll(".node").length || 1;
  let finished = 0;
  const trace = [], startedAt = {}, agentIO = {};        // журнал живого прогона сохраняем в сценарий
  S.liveTimer = setInterval(() => {
    $("m-time").textContent = fmtTime((performance.now() - t0) / 1000);
  }, 200);
  setPlayIcon(true);
  liveMessage("запуск", "runner", `Система запущена на реальных моделях (${S.models.run}). Первые ответы агентов появятся здесь.`);

  S.abort = new AbortController();
  try {
    const r = await fetch("api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config: S.preset.config, kind: S.preset.kind,
                             query: $("query").value, model: S.models.run,
                             tools: S.preset.tools || null,
                             custom_mcp: S.preset.customMcp || null }),
      signal: S.abort.signal,
    });
    // Единственный поток, где код ответа не проверялся: при 401 (нет токена доступа)
    // или 403 (запрос сочтён межсайтовым) тело — обычный JSON без строк «data:»,
    // цикл молча заканчивался, и человек видел «Система запущена…» и тишину.
    if (!r.ok) {
      let reason = `сервер ответил ${r.status}`;
      try {
        const body = await r.json();
        if (body && body.error) reason = body.error;
      } catch { /* тело не JSON — оставляем код ответа */ }
      throw new Error(reason);
    }
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const chunks = buf.split("\n\n");
      buf = chunks.pop();
      for (const chunk of chunks) {
        // Строки, начинающиеся с двоеточия, — служебный пульс SSE, данных в них нет
        const raw = chunk.split("\n").find((x) => x.startsWith("data: "));
        const line = raw ? raw.slice(6).trim() : "";
        if (!line) continue;
        const ev = JSON.parse(line);
        if (ev.type === "agent_start") {
          startedAt[ev.agent] = performance.now(); liveActivate(ev.agent);
          // Копим вход агента, чтобы приложить его к сообщению о результате
          agentIO[ev.agent] = {
            instruction: ev.instruction || "",
            incoming: Object.entries(ev.incoming || {})
              .map(([k, v]) => `${k}:\n${display(v)}`).join("\n\n"),
          };
        }
        else if (ev.type === "agent_done") {
          liveFinish(ev.agent);
          finished++;
          $("p-fill").style.width = Math.min(100, (100 * finished) / total).toFixed(0) + "%";
          const io = Object.assign(agentIO[ev.agent] || {},
            { output: ev.output || "", outputKey: ev.output_key || "" });
          agentIO[ev.agent] = io;
          if (io.output) {
            const head = io.output.trim().split("\n")[0].slice(0, 160);
            liveMessage("результат", ev.agent, head || "(агент завершил работу)");
            const last = trace[trace.length - 1];
            const entry = { agent: ev.agent, phase: "результат", text: head, tokens: 0,
                            ms: ev.ms || 1200, io };
            trace.push(entry);
            const node = document.querySelector(".msg:last-child");
            if (node && !node.querySelector(".msg-io")) {
              const d = document.createElement("details");
              d.className = "msg-io";
              d.innerHTML = `<summary>вход и выход</summary>
                <div class="io-label">инструкция агента</div><pre>${esc(io.instruction || "—")}</pre>
                <div class="io-label">на входе из состояния</div><pre>${esc(io.incoming || "—")}</pre>
                <div class="io-label">результат агента${io.outputKey ? " → " + esc(io.outputKey) : ""}</div><pre>${esc(io.output || "—")}</pre>`;
              node.appendChild(d);
            }
          }
        } else if (ev.type === "tool") {
          // маршрутизацию координатора показываем словами, остальные вызовы — как есть
          const routing = ev.tool === "transfer_to_agent";
          const text = routing ? `Передаёт задачу агенту ${ev.target || "…"}`
                               : (ev.args ? `${ev.tool}(${display(ev.args)})` : `Вызов инструмента ${ev.tool}`);
          liveMessage(routing ? "маршрутизация" : "инструмент", ev.agent, text, routing ? null : ev.tool);
          trace.push({ agent: ev.agent, phase: routing ? "маршрутизация" : "инструмент", text,
                       tool: ev.tool, tokens: 0, ms: routing ? 1200 : 1400 });
        }
        else if (ev.type === "tool_result" && ev.error) {
          liveMessage("ошибка инструмента", ev.agent, display(ev.text), ev.tool);
          trace.push({ agent: ev.agent, phase: "ошибка инструмента", text: ev.text, tool: ev.tool, tokens: 0, ms: 1200 });
        }
        else if (ev.type === "text") {
          S.tokens += ev.tokens || 0;
          $("m-tokens").textContent = nfmt(S.tokens);
          liveMessage("шаг агента", ev.agent, ev.text);
          const started = startedAt[ev.agent] || performance.now();
          trace.push({ agent: ev.agent, phase: "шаг агента", text: ev.text, tokens: ev.tokens || 0,
                       ms: Math.min(9000, Math.max(1200, Math.round(performance.now() - started))) });
        } else if (ev.type === "tokens") {
          S.tokens += ev.tokens || 0;
          $("m-tokens").textContent = nfmt(S.tokens);
        } else if (ev.type === "done") {
          $("m-tokens").textContent = nfmt(ev.tokens || S.tokens);
          $("m-time").textContent = fmtTime(ev.elapsed);
          $("p-fill").style.width = "100%";
          // Итог — артефакт последнего содержательного агента; отзыв критика ответом не является.
          const criticRe = /критик|critic|валид|valid|провер|review|judge|качеств|контрол|аудит|реценз|quality/i;
          const cfgAgents = S.preset.kind === "mas"
            ? [S.preset.config.coordinator, ...(S.preset.config.workers || [])]
            : (S.preset.config.agents || []);
          const criticKeys = new Set(cfgAgents.filter((a) => criticRe.test(a.name))
                                              .map((a) => a.output_key).filter(Boolean));
          const keys = Object.keys(ev.state || {}).filter((k) => display(ev.state[k] ?? "").trim());
          const useful = keys.filter((k) => !criticKeys.has(k));
          const pool = useful.length ? useful : keys;
          // Последний по порядку агент иногда отдаёт короткую сводку вместо расчёта —
          // тогда берём самый содержательный артефакт, а не формально последний.
          let last = pool.length ? display(ev.state[pool[pool.length - 1]] ?? "") : "";
          const longest = pool.reduce((best, k) => {
            const v = display(ev.state[k] ?? "");
            return v.length > best.length ? v : best;
          }, "");
          if (longest.length > last.length * 2) last = longest;
          S.answer = { text: last || "(система не вернула текстового результата)",
                       meta: `${S.preset.kind === "mas" ? "MASConfig" : "MAWConfig"} · ${nfmt(ev.tokens || 0)} токенов · ${String(ev.elapsed).replace(".", ",")} с` };
          S.query = $("query").value;
          S.baseline = null; S.judge = null;
          renderAnswer();
          const prev = document.querySelector(".msg:last-child .msg-text");
          const duplicate = prev && last && prev.textContent.trim().startsWith(last.trim().slice(0, 60));
          if (!duplicate) liveMessage("готово", "результат", last || "Система завершила работу.");
          else prev.closest(".msg").classList.add("final");

          // сценарий получает журнал, ответ и стоимость — дальше его можно проигрывать без сети
          if (trace.length) {
            trace[trace.length - 1].final = true;
            S.preset.trace = trace;
            S.preset.answer = S.answer.text;
            S.preset.answerMeta = S.answer.meta;
            S.preset.auto = `${fmtTime(ev.elapsed)} · ${(( ev.tokens || S.tokens) / 1000).toFixed(1).replace(".", ",")}к токенов`;
            $("stat-auto").innerHTML = `<b class="hl">${esc(S.preset.auto)}</b>`;
            persistPreset();
          }
          $("stat-auto").innerHTML = `<b class="hl">${esc(String(ev.elapsed).replace(".", ","))} с · ${esc(nfmt(ev.tokens))} токенов</b>`;
        } else if (ev.type === "error") {
          liveMessage("ошибка", "runner", ev.error);
        }
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") liveMessage("ошибка", "gui", String(err.message || err));
  } finally {
    stopLive();
    $("graph").querySelectorAll(".node.active").forEach((n) => { n.classList.remove("active"); n.classList.add("done"); });
  }
}

/* ─────────────────────────── Тема и кегль ───────────────────────────
 * Светлая тема — по умолчанию: демонстрация идёт на большом экране.
 */
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try { localStorage.setItem("fedotmas-theme", theme); } catch {}
  if (S.preset) { renderGraph(); renderInspector(); }
}

function initAppearance() {
  let theme = "light";
  try { theme = localStorage.getItem("fedotmas-theme") || "light"; } catch {}
  applyTheme(theme);
  $("btn-theme").addEventListener("click", () =>
    applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark"));

  let zoomed = false;
  try { zoomed = localStorage.getItem("fedotmas-zoom") === "1"; } catch {}
  const applyZoom = () => {
    document.body.classList.toggle("zoomed", zoomed);
    $("btn-zoom").classList.toggle("chip-demo", zoomed);
    try { localStorage.setItem("fedotmas-zoom", zoomed ? "1" : "0"); } catch {}
  };
  applyZoom();
  $("btn-zoom").addEventListener("click", () => { zoomed = !zoomed; applyZoom(); });
}

/* ─────────────────────────── Свой сценарий ───────────────────────────
 * Пользователь описывает задачу и данные, выбирает тип системы (MAS или MAW),
 * мета-агент проектирует под это конфигурацию. Нужен живой режим.
 */
function parseSources(text) {
  return text.split("\n").map((line) => line.trim()).filter(Boolean).map((line) => {
    const [title, url] = line.split("|").map((x) => (x || "").trim());
    return url ? { title, url } : { title, origin: "введено вручную" };
  });
}

/* ─────────────── Файлы как источники ───────────────
 * Ссылку агент скачивает сам, а лежащий на компьютере файл иначе до него не дойдёт.
 * Загружаем файл на сервер, а путь к нему дописываем в запрос — дальше его читает
 * тот же инструмент «document», что и скачанные файлы.
 */
function renderAttached() {
  const box = $("new-file-list");
  if (!box) return;
  box.innerHTML = S.files.map((f, i) => `
    <span class="mcp-chip">${esc(f.name)}
      <button type="button" data-i="${i}" title="Убрать">✕</button></span>`).join("");
  box.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    S.files.splice(Number(b.dataset.i), 1); renderAttached();
  }));
}

async function attachFiles(fileList) {
  const note = $("new-file-note");
  for (const file of fileList) {
    note.textContent = `Загружаем ${file.name}…`;
    const form = new FormData();
    form.append("file", file);
    try {
      const r = await fetch("api/upload", { method: "POST", body: form });
      const d = await readJson(r, "загрузка файла");
      if (!d.ok) { note.textContent = `${file.name}: ${d.error}`; continue; }
      S.files.push({ name: d.name, path: d.path });
      // Если файл пришлось привести к другому виду, человек должен об этом знать:
      // молча изменённые данные хуже, чем изменённые с объяснением.
      note.textContent = d.note ? `${d.name}: ${d.note}` : "";
      renderAttached();
    } catch (e) {
      note.textContent = `${file.name}: ${e.message}`;
    }
  }
}

function initFileSources() {
  const pick = $("new-file-pick"), input = $("new-file");
  if (!pick || !input) return;
  pick.addEventListener("click", () => input.click());
  input.addEventListener("change", (e) => {
    attachFiles([...(e.target.files || [])]);
    e.target.value = "";
  });
}

// Список инструментов: по умолчанию отмечены все доступные — так система работает сейчас
function renderToolPicks() {
  const box = $("new-tools");
  if (!box) return;
  const tools = (S.backend && S.backend.tools) || [];
  box.innerHTML = tools.map((t) => `
    <label class="tool-pick" title="${esc(t.note || "")}">
      <input type="checkbox" value="${esc(t.id)}" checked>
      <span><b>${esc(t.id)}</b></span>
    </label>`).join("") || '<span class="label-hint">бэкенд не сообщил список инструментов</span>';
  box.addEventListener("change", updateToolCount);
  updateToolCount();
}

function pickedTools() {
  return [...document.querySelectorAll("#new-tools input:checked")].map((i) => i.value);
}

function updateToolCount() {
  const all = document.querySelectorAll("#new-tools input").length;
  $("new-tools-count").textContent = all ? `выбрано ${pickedTools().length} из ${all}` : "";
}

function openNewScenario(prefillTask) {
  const web = !!(S.backend && S.backend.web_search);
  renderToolPicks();
  renderMcpAdded();
  S.files = [];
  renderAttached();
  $("new-file-note").textContent = "";
  $("new-web").disabled = !web;
  $("new-web").checked = web;
  $("new-web-hint").textContent = web
    ? "агенту будет выдан веб-поиск"
    : "SearXNG не поднят — агенты будут работать только с данными из текста";
  $("new-note").textContent = S.backend
    ? ""
    : "Генерация доступна только в живом режиме: запустите gui/run.py и обновите страницу.";
  $("new-submit").disabled = !S.backend;
  if (typeof prefillTask === "string" && prefillTask.trim() && !$("new-text").value.trim())
    $("new-text").value = prefillTask;
  $("new-modal").classList.remove("hidden");
  $("new-title").focus();
}

function closeNewScenario() {
  $("new-modal").classList.add("hidden");
}

// Читает SSE-поток и отдаёт события по одному: генерация занимает десятки секунд,
// без обратной связи окно выглядит зависшим.
async function* sseEvents(url, payload) {
  const r = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok || !r.body) throw new Error(`сервер ответил ${r.status}`);
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop();
    for (const chunk of parts) {
      const line = chunk.split("\n").find((x) => x.startsWith("data: "));
      if (line) yield JSON.parse(line.slice(6));
    }
  }
}

// Понятные подписи вместо служебных имён агентов мета-агента
const GEN_STAGE_LABELS = {
  pool_generator: "Подбираем состав агентов",
  pipeline_generator: "Собираем пайплайн",
  agent_pool_generator: "Подбираем состав агентов",
};

function genProgress() {
  const box = $("new-progress");
  const fill = $("new-progress-fill");
  const log = $("new-progress-log");
  const t0 = Date.now();
  let pct = 0;
  const timer = setInterval(() => {
    $("new-progress-time").textContent = Math.round((Date.now() - t0) / 1000) + " с";
  }, 250);
  box.classList.remove("hidden");
  log.innerHTML = "";
  return {
    step(text, percent) {
      $("new-progress-step").textContent = text;
      if (percent != null) pct = percent;
      fill.style.width = pct + "%";
    },
    note(text) {
      const line = document.createElement("div");
      line.textContent = text;
      log.appendChild(line);
      log.scrollTop = log.scrollHeight;
    },
    bump(delta) {                       // события идут неравномерно — двигаем шкалу понемногу
      pct = Math.min(92, pct + delta);
      fill.style.width = pct + "%";
    },
    stop() { clearInterval(timer); box.classList.add("hidden"); },
  };
}

async function submitNewScenario() {
  const title = $("new-title").value.trim() || "Свой сценарий";
  const text = $("new-text").value.trim();
  const kind = document.querySelector('input[name="new-kind"]:checked').value;
  const web = $("new-web").checked && !$("new-web").disabled;
  const wantEffort = $("new-effort").checked;
  const tools = pickedTools();
  const customMcp = S.mcpCustom.slice();
  const allTools = tools;
  if (!text) { $("new-note").textContent = "Опишите задачу — что нужно сделать."; return; }
  if (!S.backend) { $("new-note").textContent = "Нужен живой режим."; return; }

  const btn = $("new-submit");
  btn.disabled = true;
  $("new-note").textContent = "";
  const ui = genProgress();
  try {
    // 1. Свободный текст делится на постановку для мета-агента и запрос для запуска
    btn.textContent = "Генерируем…";
    ui.step("Разбираем задачу на постановку и запрос", 8);
    const pr = await fetch("api/prepare", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, kind, model: S.models.gen, web, tools }),
    });
    const split = await readJson(pr, "разбор постановки");
    if (!split.ok) throw new Error(split.error || "не удалось разобрать текст");
    // Путь к прикреплённому файлу должен оказаться в запросе: иначе агент о нём
    // не узнает. Читать его будет инструмент document — тот же, что и скачанные файлы.
    if (S.files.length) {
      split.query += "\n\nФайлы с исходными данными (прочитай их инструментом document "
        + "по указанному пути, не выдумывай содержимое):\n"
        + S.files.map((f) => `- ${f.name}: ${f.path}`).join("\n");
    }
    ui.note("Постановка и запрос выделены");

    // 2. Мета-агент проектирует систему; события приходят потоком
    ui.step("Мета-агент проектирует систему", 22);
    let d = null;
    for await (const ev of sseEvents("api/generate_stream",
        { task: split.task, query: split.query, kind, model: S.models.gen, web,
          tools: allTools, custom_mcp: customMcp })) {
      if (ev.type === "done") { d = ev; break; }
      if (ev.type === "agent_start" && GEN_STAGE_LABELS[ev.agent]) {
        ui.step(GEN_STAGE_LABELS[ev.agent], ev.agent.startsWith("pipeline") ? 55 : 30);
      } else if (ev.type === "agent_done" && GEN_STAGE_LABELS[ev.agent]) {
        ui.note(`${GEN_STAGE_LABELS[ev.agent]} — готово за ${(ev.ms / 1000).toFixed(0)} с`);
      } else if (ev.type === "tokens" || ev.type === "text") {
        ui.bump(2);
      }
    }
    if (!d) throw new Error("поток прервался");
    if (!d.ok) throw new Error(d.error || "мета-агент вернул ошибку");
    const agents = (d.config.agents || [d.config.coordinator, ...(d.config.workers || [])]).filter(Boolean);
    ui.note(`Собрано агентов: ${agents.length}`);

    // 3. Трудоёмкость: сразу раскладываем задачу на подзадачи и складываем часы.
    // Общая оценка «примерно столько-то дней» ничему не учит — итог должен быть
    // суммой того, что видно построчно, иначе цифре нечем себя объяснить.
    let manual = "—";
    let manualNote = "Оценка трудоёмкости не запрашивалась при создании сценария.";
    let breakdown = null;
    if (wantEffort) {
      ui.step("Раскладываем задачу на подзадачи", 88);
      try {
        const er = await fetch("api/effort_breakdown", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ task: split.task, config: d.config, kind, model: S.models.gen }),
        });
        const eff = await readJson(er, "разбор трудоёмкости");
        if (eff.ok) {
          breakdown = eff;
          manual = manualLabel(eff);
          manualNote = `Сумма по ${eff.subtasks.length} подзадачам: ${hoursText(eff.total_hours)}.`
            + ` Дни — часы, делённые на восьмичасовой рабочий день.`
            + ` Оценил ${eff.model} при создании сценария.`;
          ui.note(`Трудоёмкость вручную: ${manual} по ${eff.subtasks.length} подзадачам`);
        } else {
          manualNote = "Разложить задачу не удалось: " + (eff.error || "неизвестная ошибка");
        }
      } catch (err) {
        manualNote = "Разложить задачу не удалось: " + err.message;
      }
    }
    ui.step("Готово", 100);

    const gen = `${String(d.gen.seconds).replace(".", ",")} с · ${(d.gen.tokens / 1000).toFixed(1).replace(".", ",")}к токенов`;
    const preset = {
      id: `custom_${Date.now()}`, title, domain: "сценарий", kind: d.kind, model: d.model,
      query: split.query,
      brief: split.task,
      summary: "",
      tools,
      manual, manualNote, breakdown,
      gen, auto: "—", genSteps: [], config: d.config, trace: [],
      tools: allTools, customMcp,
      sources: parseSources($("new-sources").value)
        .concat(S.files.map((f) => ({ title: f.name, origin: "файл прикреплён к сценарию" }))),
    };
    closeNewScenario();
    addScenario(preset);
    showTab("answer");
  } catch (err) {
    $("new-note").textContent = "Не удалось: " + err.message;
  } finally {
    ui.stop();
    btn.disabled = false; btn.textContent = "Сгенерировать систему";
  }
}

function initNewScenario() {
  $("btn-new").addEventListener("click", () => openNewScenario());   // без аргумента: иначе в поле попадёт объект события
  $("new-close").addEventListener("click", closeNewScenario);
  $("new-cancel").addEventListener("click", closeNewScenario);
  $("new-submit").addEventListener("click", submitNewScenario);
  $("new-modal").addEventListener("click", (e) => { if (e.target === $("new-modal")) closeNewScenario(); });
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("new-modal").classList.contains("hidden")) closeNewScenario();
  });
}

/* ─────────────────────────── Инициализация ─────────────────────────── */
/* Список сценариев: встроенные плюс добавленные пользователем.
 * Свои сценарии и скрытые встроенные переживают перезагрузку страницы. */
const LS_CUSTOM = "fedotmas-custom", LS_HIDDEN = "fedotmas-hidden";

function loadStored(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key) || "") ?? fallback; } catch { return fallback; }
}

function storeScenarios() {
  try {
    localStorage.setItem(LS_CUSTOM, JSON.stringify(S.custom));
    localStorage.setItem(LS_HIDDEN, JSON.stringify(S.hidden));
  } catch { /* приватный режим браузера — просто не сохраняем */ }
}

function scenarioList() {
  // Показываем только то, что создано в этой сессии: встроенные записи в список не идут.
  return S.custom.filter((p) => !S.hidden.includes(p.id));
}

/** Стартовый вид без сценариев: показываем, с чего начать, вместо пустого графа. */
function showEmptyState() {
  S.preset = null; S.events = []; S.answer = null; S.baseline = null; S.judge = null;
  $("scenario-title").textContent = "Сценариев пока нет";
  $("scenario-sub").textContent = "Нажмите «Добавить сценарий», опишите задачу — система соберётся под неё.";
  $("kind-badge").textContent = "";
  $("graph").innerHTML = "";
  $("feed").innerHTML = '<div class="empty">Журнал появится после запуска системы</div>';
  $("answer").innerHTML = '<div class="empty">Ответ системы появится после запуска</div>';
  ["m-agents", "m-steps", "m-tools", "m-tokens"].forEach((id) => { $(id).textContent = "0"; });
  $("m-time").textContent = "0,0 с";
  $("p-fill").style.width = "0%";
  $("stat-manual").textContent = "—";
  $("stat-auto").textContent = "—";
  $("btn-run").disabled = true;
  $("btn-baseline").disabled = true;
  $("btn-judge").disabled = true;
  const gen = $("stat-gen-row"); if (gen) gen.classList.add("hidden");
  const data = $("data-block"); if (data) data.classList.add("hidden");
  renderEffort(null);
}

function renderPresetList() {
  const host = $("presets");
  const list = scenarioList();
  host.innerHTML = list.map((p) => `
    <button class="preset" data-id="${esc(p.id)}">
      <span class="preset-del" data-del="${esc(p.id)}" title="Убрать сценарий из списка">✕</span>
      <div class="preset-name">${esc(p.title)}</div>
      <div class="preset-meta">
        <span class="tag tag-${p.kind === "mas" ? "mas" : "maw"}">${esc(kindLabel(p.kind))}</span>
        <span>${esc(p.domain)}</span>
        <span>·</span>
        <span>${p.kind === "mas" ? (p.config.workers || []).length + 1 : (p.config.agents || []).length} агентов</span>
      </div>
    </button>`).join("");
  $("preset-count").textContent = list.length;

  host.querySelectorAll(".preset").forEach((b) => b.addEventListener("click", (e) => {
    if (e.target.dataset.del) return;                       // клик по крестику — не выбор
    const p = scenarioList().find((x) => x.id === b.dataset.id);
    if (p) loadPreset(p);
  }));
  host.querySelectorAll(".preset-del").forEach((x) => x.addEventListener("click", (e) => {
    e.stopPropagation();
    removeScenario(x.dataset.del);
  }));
  const active = S.preset && S.preset.id;
  host.querySelectorAll(".preset").forEach((b) => b.classList.toggle("active", b.dataset.id === active));
}

function removeScenario(id) {
  const wasActive = S.preset && S.preset.id === id;
  const before = S.custom.length;
  S.custom = S.custom.filter((p) => p.id !== id);
  if (S.custom.length === before && !S.hidden.includes(id)) S.hidden.push(id);
  storeScenarios();
  renderPresetList();
  const list = scenarioList();
  if (wasActive && list.length) loadPreset(list[0]);
}

function addScenario(preset) {
  preset.custom = true;
  preset.id = preset.id && preset.id.startsWith("custom_") ? preset.id : `custom_${Date.now()}`;
  S.custom.push(preset);
  storeScenarios();
  renderPresetList();
  loadPreset(preset);
}

const LS_RUN = "fedotmas-run";

function initPresets() {
  S.custom = loadStored(LS_CUSTOM, []) || [];
  S.hidden = loadStored(LS_HIDDEN, []) || [];
  // В автономной копии показывать нечего: запускать она не умеет, а список берётся
  // из localStorage, которого у нового читателя нет. Подставляем записанные прогоны,
  // вшитые в саму копию. На живом стенде флага нет и поведение прежнее.
  if (window.OFFLINE_DEMO && !S.custom.length && Array.isArray(window.PRESETS)) {
    S.custom = window.PRESETS.slice();
  }
  renderPresetList();
}

/** Сценарии живут только в пределах запуска сервера: перезапуск начинает показ с чистого листа. */
function resetOnServerRestart(runId) {
  if (!runId) return;
  let stored = null;
  try { stored = localStorage.getItem(LS_RUN); } catch { return; }
  if (stored === runId) return;
  try {
    localStorage.setItem(LS_RUN, runId);
    localStorage.removeItem(LS_CUSTOM);
    localStorage.removeItem(LS_HIDDEN);
  } catch { /* приватный режим — просто очищаем состояние в памяти */ }
  S.custom = []; S.hidden = [];
  renderPresetList();
  showEmptyState();
}


function presetFromConfig(cfg, name) {
  const kind = cfg.coordinator ? "mas" : "maw";
  const first = kind === "mas" ? cfg.coordinator : (cfg.agents || [])[0] || {};
  return {
    id: "__loaded__", title: name, domain: "загружено", kind,
    model: first.model || "по умолчанию",
    query: "", summary: "Конфигурация загружена из файла. Журнал выполнения отсутствует — доступен просмотр структуры.",
    manual: "—", auto: "—", genSteps: [], config: cfg, trace: [],
  };
}


// Разворот правой панели на всю ширину: длинные ответы с таблицами в колонке 372px
// читать неудобно, а на демонстрации ответ — главное, что смотрят.
function toggleInspectorWidth() {
  const layout = document.querySelector(".layout");
  const full = layout.classList.toggle("inspect-full");
  const btn = $("btn-expand");
  btn.title = full ? "Вернуть обычный вид" : "Развернуть панель на всю ширину";
  btn.innerHTML = full
    ? '<svg viewBox="0 0 24 24" class="ic"><path d="M4 9h5V4M20 15h-5v5M15 4v5h5M9 20v-5H4"/></svg>'
    : '<svg viewBox="0 0 24 24" class="ic"><path d="M9 4H4v5M15 20h5v-5M20 9V4h-5M4 15v5h5"/></svg>';
  if (!full) { S.needFit = true; fitView(); }     // граф вернулся — пересчитываем камеру
}

function showTab(name) {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("tab-active", x.dataset.tab === name));
  ["answer", "feed", "agents", "tools", "effort", "json"].forEach((n) => $("tab-" + n).classList.toggle("hidden", n !== name));
}

function initTabs() {
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));
}

function init() {
  initAppearance(); initTabs(); initCamera();
  $("btn-expand").addEventListener("click", toggleInspectorWidth);
  $("btn-copy").addEventListener("click", () => {
    if (S.preset) navigator.clipboard.writeText(JSON.stringify(S.preset.config, null, 2)).catch(() => {});
  });
  $("query").readOnly = true;
  ["btn-run", "btn-generate", "btn-baseline", "btn-judge", "btn-import",
   "p-play", "p-restart", "p-export", "btn-new"].forEach(id => {
    const el = $(id); if (el) { el.disabled = true; el.hidden = true; el.style.display = "none"; }
  });
  ["model-gen", "presets", "file-sources"].forEach(id => {
    const el = $(id); if (el) { const block = el.closest(".block"); if (block) block.hidden = true; }
  });
  showEmptyState();
}

document.addEventListener("DOMContentLoaded", init);

/* ───────── CoScientist live bridge ─────────
 * Mirrors native liveRun()'s bookkeeping (timer, progress bar, per-agent
 * input/output details, final-answer extraction) so a real fedot_tool run
 * looks the same here as a run started from this page's own "Запустить".
 */
function _coscientistLiveConnect() {
  let liveT0 = null, liveTimer = null, liveTotal = 1, liveFinished = 0, liveAgentIO = {};

  let currentRun = null;
  const evMs = (ev) => ev.ts * 1000;
  const display = (value) => typeof value === "string" ? value : JSON.stringify(value, null, 2);
  function clearRun(meta) {
    clearInterval(liveTimer); liveTimer = null; liveT0 = null;
    currentRun = meta; liveAgentIO = {}; liveTotal = 1; liveFinished = 0;
    resetRun(); showEmptyState();
    S.agents = new Map(); S.query = meta?.task || "";
    ["agent-cards", "tool-cards", "json"].forEach(id => { if ($(id)) $(id).textContent = ""; });
    $("scenario-title").textContent = meta ? "FEDOT · " + meta.run_id.slice(0, 8) : "Запуск не выбран";
    $("scenario-sub").textContent = meta ? "Сохранённый журнал запуска · " + meta.status : "Выберите запуск из этой сессии";
    $("query").value = meta?.task || "";
  }
  function handleEvent(ev) {
    if (ev.type === "config") {
      const preset = presetFromConfig(ev.config, "FEDOT · " + ev.run_id.slice(0, 8));
      preset.id = "__session__";
      preset.real = true;
      preset.query = currentRun?.task || "";
      const previousFeed = $("feed").innerHTML;
      const tokens = S.tokens;
      loadPreset(preset);
      $("feed").innerHTML = previousFeed;
      S.tokens = tokens; $("m-tokens").textContent = nfmt(tokens);
      $("btn-run").title = "Запуск выполняется из чата CoScientist";
      showTab("feed");
      liveTotal = $("graph").querySelectorAll(".node").length || 1;
      liveFinished = 0;
      return;
    }
    if (ev.type === "run_start") {
      clearRun(currentRun);
      liveAgentIO = {};
      liveT0 = evMs(ev);
      clearInterval(liveTimer);
      if (currentRun?.status === "running" && !currentRun.imported) liveTimer = setInterval(() => {
        $("m-time").textContent = fmtTime((Date.now() - liveT0) / 1000);
      }, 200);
      setPlayIcon(true);
      liveMessage("запуск", "runner", "Система запущена на реальных моделях. Первые ответы агентов появятся здесь.");
      return;
    }
    if (ev.type === "agent_start") {
      liveActivate(ev.agent);
      liveAgentIO[ev.agent] = {
        instruction: ev.instruction || "",
        incoming: Object.entries(ev.incoming || {}).map(([k, v]) => `${k}:\n${v}`).join("\n\n"),
      };
      liveMessage("старт", ev.agent, (ev.instruction || "").slice(0, 200));
      return;
    }
    if (ev.type === "agent_done") {
      liveFinish(ev.agent);
      liveFinished++;
      $("p-fill").style.width = Math.min(100, (100 * liveFinished) / liveTotal).toFixed(0) + "%";
      const io = Object.assign(liveAgentIO[ev.agent] || {},
        { output: ev.output || "", outputKey: ev.output_key || "" });
      liveAgentIO[ev.agent] = io;
      const head = (io.output || "").trim().split("\n")[0].slice(0, 160);
      liveMessage("результат", ev.agent, head || "(агент завершил работу)");
      const node = document.querySelector(".msg:last-child");
      if (node && !node.querySelector(".msg-io")) {
        const d = document.createElement("details");
        d.className = "msg-io";
        d.innerHTML = `<summary>вход и выход</summary>
          <div class="io-label">инструкция агента</div><pre>${esc(io.instruction || "—")}</pre>
          <div class="io-label">на входе из состояния</div><pre>${esc(io.incoming || "—")}</pre>
          <div class="io-label">результат агента${io.outputKey ? " → " + esc(io.outputKey) : ""}</div><pre>${esc(io.output || "—")}</pre>`;
        node.appendChild(d);
      }
      return;
    }
    if (ev.type === "tool") {
      const routing = ev.tool === "transfer_to_agent";
      const text = routing ? `Передаёт задачу агенту ${ev.target || "…"}`
                           : (ev.args ? `${ev.tool}(${ev.args})` : `Вызов инструмента ${ev.tool}`);
      liveMessage(routing ? "маршрутизация" : "инструмент", ev.agent, text, routing ? null : ev.tool);
      return;
    }
    if (ev.type === "tool_result") {
      if (ev.error) liveMessage("ошибка инструмента", ev.agent, ev.text, ev.tool);
      return;
    }
    if (ev.type === "text") {
      S.tokens += ev.tokens || 0;
      $("m-tokens").textContent = nfmt(S.tokens);
      liveMessage("шаг агента", ev.agent, ev.text);
      return;
    }
    if (ev.type === "tokens") {
      S.tokens += ev.tokens || 0;
      $("m-tokens").textContent = nfmt(S.tokens);
      return;
    }
    if (ev.type === "run_end") {
      clearInterval(liveTimer);
      liveTimer = null;
      if (liveT0 != null) $("m-time").textContent = fmtTime((evMs(ev) - liveT0) / 1000);
      setPlayIcon(false);
      $("p-fill").style.width = "100%";

      // Same "skip the critic's verdict, take the most substantial artifact"
      // extraction as native liveRun's "done" handler — the last agent in the
      // pipeline is often a reviewer, not the one holding the actual answer.
      if (ev.status === "success" && ev.state && Object.keys(ev.state).length) {
        const criticRe = /критик|critic|валид|valid|провер|review|judge|качеств|контрол|аудит|реценз|quality/i;
        const cfg = S.preset?.config || {};
        const cfgAgents = cfg.agents || [cfg.coordinator, ...(cfg.workers || [])].filter(Boolean);
        const criticKeys = new Set(cfgAgents.filter((a) => criticRe.test(a.name))
                                            .map((a) => a.output_key).filter(Boolean));
        const keys = Object.keys(ev.state).filter((k) => k !== "user_query" && !k.startsWith("_") && display(ev.state[k] ?? "").trim());
        const useful = keys.filter((k) => !criticKeys.has(k));
        const pool = useful.length ? useful : keys;
        let last = pool.length ? display(ev.state[pool[pool.length - 1]] ?? "") : "";
        const longest = pool.reduce((best, k) => {
          const v = display(ev.state[k] ?? "");
          return v.length > best.length ? v : best;
        }, "");
        if (longest.length > last.length * 2) last = longest;
        if (last) {
          S.answer = { text: last, meta: (currentRun?.engine || "FEDOT").toUpperCase() + " · " + ev.run_id.slice(0, 8) };
          S.baseline = null; S.judge = null;
          renderAnswer();
        }
      }

      if (ev.status === "success" && !S.answer && typeof ev.result === "string") {
        S.answer = {text: ev.result, meta: "FEDOT · " + ev.run_id.slice(0, 8)};
        renderAnswer();
      }
      liveMessage(ev.status === "success" ? "готово" : "ошибка", "система",
                  ev.status === "success" ? "Прогон завершён." : (ev.error || "Прогон завершился с ошибкой."));
      return;
    }
  }
  window.FedotRuns({
    host: $("fedot-runs"), onReset: clearRun, onEvent: handleEvent,
    onStatus: () => { clearInterval(liveTimer); liveTimer = null; }
  });
}
document.addEventListener("DOMContentLoaded", _coscientistLiveConnect);

})();
