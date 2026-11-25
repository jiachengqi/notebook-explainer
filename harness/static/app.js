// Notebook Explainer UI. Plain JavaScript, no build step. All text is inserted with
// textContent / DOM nodes (never innerHTML), because notebooks and model output are untrusted.
"use strict";

const S = { config: null, nb: null, facts: null, nbId: null, run: null, tab: "overview", poll: null, active: null };
const AUD = { manager: "Manager", developer: "Developer", agent: "AI agent" };
const MARK = { verified: "✓", partial: "≈", unchecked: "○", unverified: "⚠", invalid: "⚠", dropped: "×" };
const STATUS_HELP = {
  verified: "Verified: the evidence quote and every name in the claim appear in the cited lines.",
  partial: "Partly supported: some of what the claim names is not in the cited lines.",
  unchecked: "Cited: the lines exist, but the claim has nothing that can be matched word for word.",
  unverified: "Not supported by the cited lines (kept and flagged after repair).",
  invalid: "The citation points outside the notebook.",
};

const $ = (id) => document.getElementById(id);

function el(tag, attrs, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    n.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return n;
}

async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json", "X-Harness": "1" }, body: JSON.stringify(body),
  };
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

function banner(msg, kind) {
  const b = $("banner");
  if (!msg) { b.hidden = true; return; }
  b.textContent = msg;
  b.className = "banner" + (kind === "info" ? " info" : "");
  b.hidden = false;
}

// ---- citations -------------------------------------------------------------

function parseCite(c) {
  const m = /^C(\d+):L(\d+)(?:-L?(\d+))?$/.exec(c || "");
  if (!m) return null;
  return { cell: +m[1], start: +m[2], end: m[3] ? +m[3] : +m[2] };
}

function citeExists(c) {
  const p = parseCite(c);
  if (!p || !S.nb) return false;
  const cell = S.nb.cell_list[p.cell - 1];
  return !!cell && p.start >= 1 && p.end <= cell.lines.length;
}

function citeText(c) {
  const p = parseCite(c);
  if (!p || !citeExists(c)) return "";
  return S.nb.cell_list[p.cell - 1].lines.slice(p.start - 1, p.end).map((l) => l.text).join("\n");
}

function showCite(c, chip) {
  const p = parseCite(c);
  if (!p || !citeExists(c)) return;
  document.querySelectorAll(".row.hl").forEach((r) => r.classList.remove("hl", "flash"));
  document.querySelectorAll(".chip.active").forEach((x) => x.classList.remove("active"));
  if (chip) chip.classList.add("active");
  let first = null;
  for (let n = p.start; n <= p.end; n++) {
    const row = document.getElementById(`L-${p.cell}-${n}`);
    if (row) { row.classList.add("hl", "flash"); first = first || row; }
  }
  if (first) first.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
  const cell = S.nb.cell_list[p.cell - 1];
  const fl = cell.lines[p.start - 1].file_line;
  $("code-sel").textContent = `${c} · cell ${p.cell} (${cell.lang})` + (fl ? ` · file line ${fl}` : "");
}

function chip(c) {
  const ok = citeExists(c);
  const b = el("button", { class: "chip" + (ok ? "" : " bad"), type: "button",
    title: ok ? citeText(c).slice(0, 400) : "This address does not exist in the notebook" }, c);
  if (ok) b.addEventListener("click", () => showCite(c, b));
  return b;
}

function richText(text) {
  // `code` spans become <code>; everything else stays text.
  const parts = String(text).split("`");
  return parts.map((p, i) => (i % 2 ? el("code", {}, p) : document.createTextNode(p)));
}

// ---- code pane ---------------------------------------------------------------

const KW = new Set(("def return if elif else for while in not and or is import from as with try except finally raise " +
  "lambda class None True False pass break continue global yield select from where group by order join left inner " +
  "on as and or not null case when then end insert into overwrite merge update set delete create replace table view " +
  "using having limit union all distinct with").split(" "));

function highlight(text, lang) {
  const frag = document.createDocumentFragment();
  const isSql = lang === "sql";
  const re = isSql
    ? /(--.*$)|('(?:[^'\\]|\\.)*')|(\b\d+(?:\.\d+)?\b)|([A-Za-z_]\w*)/g
    : /(#.*$)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\b\d+(?:\.\d+)?\b)|([A-Za-z_]\w*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) frag.append(text.slice(last, m.index));
    let cls = null;
    if (m[1]) cls = "tok-com";
    else if (m[2]) cls = "tok-str";
    else if (m[3]) cls = "tok-num";
    else if (m[4] && KW.has(isSql ? m[4].toLowerCase() : m[4])) cls = "tok-kw";
    frag.append(cls ? el("span", { class: cls }, m[0]) : m[0]);
    last = m.index + m[0].length;
  }
  if (last < text.length) frag.append(text.slice(last));
  return frag;
}

function renderCode() {
  const box = $("code");
  box.replaceChildren();
  if (!S.nb) return;
  $("code-title").textContent = `${S.nb.name} · ${S.nb.cells} cells · ${S.nb.file_lines || S.nb.cell_lines} lines`;
  for (const cell of S.nb.cell_list) {
    const head = el("div", { class: "cell-head" }, el("b", {}, `Cell ${cell.idx}`), ` · ${cell.lang}`,
      cell.title ? ` · ${cell.title}` : "", cell.lines.length ? "" : " · empty");
    const rows = cell.lines.map((l) => el("div", { class: "row", id: `L-${cell.idx}-${l.n}` },
      el("span", { class: "ln" }, l.n), el("span", { class: "fl", title: "line in the file" }, l.file_line || ""),
      el("span", { class: "tx" }, highlight(l.text, cell.lang))));
    box.append(el("div", { class: "cell", id: `cell-${cell.idx}` }, head, ...rows));
  }
}

// ---- panels ------------------------------------------------------------------

function setTab(tab) {
  S.tab = tab;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  renderPanel();
}

function renderPanel() {
  const panel = $("panel");
  panel.replaceChildren();
  if (!S.nb) { panel.append(el("div", { class: "empty" }, el("h2", {}, "Point it at a notebook"),
    el("p", {}, "Type a path or choose a file, then press Generate."))); return; }
  if (S.tab === "overview") return renderOverview(panel);
  if (S.tab === "report") return renderReport(panel);
  const doc = S.run && S.run.docs && S.run.docs[S.tab];
  if (!doc) {
    panel.append(el("div", { class: "empty" }, el("p", {},
      S.run && S.run.status === "running" ? "Generating… (see Overview for progress)" : "Press Generate to write this document.")));
    return;
  }
  renderDoc(panel, doc);
}

function statusPills(stats) {
  return el("div", { class: "stats" },
    el("span", { class: "pill" }, `${stats.claims} claims`),
    el("span", { class: "pill verified", title: STATUS_HELP.verified }, `✓ ${stats.verified} verified`),
    stats.partial ? el("span", { class: "pill partial", title: STATUS_HELP.partial }, `≈ ${stats.partial} partial`) : null,
    stats.unchecked ? el("span", { class: "pill", title: STATUS_HELP.unchecked }, `○ ${stats.unchecked} unchecked`) : null,
    stats.unverified + stats.invalid ? el("span", { class: "pill unverified", title: STATUS_HELP.unverified },
      `⚠ ${stats.unverified + stats.invalid} unverified`) : null,
    stats.repaired ? el("span", { class: "pill" }, `${stats.repaired} repaired`) : null,
    stats.static_added ? el("span", { class: "pill static" }, `⚙ ${stats.static_added} added by static analysis`) : null);
}

function renderDoc(panel, doc) {
  const box = el("div", { class: "doc" });
  box.append(el("h2", {}, doc.title));
  const p = S.run.provider;
  box.append(el("div", { class: "meta" }, `For ${AUD[doc.audience]} · ${p.name} / ${p.model} · ` +
    "every claim cites cell:line; click a citation to see the code"));
  box.append(statusPills(doc.stats));
  for (const sec of doc.sections) {
    const claims = sec.claims.filter((c) => c.status !== "dropped");
    if (!claims.length) continue;
    box.append(el("h3", {}, sec.heading));
    const ul = el("ul", { class: "claims" });
    for (const c of claims) {
      const st = c.origin === "static" ? "static" : c.status;
      const li = el("li", { class: `claim ${c.status} ${c.origin === "static" ? "static" : ""}` },
        el("span", { class: "mark", title: c.origin === "static" ? "Added by static analysis: the model did not cite this red flag."
          : STATUS_HELP[c.status] || "" }, c.origin === "static" ? "⚙" : MARK[c.status] || ""),
        el("div", { class: "text", title: c.evidence ? `evidence: ${c.evidence}` : "" },
          c.kind && c.kind !== "fact" ? el("span", { class: `kind ${c.kind}` }, c.kind) : null,
          richText(c.text), el("span", { class: "cites" }, c.citations.map(chip))),
        ["unverified", "invalid", "partial"].includes(c.status) && c.problems && c.problems.length
          ? el("div", { class: "problems" }, c.problems.slice(0, 2).join(" · ")) : null);
      li.dataset.status = st;
      ul.append(li);
    }
    box.append(ul);
  }
  panel.append(box);
}

function renderOverview(panel) {
  const s = S.nb, f = S.facts;
  const box = el("div", { class: "doc" });
  box.append(el("h2", {}, s.name));
  box.append(el("div", { class: "meta" }, `${s.format} · sha256 ${s.sha256.slice(0, 12)} · ${s.cells} cells · ` +
    `${s.cell_lines} lines in cells · ` + Object.entries(s.languages).map(([k, v]) => `${k} ${v}`).join(", ")));
  if (S.run) {
    box.append(el("h3", {}, S.run.status === "running" ? el("span", { class: "spinner" }) : null,
      `Run ${S.run.id} · ${S.run.status}`));
    const lines = S.run.events.map((e) => `${String(e.t).padStart(6)}s  ${e.stage.padEnd(9)} ${e.audience ? "[" + e.audience + "] " : ""}${e.msg}`);
    const ev = el("div", { class: "events" }, lines.join("\n"));
    box.append(ev);
    setTimeout(() => { ev.scrollTop = ev.scrollHeight; }, 0);
  }
  box.append(el("h3", {}, `Static red flags (${f.flags.length}), found without a model`));
  const ul = el("ul", { class: "claims flags" });
  for (const fl of f.flags) {
    ul.append(el("li", { class: "claim" }, el("span", { class: "mark" }, ""),
      el("div", { class: "text" }, el("span", { class: `sev ${fl.severity}` }, fl.severity), richText(fl.title),
        el("span", { class: "cites" }, fl.citations.map(chip)))));
  }
  box.append(ul);
  if (f.parse_errors.length) box.append(el("p", { class: "problems" }, f.parse_errors.join(" · ")));
  const facts = [
    ["Parameters", f.widgets.map((w) => [`${w.name} = ${w.default}`, w.cite])],
    ["External code", f.runs.map((r) => [`%run ${r.target}`, r.cite]).concat(
      Object.entries(f.externals).map(([n, c]) => [`${n} (not defined here)`, c[0]]))],
    ["Writes", f.writes.map((w) => [`${w.target} · ${w.mode}${w.guard ? " · only when " + w.guard : ""}`, w.cite])],
  ];
  const t = el("table", { class: "kv" });
  for (const [label, rows] of facts) {
    if (!rows.length) continue;
    rows.forEach(([txt, c], i) => t.append(el("tr", {}, el("th", {}, i ? "" : label), el("td", {}, txt, " ", chip(c)))));
  }
  box.append(el("h3", {}, "What the notebook touches"), t);
  panel.append(box);
}

function renderReport(panel) {
  const box = el("div", { class: "doc" });
  if (!S.run || !S.run.report || !S.run.report.documents) {
    box.append(el("p", { class: "empty" }, "The validation report appears after a run."));
    panel.append(box); return;
  }
  const r = S.run.report;
  box.append(el("h2", {}, "Validation report"));
  box.append(el("div", { class: "meta" }, `${r.provider.name} / ${r.provider.model} · ${r.seconds}s · ` +
    `${r.usage.calls} provider calls · ${r.usage.input_tokens} in / ${r.usage.output_tokens} out tokens · ` +
    `${r.usage.cache_hits} cache hits`));
  const t = el("table", { class: "kv" }, el("tr", {}, ...["document", "claims", "verified", "partial", "unchecked",
    "unverified", "repaired", "repair rounds", "red flags cited", "added by static"].map((h) => el("th", {}, h))));
  for (const [a, d] of Object.entries(r.documents)) {
    t.append(el("tr", {}, el("td", {}, AUD[a]), el("td", {}, d.claims), el("td", {}, d.verified), el("td", {}, d.partial),
      el("td", {}, d.unchecked), el("td", {}, d.unverified + d.invalid), el("td", {}, d.repaired),
      el("td", {}, d.repair_rounds), el("td", {}, `${d.coverage.flags_cited}/${d.coverage.flags_total}`),
      el("td", {}, d.static_added)));
  }
  box.append(t);
  box.append(el("p", { class: "meta" }, "How to read it: every citation was checked in code. ✓ means the evidence quote and " +
    "every identifier, literal and number in the claim appear in the cited lines. This proves the citation supports " +
    "the words of the claim; it does not prove the interpretation is right."));
  box.append(el("h3", {}, "Files"), el("div", { class: "downloads" },
    Object.keys(S.run.docs).map((a) => el("a", { href: `/api/runs/${S.run.id}/${a}.md` }, `${a}.md`)),
    el("div", { class: "meta" }, `Saved in ${S.run.out_dir}`)));
  if (r.notes && r.notes.length) box.append(el("h3", {}, "Notes"), el("div", { class: "events" }, r.notes.join("\n")));
  if (r.redacted_lines.length) box.append(el("p", { class: "meta" }, `Redacted before sending: ${r.redacted_lines.join(", ")}`));
  box.append(el("h3", {}, "Prompt hashes (for reproducibility)"),
    el("div", { class: "events" }, Object.entries(r.prompt_hashes).map(([k, v]) => `${k.padEnd(22)} ${v}`).join("\n")));
  panel.append(box);
}

function updateTabCounts() {
  document.querySelectorAll("#tabs button").forEach((b) => {
    const d = S.run && S.run.docs && S.run.docs[b.dataset.tab];
    b.querySelector(".count")?.remove();
    if (d) b.append(el("span", { class: "count" }, d.stats.claims));
  });
}

// ---- actions -----------------------------------------------------------------

async function loadNotebook(payload) {
  banner(null);
  try {
    const res = await api("/api/load", payload);
    S.nb = res.notebook; S.facts = res.facts; S.nbId = res.notebook_id; S.run = null;
    renderCode(); updateTabCounts(); setTab("overview");
    $("generate").disabled = false;
    $("code-sel").textContent = "";
  } catch (e) { banner(e.message); }
}

async function generate() {
  if (!S.nbId) return;
  banner(null);
  $("generate").disabled = true;
  try {
    const res = await api("/api/generate", { notebook_id: S.nbId, provider: $("provider").value,
      model: $("model").value, use_cache: $("use-cache").checked });
    S.run = { id: res.run_id, status: "running", events: [], docs: {}, provider: {} };
    setTab("overview");
    clearInterval(S.poll);
    S.poll = setInterval(pollRun, 800);
  } catch (e) { banner(e.message); $("generate").disabled = false; }
}

async function pollRun() {
  try {
    const st = await api(`/api/runs/${S.run.id}`);
    S.run = st;
    updateTabCounts();
    if (st.status === "done" || st.status === "error") {
      clearInterval(S.poll);
      $("generate").disabled = false;
      if (st.status === "error") { banner(`Run failed: ${st.error}`); renderPanel(); return; }
      const errs = (st.report.errors || []);
      if (errs.length) banner(`Some documents failed: ${errs.join(" · ")}`);
      setTab(S.tab === "overview" ? Object.keys(st.docs)[0] || "overview" : S.tab);
      return;
    }
    if (S.tab === "overview") renderPanel();
  } catch (e) { clearInterval(S.poll); banner(e.message); $("generate").disabled = false; }
}

async function init() {
  document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => setTab(b.dataset.tab)));
  $("load-form").addEventListener("submit", (e) => { e.preventDefault(); const p = $("path").value.trim(); if (p) loadNotebook({ path: p }); });
  $("file").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (f) loadNotebook({ filename: f.name, content: await f.text() });
    e.target.value = "";
  });
  $("generate").addEventListener("click", generate);
  try {
    S.config = await api("/api/config");
    const sel = $("provider");
    for (const p of S.config.providers) {
      sel.append(el("option", { value: p.name }, p.name + (p.key_present ? "" : ` (set ${p.key_env})`)));
    }
    sel.value = S.config.default_provider;
    const sync = () => { const p = S.config.providers.find((x) => x.name === sel.value); $("model").value = p ? p.model : ""; };
    sel.addEventListener("change", sync);
    sync();
  } catch (e) { banner(`Cannot reach the local server: ${e.message}`); }
}

init();
