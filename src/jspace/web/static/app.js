"use strict";

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
let currentPrompt = "";
let currentTokens = [];
let currentMethod = "logit";
// Accumulated branch edits (position -> replacement token) applied to the
// current prompt/conversation. Cleared when a fresh lens run starts.
let branchEdits = [];
// token -> English gloss (accumulated across runs).
const translations = {};
// Every top token currently shown in the grid, for the translate pass.
let shownTokens = new Set();

function TR(token) {
  return translations[token] || token;
}

async function refreshStatus() {
  try {
    const r = await fetch("/api/status");
    const s = await r.json();
    if (s.loaded) {
      const i = s.info || {};
      const nf = (s.fitted_layers || []).length;
      const fitted = nf ? ` · J-lens fitted: ${nf} layer${nf === 1 ? "" : "s"}` : "";
      statusEl.textContent = `${i.architecture || "model"} · ${i.num_layers} layers · hidden ${i.hidden_size} · vocab ${i.vocab_size}${fitted}`;
      statusEl.className = "status ok";
    } else if (s.loading) {
      statusEl.textContent = "loading model…";
      statusEl.className = "status";
      setTimeout(refreshStatus, 1500);
    } else if (s.error) {
      statusEl.textContent = "error: " + s.error;
      statusEl.className = "status err";
    } else {
      statusEl.textContent = `model not loaded (${s.model_path}) — will load on first run`;
      statusEl.className = "status";
    }
  } catch (e) {
    statusEl.textContent = "backend unreachable";
    statusEl.className = "status err";
  }
}

function parseLayers() {
  const raw = $("layers").value.trim();
  if (!raw) return null;
  const nums = raw.split(",").map((x) => parseInt(x.trim(), 10)).filter((n) => !isNaN(n));
  // Deduplicate (preserve order) so grid cell ids stay unique.
  return [...new Set(nums)];
}

function parseCorpus() {
  const raw = $("corpus").value.trim();
  if (!raw) return null;
  return raw.split("\n").map((l) => l.trim()).filter(Boolean);
}

function buildRequest() {
  return {
    prompt: $("prompt").value,
    method: $("method").value,
    top_k: parseInt($("topk").value, 10) || 6,
    layers: parseLayers(),
    use_chat_template: $("chat").checked,
    corpus: parseCorpus(),
  };
}

function initGrid(header) {
  currentTokens = header.tokens;
  currentMethod = header.method;
  shownTokens = new Set();
  const table = $("grid");
  table.innerHTML = "";
  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  hr.appendChild(th("layer \\ token"));
  header.tokens.forEach((t, i) => hr.appendChild(th(fmtTok(t), `col-${i}`)));
  thead.appendChild(hr);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  tbody.id = "grid-body";
  // Rows are added top (last layer) to bottom so output-side is on top,
  // matching the paper's layer diagrams (output at top).
  header.layers.slice().reverse().forEach((ell) => {
    const tr = document.createElement("tr");
    tr.id = `row-${ell}`;
    const rowHead = th(`L${ell}`);
    rowHead.scope = "row";
    tr.appendChild(rowHead);
    header.tokens.forEach((_, i) => {
      const td = document.createElement("td");
      td.id = `cell-${ell}-${i}`;
      td.textContent = "·";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  $("legend").classList.remove("hidden");
}

function th(text, cls) {
  const el = document.createElement("th");
  el.textContent = text;
  if (cls) el.className = cls;
  return el;
}

function fmtTok(t) {
  if (t === "\n") return "↵";
  if (t.trim() === "") return "␣";
  return t.length > 12 ? t.slice(0, 12) + "…" : t;
}

// Paint a cell's label. When translation is on and the token has an English
// gloss that differs from the raw token, show "gloss" with the raw token faded.
function paintCell(td, best) {
  const raw = best.token;
  const gloss = TR(raw);
  const translateOn = $("translate") && $("translate").checked;
  let label;
  if (translateOn && gloss !== raw) {
    label =
      `<span class="cell-top">${escapeHtml(fmtTok(gloss))}</span>` +
      ` <span class="cell-rest">(${escapeHtml(fmtTok(raw))})</span>`;
  } else {
    label = `<span class="cell-top">${escapeHtml(fmtTok(raw))}</span>`;
  }
  td.innerHTML = label + ` <span class="cell-score">${best.score.toFixed(2)}</span>`;
}

function renderCell(cell) {
  const td = $(`cell-${cell.layer}-${cell.position}`);
  if (!td) return;
  const top = cell.top;
  if (!top || !top.length) { td.textContent = ""; return; }
  const best = top[0];
  // Remember the raw token so a later translation pass can re-render it.
  td.dataset.token = best.token;
  td.dataset.score = String(best.score);
  shownTokens.add(best.token);
  paintCell(td, best);
  td.title = top.map((t) => `${JSON.stringify(t.token)}  ${t.score.toFixed(3)}`).join("\n");
  // Shade by confidence.
  const a = Math.min(0.55, best.score * 0.6);
  td.style.background = `rgba(110,168,254,${a.toFixed(3)})`;
  td.onclick = () => decompose(cell.layer, cell.position);
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

async function runLens() {
  const req = buildRequest();
  currentPrompt = req.prompt;
  branchEdits = [];            // a fresh run starts from the unbranched context
  if (typeof renderBranchBanner === "function") renderBranchBanner();
  $("run").disabled = true;
  statusEl.textContent = "running lens…";
  statusEl.className = "status";
  try {
    // The Jacobian lens needs its operator fitted once over the corpus. Fit
    // first (with visible progress); already-fitted layers are skipped.
    if (req.method === "jacobian") {
      await fitJacobian(req);
    }
    const resp = await fetch("/api/lens/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!resp.ok || !resp.body) {
      const t = await resp.text();
      throw new Error(t || resp.statusText);
    }
    await readSSE(resp, handleEvent);
    refreshStatus();
  } catch (e) {
    statusEl.textContent = "lens error: " + e.message;
    statusEl.className = "status err";
  } finally {
    $("run").disabled = false;
  }
}

// Read an SSE response body, calling onEvent(obj) per `data:` event.
async function readSSE(resp, onEvent) {
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 2);
      if (chunk.startsWith("data:")) onEvent(JSON.parse(chunk.slice(5).trim()));
    }
  }
}

// Fit the Jacobian operator for the requested layers, showing progress.
async function fitJacobian(req) {
  const resp = await fetch("/api/fit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      layers: req.layers,
      use_chat_template: req.use_chat_template,
      corpus: req.corpus,
    }),
  });
  if (!resp.ok || !resp.body) throw new Error(await resp.text());
  let needFit = false;
  await readSSE(resp, (ev) => {
    if (ev.type === "fit_start") {
      const todo = (ev.layers || []).filter(
        (l) => !(ev.already_fitted || []).includes(l));
      needFit = todo.length > 0;
      if (needFit) {
        statusEl.textContent =
          `fitting Jacobian lens (${ev.corpus_size} contexts, one-time)…`;
        statusEl.className = "status";
      }
    } else if (ev.type === "fit_progress") {
      statusEl.textContent =
        `fitting Jacobian lens… context ${ev.done}/${ev.total}`;
    } else if (ev.type === "error") {
      throw new Error(ev.detail);
    }
  });
}
function handleEvent(ev) {
  if (ev.type === "header") {
    initGrid(ev);
  } else if (ev.type === "layer") {
    ev.cells.forEach(renderCell);
    const row = $(`row-${ev.layer}`);
    if (row) row.querySelector("th").style.color = "var(--good)";
  } else if (ev.type === "done") {
    statusEl.textContent = "done";
    statusEl.className = "status ok";
    if ($("translate").checked) applyTranslations();
  } else if (ev.type === "error") {
    statusEl.textContent = "lens error: " + ev.detail;
    statusEl.className = "status err";
  }
}

// Translate every top token currently shown, then repaint the grid cells.
async function applyTranslations() {
  const need = [...shownTokens].filter(
    (t) => !(t in translations) && /[^\x00-\x7f]/.test(t));
  if (need.length) {
    const prev = statusEl.textContent;
    statusEl.textContent = `translating ${need.length} tokens…`;
    try {
      const r = await fetch("/api/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tokens: need }),
      });
      if (r.ok) {
        const data = await r.json();
        Object.assign(translations, data.translations || {});
      }
    } catch (e) { /* leave raw tokens on failure */ }
    statusEl.textContent = prev;
  }
  repaintGrid();
}

// Repaint all populated cells honoring the current translate toggle. Uses the
// stored raw token; score is re-parsed from the existing score span.
function repaintGrid() {
  document.querySelectorAll("#grid tbody td").forEach((td) => {
    const raw = td.dataset.token;
    if (raw === undefined) return;
    const score = parseFloat(td.dataset.score);
    paintCell(td, { token: raw, score: isNaN(score) ? 0 : score });
  });
}

async function decompose(layer, position) {
  const panel = $("panel");
  const body = $("panel-body");
  panel.classList.remove("hidden");
  $("panel-title").textContent = `J-space @ layer ${layer}, token ${position}`;
  body.innerHTML = "<div class='meta'>decomposing…</div>";
  try {
    const r = await fetch("/api/decompose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: currentPrompt,
        layer, position,
        use_chat_template: (mode === "chat") ? true : $("chat").checked,
        corpus: parseCorpus(),
        messages: (mode === "chat" && conversation.length) ? conversation : null,
        at_generation_point: true,
      }),
    });
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();
    // Translate any non-English atoms before rendering, if enabled.
    if ($("translate").checked) {
      const need = d.atoms.map((a) => a.token)
        .filter((t) => !(t in translations) && /[^\x00-\x7f]/.test(t));
      if (need.length) {
        try {
          const tr = await fetch("/api/translate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ tokens: need }),
          });
          if (tr.ok) Object.assign(translations, (await tr.json()).translations || {});
        } catch (e) { /* ignore */ }
      }
    }
    renderDecomp(d);
  } catch (e) {
    body.innerHTML = `<div class='meta err'>error: ${escapeHtml(e.message)}</div>`;
  }
}

// Replace the token at `position` with `tokenId` and re-run the lens on the
// branched context, so we can see how a change far back affects downstream
// thinking. Works in completion mode (updates the prompt to the branched text);
// in chat mode it branches the templated context for this run.
async function branchFrom(position, tokenId, label) {
  const method = (mode === "chat") ? $("chat-method").value : $("method").value;
  currentMethod = method;
  $("panel").classList.add("hidden");
  statusEl.textContent = `branching: token ${position} \u2192 ${label}\u2026`;
  statusEl.className = "status";
  try {
    if (method === "jacobian") {
      // Make sure the operator is fitted so the branched run is fast.
      const fr = await fetch("/api/fit", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ layers: parseLayers() }),
      });
      if (fr.ok) await readSSE(fr, (ev) => {
        if (ev.type === "fit_progress") statusEl.textContent = `fitting\u2026 ${ev.done}/${ev.total}`;
      });
    }
    // Accumulate this edit on top of any prior branch edits at other positions.
    // A repeat edit at the same position overrides the earlier one.
    branchEdits = branchEdits.filter((e) => e.position !== position);
    branchEdits.push({ position, branch_token_id: tokenId, label });
    branchEdits.sort((a, b) => a.position - b.position);
    const body = {
      edits: branchEdits.map((e) => ({ position: e.position, branch_token_id: e.branch_token_id })),
      method, layers: parseLayers(), top_k: parseInt($("topk").value, 10) || 6,
      corpus: parseCorpus(), at_generation_point: true,
    };
    if (mode === "chat" && conversation.length) body.messages = conversation;
    else { body.prompt = currentPrompt; body.use_chat_template = $("chat").checked; }
    const r = await fetch("/api/branch", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();
    renderLensResult(d);
    renderBranchBanner();
    statusEl.textContent = `branched \u2713 (${branchEdits.length} edit${branchEdits.length === 1 ? "" : "s"})`;
    statusEl.className = "status ok";
    if ($("translate").checked) applyTranslations();
  } catch (e) {
    // Roll back the just-added edit on failure.
    branchEdits = branchEdits.filter((e2) => e2.position !== position);
    statusEl.textContent = "branch error: " + e.message;
    statusEl.className = "status err";
  }
}

// Show a small banner listing active branch edits, with a reset control. Also
// mark the branched columns in the grid header.
function renderBranchBanner() {
  let bar = $("branch-bar");
  if (!bar) {
    bar = document.createElement("div");
    bar.id = "branch-bar";
    bar.className = "branch-bar";
    $("legend").insertAdjacentElement("afterend", bar);
  }
  if (!branchEdits.length) { bar.classList.add("hidden"); bar.innerHTML = ""; return; }
  bar.classList.remove("hidden");
  const chips = branchEdits.map((e) =>
    `<span class="branch-chip">pos ${e.position} \u2192 <b>${escapeHtml(fmtTok(TR(e.label)))}</b></span>`).join("");
  bar.innerHTML = `<span class="muted">\u2387 branched context:</span> ${chips}` +
    `<button id="branch-reset" class="secondary">reset</button>`;
  $("branch-reset").addEventListener("click", resetBranch);
  // Highlight the edited columns.
  branchEdits.forEach((e) => {
    const th = document.querySelector(`#grid thead th.col-${e.position}`);
    if (th) th.classList.add("branched-col");
  });
}

function resetBranch() {
  branchEdits = [];
  renderBranchBanner();
  runLens();  // re-run on the original, unbranched context
}

function renderDecomp(d) {
  const body = $("panel-body");
  const tok = currentTokens[d.position] !== undefined ? currentTokens[d.position] : "?";
  const varPct = (d.jspace_variance_fraction * 100).toFixed(1);
  const resPct = (d.residual_fraction * 100).toFixed(1);
  let html = `<div class="meta">token: <b>${escapeHtml(fmtTok(tok))}</b><br>` +
    `J-space variance captured: <b>${varPct}%</b><br>` +
    `residual norm: ${resPct}% of activation</div>`;
  const max = d.atoms.reduce((m, a) => Math.max(m, Math.abs(a.coefficient)), 1e-6);
  html += d.atoms.map((a) => {
    const w = Math.max(2, (Math.abs(a.coefficient) / max) * 100);
    const translateOn = $("translate").checked;
    const gloss = TR(a.token);
    const label = (translateOn && gloss !== a.token)
      ? `${escapeHtml(fmtTok(gloss))} <span class="cell-rest">(${escapeHtml(fmtTok(a.token))})</span>`
      : escapeHtml(fmtTok(a.token));
    const acts =
      `<span class="acts">` +
      `<button data-iv="steer" data-layer="${d.layer}" data-token="${a.token_id}" data-label="${escapeHtml(a.token)}">steer</button>` +
      `<button data-iv="swap" data-layer="${d.layer}" data-token="${a.token_id}" data-label="${escapeHtml(a.token)}">swap</button>` +
      `<button data-iv="ablate" data-layer="${d.layer}" data-token="${a.token_id}" data-label="${escapeHtml(a.token)}">ablate</button>` +
      `<button class="branch-btn" data-branch-token="${a.token_id}" data-label="${escapeHtml(a.token)}" title="Replace token ${d.position} with this and re-run the lens">\u2387 branch</button>` +
      `</span>`;
    return `<div class="atom"><span class="tok">${label}</span>` +
      `<span class="coef">${a.coefficient.toFixed(3)}</span>${acts}</div>` +
      `<div class="bar"><span style="width:${w}%"></span></div>`;
  }).join("");
  if (!d.atoms.length) html += "<div class='meta'>no active J-space atoms found</div>";
  body.innerHTML = html;
  // Wire the per-atom steer/swap/ablate buttons.
  body.querySelectorAll(".acts button[data-iv]").forEach((btn) => {
    btn.addEventListener("click", () => addIntervention({
      mode: btn.dataset.iv,
      layer: parseInt(btn.dataset.layer, 10),
      token_id: parseInt(btn.dataset.token, 10),
      label: btn.dataset.label,
    }));
  });
  // Wire the per-atom "branch from here" buttons. They replace the token at
  // THIS cell's position with the atom's token and re-run the lens.
  body.querySelectorAll(".branch-btn").forEach((btn) => {
    btn.addEventListener("click", () => branchFrom(
      d.position,
      parseInt(btn.dataset.branchToken, 10),
      btn.dataset.label,
    ));
  });
}

$("run").addEventListener("click", runLens);
$("panel-close").addEventListener("click", () => $("panel").classList.add("hidden"));
$("translate").addEventListener("change", () => {
  if ($("translate").checked) applyTranslations();
  else repaintGrid();
});
$("prompt").addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") runLens();
});

refreshStatus();

// ===================== Mode switching =====================
let mode = "completion";
function setMode(m) {
  mode = m;
  $("mode-completion").classList.toggle("active", m === "completion");
  $("mode-chat").classList.toggle("active", m === "chat");
  $("completion-controls").classList.toggle("hidden", m !== "completion");
  $("chat-controls").classList.toggle("hidden", m !== "chat");
}
$("mode-completion").addEventListener("click", () => setMode("completion"));
$("mode-chat").addEventListener("click", () => setMode("chat"));

// ===================== Chat =====================
let conversation = [];

function sampling() {
  const num = (id) => { const v = $(id).value.trim(); return v === "" ? null : parseFloat(v); };
  return {
    temperature: num("s-temp") ?? 0.7,
    top_p: num("s-topp"),
    top_k: $("s-topk").value.trim() === "" ? null : parseInt($("s-topk").value, 10),
    min_p: num("s-minp"),
    max_new_tokens: parseInt($("s-max").value, 10) || 200,
  };
}

function renderConversation() {
  const el = $("conversation");
  el.innerHTML = conversation.map((m) => {
    const body = escapeHtml(m.content).replace(
      /&lt;think&gt;([\s\S]*?)(&lt;\/think&gt;|$)/g,
      (_, t) => `<span class="think">\ud83d\udcad${t}</span>`);
    return `<div class="msg ${m.role}"><div class="who">${m.role}</div>${body}</div>`;
  }).join("");
  el.scrollTop = el.scrollHeight;
}

async function chatSend() {
  const text = $("chat-input").value.trim();
  if (!text) return;
  conversation.push({ role: "user", content: text });
  $("chat-input").value = "";
  renderConversation();
  const s = sampling();
  $("chat-send").disabled = true;
  statusEl.textContent = "generating\u2026"; statusEl.className = "status";
  try {
    const r = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: conversation, ...s }),
    });
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();
    conversation.push({ role: "assistant", content: d.reply });
    renderConversation();
    statusEl.textContent = "done"; statusEl.className = "status ok";
  } catch (e) {
    statusEl.textContent = "chat error: " + e.message; statusEl.className = "status err";
  } finally { $("chat-send").disabled = false; }
}

async function chatLens() {
  if (!conversation.length) return;
  const method = $("chat-method").value;
  currentMethod = method;
  const req = { messages: conversation, method, top_k: 6, at_generation_point: true };
  $("chat-lens").disabled = true;
  statusEl.textContent = "running lens on conversation\u2026"; statusEl.className = "status";
  try {
    if (method === "jacobian") {
      const fr = await fetch("/api/fit", { method: "POST",
        headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) });
      if (fr.ok) await readSSE(fr, (ev) => {
        if (ev.type === "fit_progress") statusEl.textContent = `fitting\u2026 ${ev.done}/${ev.total}`;
      });
    }
    const r = await fetch("/api/chat/lens", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();
    renderLensResult(d);
    statusEl.textContent = "done"; statusEl.className = "status ok";
    if ($("translate").checked) applyTranslations();
  } catch (e) {
    statusEl.textContent = "lens error: " + e.message; statusEl.className = "status err";
  } finally { $("chat-lens").disabled = false; }
}

// Render a full (non-streamed) lens result into the grid.
function renderLensResult(d) {
  initGrid({ tokens: d.tokens, layers: d.layers, method: d.method });
  d.cells.forEach(renderCell);
  // Persist the branched-column highlight across re-renders.
  branchEdits.forEach((e) => {
    const th = document.querySelector(`#grid thead th.col-${e.position}`);
    if (th) th.classList.add("branched-col");
  });
}

$("chat-send").addEventListener("click", chatSend);
$("chat-lens").addEventListener("click", chatLens);
$("chat-reset").addEventListener("click", () => { conversation = []; renderConversation(); });
$("chat-input").addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") chatSend();
});

// ===================== J-space interventions =====================
let interventions = [];

function addIntervention(iv) {
  if (iv.mode === "swap") {
    const to = prompt(`Swap \"${iv.label}\" for which word? (English word/token)`);
    if (!to) return;
    iv.swap_to_text = to;
  }
  iv.strength = iv.mode === "steer" ? 4 : 1;
  interventions.push(iv);
  renderInterventions();
  $("intervention-bar").classList.remove("hidden");
}

function renderInterventions() {
  const el = $("iv-list");
  if (!interventions.length) { el.innerHTML = "<span class='muted'>no edits \u2014 click steer/swap/ablate on a J-space atom</span>"; return; }
  el.innerHTML = interventions.map((iv, i) => {
    let mid = `<span class="tok">${escapeHtml(fmtTok(iv.label))}</span> <span class="muted">L${iv.layer}</span>`;
    if (iv.mode === "swap") mid += ` \u2192 <span class="tok">${escapeHtml(iv.swap_to_text)}</span>`;
    let slider = "";
    if (iv.mode === "steer") {
      slider = `<input type="range" min="-8" max="8" step="0.5" value="${iv.strength}" data-i="${i}" class="iv-strength"/><span class="muted" id="ivs-${i}">${iv.strength}</span>`;
    }
    return `<div class="iv-chip ${iv.mode}"><span class="tag">${iv.mode}</span>${mid}${slider}<span class="rm" data-rm="${i}">\u00d7</span></div>`;
  }).join("");
  el.querySelectorAll(".iv-strength").forEach((s) => s.addEventListener("input", () => {
    const i = parseInt(s.dataset.i, 10);
    interventions[i].strength = parseFloat(s.value);
    $(`ivs-${i}`).textContent = s.value;
  }));
  el.querySelectorAll("[data-rm]").forEach((x) => x.addEventListener("click", () => {
    interventions.splice(parseInt(x.dataset.rm, 10), 1); renderInterventions();
  }));
}

async function resolveSwapTargets() {
  // Turn swap_to_text into a token_id via the tokenizer on the server side is
  // not exposed; instead we send the text and let /api/intervene map it. We map
  // client-side using a tiny /api/translate-free trick: encode via a dedicated
  // endpoint. Here we piggyback on the model by asking the server to tokenize.
  for (const iv of interventions) {
    if (iv.mode === "swap" && iv.swap_to == null && iv.swap_to_text) {
      const r = await fetch("/api/tokenize", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: iv.swap_to_text }),
      });
      if (r.ok) { const d = await r.json(); iv.swap_to = d.token_id; }
    }
  }
}

async function runInterventions() {
  if (!interventions.length) return;
  $("iv-run").disabled = true;
  const out = $("iv-output");
  out.classList.remove("hidden");
  out.innerHTML = "<div class='col'><h4>working\u2026</h4></div>";
  statusEl.textContent = "generating with edits\u2026"; statusEl.className = "status";
  try {
    await resolveSwapTargets();
    const edits = interventions.map((iv) => ({
      layer: iv.layer, token_id: iv.token_id, mode: iv.mode,
      strength: iv.strength, swap_to: iv.swap_to ?? null,
    }));
    const body = { interventions: edits, include_baseline: true,
      max_new_tokens: sampling().max_new_tokens, temperature: sampling().temperature };
    if (mode === "chat" && conversation.length) body.messages = conversation;
    else body.prompt = $("prompt").value, body.use_chat_template = $("chat").checked;
    const r = await fetch("/api/intervene", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();
    out.innerHTML =
      `<div class="col"><h4>baseline (no edits)</h4>${escapeHtml(d.baseline || "")}</div>` +
      `<div class="col steered"><h4>steered</h4>${escapeHtml(d.steered || "")}</div>`;
    statusEl.textContent = "done"; statusEl.className = "status ok";
  } catch (e) {
    out.innerHTML = `<div class='col'><h4>error</h4>${escapeHtml(e.message)}</div>`;
    statusEl.textContent = "intervene error"; statusEl.className = "status err";
  } finally { $("iv-run").disabled = false; }
}

$("iv-run").addEventListener("click", runInterventions);
$("iv-clear").addEventListener("click", () => { interventions = []; renderInterventions(); $("iv-output").classList.add("hidden"); });
renderInterventions();

// ===================== Streaming generation + live J-space =====================
let streamAbort = null;
// Tokens the model uses to signal end-of-turn; hidden from the visible text.
const STOP_MARKERS = new Set(["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end|>"]);

// Build a grid that starts with the layer rows but zero token columns; columns
// are appended live as tokens arrive.
function initStreamGrid(layers) {
  currentTokens = [];
  shownTokens = new Set();
  const table = $("grid");
  table.innerHTML = "";
  const thead = document.createElement("thead");
  const hr = document.createElement("tr"); hr.id = "grid-head-row";
  hr.appendChild(th("layer \\ generated token"));
  thead.appendChild(hr); table.appendChild(thead);
  const tb = document.createElement("tbody"); tb.id = "grid-body";
  layers.slice().reverse().forEach((ell) => {
    const tr = document.createElement("tr"); tr.id = `row-${ell}`;
    const rh = th(`L${ell}`); rh.scope = "row"; tr.appendChild(rh);
    tb.appendChild(tr);
  });
  table.appendChild(tb);
  $("legend").classList.remove("hidden");
  return layers;
}

function appendStreamColumn(idx, tokenText, cells) {
  currentTokens[idx] = tokenText;
  $("grid-head-row").appendChild(th(fmtTok(tokenText), `col-${idx}`));
  cells.forEach((c) => {
    const row = $(`row-${c.layer}`);
    if (!row) return;
    const td = document.createElement("td");
    td.id = `cell-${c.layer}-${idx}`;
    const best = c.top && c.top[0];
    if (best) {
      td.dataset.token = best.token;
      td.dataset.score = String(best.score);
      shownTokens.add(best.token);
      paintCell(td, best);
      td.title = c.top.map((t) => `${JSON.stringify(t.token)}  ${t.score.toFixed(3)}`).join("\n");
      td.style.background = `rgba(110,168,254,${Math.min(0.55, best.score * 0.6).toFixed(3)})`;
      td.onclick = () => decompose(c.layer, idx);
    } else { td.textContent = ""; }
    row.appendChild(td);
  });
}

async function streamGenerate() {
  const out = $("stream-out");
  out.classList.remove("hidden");
  const promptText = (mode === "chat") ? "" : $("prompt").value;
  out.innerHTML = `<span class="prompt-text">${escapeHtml(promptText)}</span>` +
    `<span class="gen-text"></span><span class="cursor">\u258b</span>`;
  const genEl = out.querySelector(".gen-text");
  const method = $("method").value;
  currentMethod = method;
  $("stream-run").disabled = true;
  $("stream-stop").classList.remove("hidden");
  statusEl.textContent = "generating\u2026"; statusEl.className = "status";

  const body = {
    method,
    layers: parseLayers(),
    top_k: parseInt($("topk").value, 10) || 6,
    max_new_tokens: parseInt($("gen-max").value, 10) || 60,
    temperature: 0.0,
  };
  if (mode === "chat" && conversation.length) body.messages = conversation;
  else { body.prompt = $("prompt").value; body.use_chat_template = $("chat").checked; }

  streamAbort = new AbortController();
  try {
    const resp = await fetch("/api/generate/stream", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body), signal: streamAbort.signal,
    });
    if (!resp.ok || !resp.body) throw new Error(await resp.text());
    let genText = "";
    await readSSE(resp, (ev) => {
      if (ev.type === "status") { statusEl.textContent = ev.detail; }
      else if (ev.type === "start") { initStreamGrid(ev.layers); }
      else if (ev.type === "token") {
        if (STOP_MARKERS.has(ev.token)) return;   // hide end-of-turn markers
        genText += ev.token;
        genEl.innerHTML = escapeHtml(genText.slice(0, -ev.token.length)) +
          `<span class="fresh">${escapeHtml(ev.token)}</span>`;
        out.scrollTop = out.scrollHeight;
        appendStreamColumn(ev.index, ev.token, ev.cells);
        if ($("translate").checked) scheduleStreamTranslate();
      } else if (ev.type === "done") {
        genEl.textContent = genText;
        const c = out.querySelector(".cursor"); if (c) c.remove();
        statusEl.textContent = "done"; statusEl.className = "status ok";
        if ($("translate").checked) applyTranslations();
      } else if (ev.type === "error") {
        statusEl.textContent = "stream error: " + ev.detail; statusEl.className = "status err";
      }
    });
  } catch (e) {
    if (e.name === "AbortError") { statusEl.textContent = "stopped"; statusEl.className = "status"; }
    else {
      const msg = (e instanceof TypeError)
        ? "connection dropped (is the server still running? check the terminal)"
        : e.message;
      statusEl.textContent = "stream error: " + msg; statusEl.className = "status err";
    }
    const c = out.querySelector(".cursor"); if (c) c.remove();
  } finally {
    $("stream-run").disabled = false;
    $("stream-stop").classList.add("hidden");
    streamAbort = null;
  }
}

// Debounced translation while streaming so we don't hammer the endpoint.
let streamTrTimer = null;
function scheduleStreamTranslate() {
  if (streamTrTimer) clearTimeout(streamTrTimer);
  streamTrTimer = setTimeout(() => applyTranslations(), 400);
}

$("stream-run").addEventListener("click", streamGenerate);
$("stream-stop").addEventListener("click", () => { if (streamAbort) streamAbort.abort(); });
