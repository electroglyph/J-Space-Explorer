"use strict";
// A tiny dependency-free guided tour: spotlight + popover over real UI
// elements. Each step can run a `before()` action (switch mode, seed a demo
// decomposition, etc.) so the user sees every part in context.
(function () {
  const $ = (id) => document.getElementById(id);
  const overlay = $("tour-overlay");
  const spot = $("tour-spotlight");
  const pop = $("tour-pop");

  // --- Demo helpers: populate parts of the UI so a step has something to show,
  //     without requiring the (slow) model. These build fake DOM only.
  function demoGrid() {
    if (document.querySelector("#grid tbody td")) return; // real data present
    const tokens = ["The", " capital", " of", " France", " is"];
    const layers = [32, 30, 28, 24, 16, 8, 0];
    const demo = {
      32: "Paris", 30: "Paris", 28: "Paris", 24: "Paris",
      16: "France", 8: "____", 0: "lez",
    };
    const table = $("grid");
    table.innerHTML = "";
    const thead = document.createElement("thead");
    const hr = document.createElement("tr");
    const c = document.createElement("th"); c.textContent = "layer \\ token"; hr.appendChild(c);
    tokens.forEach((t) => { const th = document.createElement("th"); th.textContent = t; hr.appendChild(th); });
    thead.appendChild(hr); table.appendChild(thead);
    const tb = document.createElement("tbody"); tb.id = "grid-body";
    layers.forEach((L) => {
      const tr = document.createElement("tr"); tr.id = `row-${L}`;
      const rh = document.createElement("th"); rh.textContent = `L${L}`; tr.appendChild(rh);
      tokens.forEach((_, i) => {
        const td = document.createElement("td"); td.id = `cell-${L}-${i}`;
        const strong = (i === 4 && L >= 24) ? 0.95 : (L <= 8 ? 0.05 : 0.4);
        const word = (i === 4) ? demo[L] : (L <= 8 ? "\u00b7" : demo[L]);
        if (word && word !== "\u00b7") {
          td.innerHTML = `<span class="cell-top">${word}</span> <span class="cell-score">${strong.toFixed(2)}</span>`;
          td.style.background = `rgba(110,168,254,${Math.min(0.55, strong * 0.6).toFixed(3)})`;
        } else { td.textContent = "\u00b7"; }
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
    table.appendChild(tb);
    $("legend").classList.remove("hidden");
  }

  function demoPanel() {
    $("panel").classList.remove("hidden");
    $("panel-title").textContent = "J-space @ layer 24, token \u2018 is\u2019";
    $("panel-body").innerHTML =
      `<div class="meta">token: <b>\u2009is</b><br>J-space variance captured: <b>5.2%</b><br>residual norm: 97% of activation</div>` +
      [["Paris", 15.8], ["located (\u4f4d\u4e8e)", 5.5], ["city (\u57ce\u5e02)", 4.9], ["French", 4.2]]
        .map(([tok, coef], i) => {
          const w = Math.max(4, (coef / 15.8) * 100);
          return `<div class="atom"><span class="tok">${tok}</span><span class="coef">${coef.toFixed(2)}</span>` +
            `<span class="acts"><button>steer</button><button>swap</button><button>ablate</button></span></div>` +
            `<div class="bar"><span style="width:${w}%"></span></div>`;
        }).join("");
  }

  function demoIntervention() {
    const bar = $("intervention-bar");
    bar.classList.remove("hidden");
    $("iv-list").innerHTML =
      `<div class="iv-chip steer"><span class="tag">steer</span><span class="tok">\u2009ocean</span> <span class="muted">L24</span>` +
      `<input type="range" min="-8" max="8" step="0.5" value="3"/><span class="muted">3</span><span class="rm">\u00d7</span></div>` +
      `<div class="iv-chip swap"><span class="tag">swap</span><span class="tok">\u2009soccer</span> \u2192 <span class="tok">rugby</span><span class="rm">\u00d7</span></div>`;
    $("iv-output").classList.remove("hidden");
    $("iv-output").innerHTML =
      `<div class="col"><h4>baseline (no edits)</h4>My favorite sport to watch is basketball \u2014 the fast pace and teamwork.</div>` +
      `<div class="col steered"><h4>steered</h4>My favorite sport to watch is the ocean \u2014 the waves rolling in\u2026</div>`;
  }

  // --- Tour steps -----------------------------------------------------------
  const steps = [
    {
      el: () => document.querySelector("header h1"),
      title: "Welcome to the J-Space Explorer",
      html: `This tool opens up a language model's <b>\u201cJ-space\u201d</b> \u2014 the small,
        privileged set of concepts it is <b>poised to say</b> at each moment, described in Anthropic's
        <i>\u201cVerbalizable Representations Form a Global Workspace\u201d</i>. You'll type text, watch what the
        model is \u201cthinking\u201d layer by layer, and even edit those thoughts. Let's walk through it.`,
    },
    {
      el: () => $("prompt"),
      before: () => setModeSafe("completion"),
      title: "1. Give the model some text",
      html: `In <b>Completion</b> mode you type a prompt and the model reads it. Here we ask for the
        capital of France. The lens will show what concepts light up internally as it prepares to answer.`,
    },
    {
      el: () => $("method"),
      title: "2. Pick a lens",
      html: `A <b>lens</b> reads an internal activation and decodes it into vocabulary words.<br><br>
        \u2022 <b>Logit lens</b> \u2014 instant; just decodes each layer directly.<br>
        \u2022 <b>Jacobian lens (J-lens)</b> \u2014 the real method from the paper. It first \u201cfits\u201d a small
        transport matrix over a corpus (one-time), then reveals <i>verbalizable</i> concepts even in early
        layers where the logit lens shows noise.`,
    },
    {
      el: () => $("layers"),
      title: "3. Choose layers",
      html: `Leave this blank to see <b>all 32 layers</b>, or list a few like <code>0,8,16,24,32</code>.
        Layer 0 is the raw input; the last layer is the output. The interesting \u201cworkspace\u201d lives in
        the middle.`,
    },
    {
      el: () => $("translate"),
      title: "4. Translate to English",
      html: `The J-lens often surfaces non-English tokens (e.g. <code>\u5df4\u9ece</code> for Paris). With this on,
        each token is glossed to English inline \u2014 <code>\u5df4\u9ece</code> becomes <b>Paris (\u5df4\u9ece)</b> \u2014 using the
        model itself as the translator.`,
    },
    {
      el: () => $("run"),
      title: "5. Run the lens",
      html: `Click <b>Run lens</b> to fill the grid. (For the Jacobian lens the first run fits the operator
        over a small corpus \u2014 a one-time step, cached to disk, with a progress bar.)`,
    },
    {
      el: () => $("stream-run"),
      title: "5b. Stream + lens",
      html: `Or click <b>Stream + lens \u25b6</b> to <i>generate</i> text token by token. Each new token streams
        into the box above <b>and</b> adds a live column to the grid showing its J-space \u2014 so you watch the
        model's concepts unfold as it writes. It stops at the model's EOS, and <b>Stop</b> interrupts it.`,
    },
    {
      el: () => { demoGrid(); return document.querySelector("#grid"); },
      title: "6. Read the grid",
      html: `Each <b>column is a token</b> of your text; each <b>row is a layer</b> (output on top, input at
        the bottom). Every cell shows the top concept the model is poised to verbalize <i>there</i>, shaded by
        confidence. Watch a concept <b>emerge across layers</b> \u2014 here \u201cParis\u201d sharpens from layer 16
        up to near-certainty by layer 24.`,
    },
    {
      el: () => { demoGrid(); return $("cell-24-4"); },
      title: "7. Click a cell to decompose it",
      html: `A single activation is a blend of many concepts. Click any cell to <b>decompose</b> it into its
        <b>J-space atoms</b> \u2014 the individual verbalizable concepts inside it, with weights.`,
    },
    {
      el: () => { demoPanel(); return $("panel"); },
      before: () => demoPanel(),
      title: "8. The J-space decomposition",
      html: `This panel lists the concepts active in that activation (here <b>Paris</b>, located, city, French)
        and how much of the activation lives in the J-space \u2014 usually under 10%, matching the paper's finding
        that the workspace is a small, selective slice of the model's processing.`,
    },
    {
      el: () => { demoPanel(); return document.querySelector("#panel-body .acts"); },
      before: () => demoPanel(),
      title: "9. Edit the model's mind",
      html: `Each atom has three actions:<br><br>
        \u2022 <b>steer</b> \u2014 amplify or (with a negative value) suppress that concept.<br>
        \u2022 <b>ablate</b> \u2014 remove the concept entirely.<br>
        \u2022 <b>swap</b> \u2014 replace it with another word (e.g. <i>soccer \u2192 rugby</i>).`,
    },
    {
      el: () => { demoIntervention(); return $("intervention-bar"); },
      before: () => demoIntervention(),
      title: "10. J-space edits + steered generation",
      html: `Your edits collect here. The <b>slider is signed</b> (\u22128\u2026+8): positive injects a concept,
        negative suppresses it. Hit <b>Generate with edits</b> and you'll see the <b>baseline</b> vs the
        <b>steered</b> output side by side \u2014 proof that these concepts causally drive what the model says.`,
    },
    {
      el: () => $("mode-chat"),
      before: () => { $("panel").classList.add("hidden"); },
      title: "11. Chat mode",
      html: `Switch to <b>Chat</b> to hold a real conversation. You get temperature / top-p / top-k / min-p
        controls, and \u2014 crucially \u2014 a <b>\u201cLens on last reply\u201d</b> button that runs the lens on the
        conversation <i>at the moment the model is about to speak</i>, revealing its unspoken reasoning.`,
    },
    {
      el: () => $("tour-start"),
      before: () => setModeSafe("completion"),
      title: "That's the tour!",
      html: `You can replay this anytime with <b>\u2728 Tour</b>. Now try it for real: type a prompt, run the
        Jacobian lens across all layers, click a cell, and steer a concept. Have fun exploring the model's
        workspace!`,
    },
  ];

  let idx = 0;

  function setModeSafe(m) {
    const btn = m === "chat" ? $("mode-chat") : $("mode-completion");
    if (btn && !btn.classList.contains("active")) btn.click();
  }

  function place() {
    const step = steps[idx];
    if (step.before) { try { step.before(); } catch (e) {} }
    const el = step.el && step.el();
    $("tour-step").textContent = `${idx + 1} / ${steps.length}`;
    $("tour-title").textContent = step.title;
    $("tour-text").innerHTML = step.html;
    $("tour-back").style.visibility = idx === 0 ? "hidden" : "visible";
    $("tour-next").textContent = idx === steps.length - 1 ? "Finish" : "Next";

    if (el) {
      el.scrollIntoView({ block: "center", behavior: "smooth" });
      const r = el.getBoundingClientRect();
      const pad = 6;
      spot.style.display = "block";
      spot.style.top = `${r.top - pad}px`;
      spot.style.left = `${r.left - pad}px`;
      spot.style.width = `${r.width + pad * 2}px`;
      spot.style.height = `${r.height + pad * 2}px`;
      // Position the popover: below the element if room, else above.
      const popW = Math.min(360, window.innerWidth * 0.9);
      let top = r.bottom + 14;
      const popH = 220;
      if (top + popH > window.innerHeight) top = Math.max(14, r.top - popH - 14);
      let left = Math.min(Math.max(14, r.left), window.innerWidth - popW - 14);
      pop.style.top = `${top}px`;
      pop.style.left = `${left}px`;
    } else {
      // No element: center the popover, hide spotlight.
      spot.style.display = "none";
      pop.style.top = "50%"; pop.style.left = "50%";
      pop.style.transform = "translate(-50%,-50%)";
      return;
    }
    pop.style.transform = "none";
  }

  // --- Make the popover draggable so it never blocks what it points at. -----
  (function enableDrag() {
    let dragging = false, offX = 0, offY = 0;
    function moveTo(cx, cy) {
      const w = pop.offsetWidth, h = pop.offsetHeight;
      const left = Math.min(Math.max(4, cx - offX), window.innerWidth - w - 4);
      const top = Math.min(Math.max(4, cy - offY), window.innerHeight - h - 4);
      pop.style.left = `${left}px`;
      pop.style.top = `${top}px`;
    }
    function onMove(e) {
      if (!dragging) return;
      const p = e.touches ? e.touches[0] : e;
      moveTo(p.clientX, p.clientY);
      e.preventDefault();
    }
    function onUp() {
      if (!dragging) return;
      dragging = false;
      pop.style.cursor = "";
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.removeEventListener("touchmove", onMove);
      document.removeEventListener("touchend", onUp);
    }
    function onDown(e) {
      // Don't start a drag from an interactive control.
      if (e.target.closest("button, a, input, select, textarea")) return;
      const p = e.touches ? e.touches[0] : e;
      dragging = true;
      const r = pop.getBoundingClientRect();
      pop.style.transform = "none";  // in case it was centered via transform
      pop.style.top = `${r.top}px`;
      pop.style.left = `${r.left}px`;
      offX = p.clientX - r.left;
      offY = p.clientY - r.top;
      pop.style.cursor = "grabbing";
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
      document.addEventListener("touchmove", onMove, { passive: false });
      document.addEventListener("touchend", onUp);
      e.preventDefault();
    }
    pop.addEventListener("mousedown", onDown);
    pop.addEventListener("touchstart", onDown, { passive: false });
  })();

  function start() { idx = 0; overlay.classList.remove("hidden"); place(); }
  function end() {
    overlay.classList.add("hidden");
    try { localStorage.setItem("jspace_tour_seen", "1"); } catch (e) {}
    // Clear demo content the tour injected. A real lens run marks its cells
    // with data-token; if none exist, everything on screen is demo scaffolding.
    const hasReal = !!document.querySelector("#grid tbody td[data-token]");
    if (!hasReal) {
      $("grid").innerHTML = "";
      $("legend").classList.add("hidden");
      $("panel").classList.add("hidden");
      $("panel-body").innerHTML = "";
      $("iv-list").innerHTML = "";
      $("iv-output").classList.add("hidden");
      $("iv-output").innerHTML = "";
      $("intervention-bar").classList.add("hidden");
    }
  }
  function next() { if (idx < steps.length - 1) { idx++; place(); } else end(); }
  function back() { if (idx > 0) { idx--; place(); } }

  $("tour-start").addEventListener("click", start);
  $("tour-next").addEventListener("click", next);
  $("tour-back").addEventListener("click", back);
  $("tour-skip").addEventListener("click", end);
  window.addEventListener("resize", () => { if (!overlay.classList.contains("hidden")) place(); });
  document.addEventListener("keydown", (e) => {
    if (overlay.classList.contains("hidden")) return;
    if (e.key === "ArrowRight" || e.key === "Enter") next();
    else if (e.key === "ArrowLeft") back();
    else if (e.key === "Escape") end();
  });

  // Auto-start once for first-time visitors, after the page settles.
  try {
    if (!localStorage.getItem("jspace_tour_seen")) {
      setTimeout(start, 900);
    }
  } catch (e) {}
})();
