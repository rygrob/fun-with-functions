// Rehearsal aids: section time budgets and a clickable contents overlay.
//
// Budgets. A slide with data-section="sin x" data-budget="1200" (seconds)
// starts a section that runs to the next such slide. Before the deck starts,
// each slide in it gets data-timing = budget / slides (unless it has its own),
// which the speaker view (S) uses for its pacing timer: ahead or behind the
// plan, per slide and for the talk.
//
// Contents. T opens an overlay: every slide by section, with the section's
// budget and its slide count; click a slide to go there, T or Esc to close.
// (Built in: G jumps to a slide number, O or Esc is the overview grid,
// Ctrl+Shift+F searches slide text.)

// The sections, in order: { name, budget, slides: [{ index, title, el }] }.
export function sections(root) {
  const out = [];
  [...root.querySelectorAll(".slides > section")].forEach((el, index) => {
    if (el.dataset.section || !out.length) {
      out.push({ name: el.dataset.section || "Start", budget: +el.dataset.budget || 0, slides: [] });
    }
    out[out.length - 1].slides.push({ index, title: title(el), el });
  });
  return out;
}

// Spread each section's budget over its slides as data-timing (seconds).
export function applyBudgets(root) {
  for (const s of sections(root)) {
    if (!s.budget) continue;
    const each = Math.round(s.budget / s.slides.length);
    for (const { el } of s.slides) if (!el.dataset.timing) el.dataset.timing = String(each);
  }
}

// A slide's title as plain text: its heading, TeX rendered as KaTeX's
// annotation (the source), else a class name for the title card.
function title(el) {
  const h = el.querySelector("h1, h2");
  if (!h) return el.classList.contains("title-card") ? "Title card" : "(untitled)";
  const copy = h.cloneNode(true);
  copy.querySelectorAll(".katex").forEach((k) => {
    const src = k.querySelector("annotation");
    k.replaceWith(src ? plainTex(src.textContent) : k.textContent);
  });
  return copy.textContent.replace(/\s+/g, " ").trim();
}

// "\\log_2 x" -> "log₂ x", "\\tan \\pi x" -> "tan π x": enough TeX for a list.
function plainTex(t) {
  return t.replace(/\\pi\b/g, "π").replace(/_2\b/g, "₂").replace(/\^x\b/g, "^x")
    .replace(/\\([a-zA-Z]+)/g, "$1").replace(/[{}]/g, "").replace(/\s+/g, " ").trim();
}

const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;

export function contents(deck, root) {
  const box = document.createElement("div");
  box.className = "contents-overlay";
  box.hidden = true;
  document.body.appendChild(box);

  const build = () => {
    const all = sections(root);
    const total = all.reduce((t, s) => t + s.budget, 0);
    const count = all.reduce((t, s) => t + s.slides.length, 0);
    const here = deck.getIndices().h;
    box.innerHTML = `<header><b>${count} slides</b> · budget ${mmss(total)} · T or Esc to close</header>`;
    const grid = document.createElement("div");
    grid.className = "contents-grid";
    for (const s of all) {
      const col = document.createElement("section");
      const each = s.budget ? ` · ${Math.round(s.budget / s.slides.length)} s each` : "";
      col.innerHTML = `<h3>${s.name}</h3><p>${s.slides.length} slides${s.budget ? ` · ${mmss(s.budget)}` : ""}${each}</p>`;
      const ol = document.createElement("ol");
      for (const sl of s.slides) {
        const li = document.createElement("li");
        li.innerHTML = `<span class="n">${sl.index}</span> `;
        li.append(sl.title);
        if (sl.index === here) li.classList.add("here");
        li.onclick = () => { close(); deck.slide(sl.index); };
        ol.appendChild(li);
      }
      col.appendChild(ol);
      grid.appendChild(col);
    }
    box.appendChild(grid);
  };
  const close = () => { box.hidden = true; };
  const toggle = () => {
    if (box.hidden) { build(); box.hidden = false; } else close();
  };
  document.addEventListener("keydown", (e) => {
    if (!box.hidden && e.key === "Escape") { e.stopPropagation(); close(); }
  }, true);
  return toggle;
}
