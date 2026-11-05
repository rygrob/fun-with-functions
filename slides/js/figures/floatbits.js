// A float's 32 bits, field by field, over a number line. Two figures:
//
// floatbits (generate.py "float-anatomy"): each step is a value; its bits,
//   the decoded (-1)^s 2^e 1.f, and a marker on an axis where every binade
//   has the same width, so the exponent picks the binade and the fraction
//   the place within it. The marker slides between steps.
//
// floatzoom (generate.py "float-pi-e"): each step is a constant (π, e); the
//   bits of the float it rounds to, and a close-up of the number line: four
//   neighbouring floats, the constant between two of them, and an arrow to
//   the one it rounds to.
//
// floatfields (generate.py "float-440"): one value, its bits and decoded
//   value, no number line; a field can be lit and the others dimmed.
//
// Parameter: step, 0 .. steps.length − 1 (fragments tween it); floatfields
// takes hl instead: the lit field's class ("sign", "exp", "frac"), the others
// dimmed, or "" for none. A string, so it switches rather than tweening
// through the fields in between; the CSS fades it.
//
// <div class="figure" data-figure="floatbits" data-src="float-anatomy"></div>

import { el, svg, set } from "../svg.js";
import { tex } from "../tex.js";

const FIELDS = [
  { name: "sign", cls: "sign", n: 1, key: "sign_tex" },
  { name: "exponent", cls: "exp", n: 8, key: "exp_tex" },
  { name: "fraction", cls: "frac", n: 23, key: "frac_tex" },
];

// The 32 bit cells in three groups, each with a name and a decoded value.
function bitsRow(root) {
  const row = document.createElement("div");
  row.className = "float-bits";
  const cells = [];
  const decoded = [];
  for (const f of FIELDS) {
    const group = document.createElement("div");
    group.className = `field ${f.cls}`;
    const strip = document.createElement("div");
    strip.className = "cells";
    for (let i = 0; i < f.n; i++) {
      const c = document.createElement("span");
      strip.appendChild(c);
      cells.push(c);
    }
    const name = document.createElement("div");
    name.className = "field-name";
    name.textContent = f.name;
    const val = document.createElement("div");
    val.className = "field-value";
    group.append(strip, name, val);
    decoded.push(val);
    row.appendChild(group);
  }
  root.appendChild(row);
  let shown = null;
  return (step) => {
    if (shown === step) return;
    shown = step;
    [...step.bits].forEach((b, i) => {
      cells[i].textContent = b;
      cells[i].classList.toggle("one", b === "1");
    });
    FIELDS.forEach((f, i) => tex(decoded[i], step[f.key]));
  };
}

function texLine(root, cls) {
  const div = document.createElement("div");
  div.className = cls;
  root.appendChild(div);
  let src = null;
  return (s) => {
    if (s !== src) tex(div, (src = s));
  };
}

// Index and blend for a fractional step.
function at(steps, step) {
  const k = Math.min(steps.length - 1, Math.max(0, step));
  const i = Math.min(steps.length - 2, Math.floor(k));
  return { cur: steps[Math.round(k)], a: steps[Math.max(0, i)], b: steps[Math.max(0, i) + 1] || steps[0], u: k - Math.max(0, i) };
}

export default {
  defaults: { step: 0 },
  sticky: [],
  scrub: [],

  create(root, data) {
    const W = +root.dataset.width || 1150;
    const bits = bitsRow(root);
    const eq = texLine(root, "float-eq");
    const ulp = texLine(root, "float-ulp");

    const s = svg(root, W, 120);
    const caption = document.createElement("div");
    caption.className = "float-caption";
    root.appendChild(caption);
    caption.innerHTML = "Every binade holds 2<sup>23</sup> evenly spaced floats";
    const bs = data.binades;
    const lo = bs[0].e, n = bs.length;
    const bw = (W - 80) / n;
    const sx = (pos) => 40 + (pos - lo) * bw;
    const y = 54;
    const bands = bs.map((b) => el("rect", { class: "binade-band", x: sx(b.e) + 2, y: y - 18, width: bw - 4, height: 36, rx: 6 }, s));
    el("line", { class: "axis", x1: sx(lo), x2: sx(lo + n), y1: y, y2: y }, s);
    for (let k = 0; k <= n; k++) {
      el("line", { class: "binade-edge", x1: sx(lo + k), x2: sx(lo + k), y1: y - 22, y2: y + 22 }, s);
      const label = k < n ? bs[k].label : String(2 ** (lo + n));
      el("text", { class: "tick", x: sx(lo + k), y: y + 46, text: label }, s);
    }
    const marker = el("g", {}, s);
    el("line", { class: "float-marker", x1: 0, x2: 0, y1: y - 26, y2: y + 26 }, marker);
    el("circle", { class: "float-dot", cx: 0, cy: y, r: 7 }, marker);
    const mlabel = el("text", { class: "float-label", x: 0, y: y - 34 }, marker);

    return {
      render({ step }) {
        const { cur, a, b, u } = at(data.steps, step);
        bits(cur);
        eq(cur.value_tex);
        ulp(cur.ulp_tex);
        const e = Math.floor(cur.pos);
        bands.forEach((r, k) => r.classList.toggle("on", bs[k].e === e));
        const pos = a.pos + (b.pos - a.pos) * u;
        set(marker, { transform: `translate(${sx(pos).toFixed(2)},0)` });
        set(mlabel, { text: cur.typed.replace("-", "−") });
      },
    };
  },
};


export const floatzoom = {
  defaults: { step: 0 },
  sticky: [],
  scrub: [],

  create(root, data) {
    const W = +root.dataset.width || 1150;
    const bits = bitsRow(root);
    const lines = document.createElement("div");
    lines.className = "float-eq";
    root.appendChild(lines);
    const trueLine = texLine(lines, "line");
    const f32Line = texLine(lines, "line");
    const errLine = texLine(lines, "line");

    const s = svg(root, W, 150);
    const y = 70;
    const xs = [0.12, 0.37, 0.63, 0.88].map((t) => t * W);
    el("line", { class: "axis", x1: 20, x2: W - 20, y1: y, y2: y }, s);
    const ticks = xs.map((x) => el("line", { class: "float-tick", x1: x, x2: x, y1: y - 16, y2: y + 16 }, s));
    const labels = xs.map((x) => el("text", { class: "tick zoom-tick", x, y: y + 46 }, s));
    const arrow = el("line", { class: "round-arrow", y1: y - 34, y2: y - 34, "marker-end": "url(#round-head)" }, s);
    const defs = el("defs", {}, s);
    const head = el("marker", { id: "round-head", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
    el("path", { d: "M0,0 L10,5 L0,10 z", class: "round-head" }, head);
    const marker = el("g", {}, s);
    el("line", { class: "true-marker", x1: 0, x2: 0, y1: y - 22, y2: y + 22 }, marker);
    el("circle", { class: "true-dot", cx: 0, cy: y, r: 7 }, marker);
    const name = el("text", { class: "true-label", x: 0, y: y - 46 }, marker);

    return {
      render({ step }) {
        const { cur, a, b, u } = at(data.steps, step);
        bits(cur);
        trueLine(cur.true_tex);
        f32Line(cur.f32_tex);
        errLine(cur.err_tex);
        cur.ticks.forEach((t, k) => set(labels[k], { text: t }));
        const nearest = cur.rounds === "up" ? 2 : 1;
        ticks.forEach((t, k) => t.classList.toggle("nearest", k === nearest));
        labels.forEach((t, k) => t.classList.toggle("nearest", k === nearest));
        const frac = a.at + (b.at - a.at) * u;
        const x = xs[1] + (xs[2] - xs[1]) * frac;
        set(marker, { transform: `translate(${x.toFixed(2)},0)` });
        set(name, { text: cur.name === "pi" ? "π" : "e" });
        set(arrow, { x1: x, x2: xs[nearest] + (nearest === 2 ? -4 : 4) });
      },
    };
  },
};


export const floatfields = {
  defaults: { hl: "" },
  sticky: [],
  scrub: [],

  create(root, data) {
    bitsRow(root)(data);
    texLine(root, "float-eq")(data.value_tex);
    const groups = [...root.querySelectorAll(".float-bits .field")];
    return {
      render({ hl }) {
        groups.forEach((g, i) => g.classList.toggle("dim", hl !== "" && FIELDS[i].cls !== hl));
      },
    };
  },
};
