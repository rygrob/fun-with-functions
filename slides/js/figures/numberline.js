// The float number line: every value of a toy float format as a tick, so
// the gap visibly doubles at each power of 2.
//
// Data (generate.py "float-line"):
//   values            every representable value, ascending
//   binades           [{ lo, hi, ulp (TeX) }], labelled under the line
//
// <div class="figure" data-figure="numberline" data-src="float-line"></div>

import { el, svg } from "../svg.js";
import { tex } from "../tex.js";

export default {
  defaults: {},
  sticky: [],
  scrub: [],

  create(root, data) {
    const W = +root.dataset.width || 1100;
    const H = +root.dataset.height || 150;
    const s = svg(root, W, H);
    // Log-free on purpose: a linear axis shows the gaps growing.
    const lo = 0, hi = data.values[data.values.length - 1];
    const sx = (v) => 24 + ((v - lo) / (hi - lo)) * (W - 48);
    const y = 50;
    el("line", { class: "axis", x1: sx(lo), x2: sx(hi), y1: y, y2: y }, s);
    for (const v of data.values) {
      el("line", { class: "float-tick", x1: sx(v), x2: sx(v), y1: y - 14, y2: y + 14 }, s);
    }
    for (const v of [0, ...data.binades.map((b) => b.hi)]) {
      el("text", { class: "tick", x: sx(v), y: y - 24, text: String(v) }, s);
    }
    // Binade labels: the gap in each, centred under its span (TeX via
    // foreignObject so the exponent sets properly).
    for (const b of data.binades) {
      const w = sx(b.hi) - sx(b.lo);
      el("line", { class: "binade", x1: sx(b.lo) + 3, x2: sx(b.hi) - 3, y1: y + 30, y2: y + 30 }, s);
      const fo = el("foreignObject", { x: sx(b.lo), y: y + 38, width: w, height: 50 }, s);
      const div = document.createElement("div");
      div.className = "binade-label";
      fo.appendChild(div);
      tex(div, b.ulp);
    }
    return { render() {} };
  },
};
