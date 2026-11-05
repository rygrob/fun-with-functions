// T_n(cos θ) = cos(nθ), drawn by a machine: a circle above the plot carries
// angle θ and drops x = cos θ straight down; a circle beside it carries nθ,
// turned a quarter so cos nθ reads as height, and projects it across. Where
// the two guides cross is (cos θ, cos nθ) = (x, T_n(x)). Sweeping θ from 0
// to π traces T_n; for n ≥ 2 the nθ point laps its circle.
// T_1 up to T_5 (T_0, where the nθ point stays put and the pen draws 1,
// is left out: it doesn't animate well).
//
// One parameter so a single tween draws one polynomial:
//   t                 0..5; n = ⌈t⌉ (1 at t = 0), θ = π (t − n + 1)
// Curves for smaller n stay drawn, faint.
//
// Pure geometry: no data. <div class="figure" data-figure="chebcircle"></div>

import { el, set, svg, pathD } from "../svg.js";

const PI = Math.PI;
const N_MAX = 5;

export default {
  defaults: { t: 0 },
  sticky: [],
  scrub: ["t"],
  tweenMs: 2600,

  create(root) {
    const R = +root.dataset.unit || 112;
    const gap = 34;
    const W = 4 * R + gap + 40, H = 4 * R + gap + 40;
    const s = svg(root, W, H);
    const pcx = W - R - 20, pcy = H - R - 20;          // plot centre
    const acx = pcx, acy = pcy - 2 * R - gap;          // θ circle, above
    const bcx = pcx - 2 * R - gap, bcy = pcy;          // nθ circle, left
    const px = (x) => pcx + R * x, py = (y) => pcy - R * y;

    // Plot frame.
    el("rect", { class: "cheb-frame", x: px(-1), y: py(1), width: 2 * R, height: 2 * R }, s);
    el("line", { class: "axis", x1: px(-1), x2: px(1), y1: py(0), y2: py(0) }, s);
    el("line", { class: "axis", x1: px(0), x2: px(0), y1: py(-1), y2: py(1) }, s);
    for (const [v, lab] of [[-1, "−1"], [1, "1"]]) {
      el("text", { class: "tick", x: px(v), y: py(-1) + 28, text: lab }, s);
      el("text", { class: "tick end", x: px(-1) - 12, y: py(v) + (v < 0 ? 0 : 6), text: lab }, s);
    }
    el("text", { class: "axis-name", x: px(1) + 8, y: py(0) + 6, text: "x" }, s);

    // Circles.
    for (const [cx, cy] of [[acx, acy], [bcx, bcy]]) {
      el("circle", { class: "unit-circle", cx, cy, r: R }, s);
      el("circle", { class: "hub", cx, cy, r: 3 }, s);
    }
    el("line", { class: "ref", x1: acx, x2: acx + R, y1: acy, y2: acy }, s);
    el("line", { class: "ref", x1: bcx, x2: bcx, y1: bcy, y2: bcy - R }, s);
    const arcA = el("path", { class: "angle-arc theta" }, s);
    const labA = el("text", { class: "angle-label theta", text: "θ" }, s);
    const labB = el("text", { class: "angle-label ntheta", x: bcx - R, y: bcy - R - 10 }, s);

    // Traces: finished curves faint, the current one solid.
    const done = Array.from({ length: N_MAX - 1 }, () => el("path", { class: "cheb-done" }, s));
    const trace = el("path", { class: "cheb-trace" }, s);

    // Moving parts.
    const radA = el("line", { class: "radius theta" }, s);
    const radB = el("line", { class: "radius ntheta" }, s);
    const guideA = el("line", { class: "guide-line theta" }, s);
    const guideB = el("line", { class: "guide-line ntheta" }, s);
    const dotA = el("circle", { class: "dot theta", r: 7 }, s);
    const dotB = el("circle", { class: "dot ntheta", r: 7 }, s);
    const dotP = el("circle", { class: "dot pen", r: 7 }, s);

    const curve = (n, th) => {
      const pts = [];
      const steps = Math.max(2, Math.ceil(200 * th / PI));
      for (let i = 0; i <= steps; i++) {
        const a = (th * i) / steps;
        pts.push([px(Math.cos(a)), py(Math.cos(n * a))]);
      }
      return pathD(pts);
    };

    return {
      render({ t }) {
        t = Math.min(N_MAX, Math.max(0, t));
        const n = Math.max(1, Math.ceil(t - 1e-9));
        const th = PI * (t - n + 1);
        const x = Math.cos(th), y = Math.cos(n * th);

        done.forEach((d, k) => set(d, { d: k + 1 < n ? curve(k + 1, PI) : "" }));
        set(trace, { d: curve(n, th) });

        const ax = acx + R * Math.cos(th), ay = acy - R * Math.sin(th);
        const bx = bcx - R * Math.sin(n * th), by = bcy - R * Math.cos(n * th);
        set(radA, { x1: acx, y1: acy, x2: ax, y2: ay });
        set(radB, { x1: bcx, y1: bcy, x2: bx, y2: by });
        set(dotA, { cx: ax, cy: ay });
        set(dotB, { cx: bx, cy: by });
        set(guideA, { x1: ax, y1: ay, x2: px(x), y2: py(y) });
        set(guideB, { x1: bx, y1: by, x2: px(x), y2: py(y) });
        set(dotP, { cx: px(x), cy: py(y) });

        // θ's arc, from the reference ray (pointing right) round to θ.
        const r = 30;
        const large = th > PI ? 1 : 0;
        set(arcA, { d: th > 0.01 ? `M${acx + r},${acy} A${r},${r} 0 ${large} 0 ${acx + r * Math.cos(th)},${acy - r * Math.sin(th)}` : "" });
        const la = Math.max(th, 0.7) / 2;  // off the radius line at small angles
        set(labA, { x: acx + 48 * Math.cos(la) - 6, y: acy - 48 * Math.sin(la) + 8 });
        set(labB, { text: t === 0 ? "nθ" : n === 1 ? "θ" : `${n}θ` });  // at rest, any n
      },
    };
  },
};
