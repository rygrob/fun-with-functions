// Equal angles, projected: N points equally spaced in angle around the
// upper half circle drop straight down to the x axis, x_k = cos θ_k, and
// land bunched toward the ends. Unnamed here: the Chebyshev nodes slide
// names them. The equally spaced points stay underneath, faint, for
// comparison.
//
// Parameters:
//   lift              0..1: the half circle and its equal-angle points
//                     appear; the equally spaced points fade to a ghost row,
//                     and the axis relabels from [−π, π] to [−1, 1]
//   drop              0..1: the points fall to the axis
//
// θ_k = (k + ½)π / N, the same nodes as Move the points (example 15).
// Pure geometry: no data. data-n sets N (default 11).
// <div class="figure" data-figure="angles" data-n="11"></div>

import { el, set, svg, pool, lerp, clamp01 } from "../svg.js";

const PI = Math.PI;

export default {
  defaults: { lift: 0, drop: 0 },
  sticky: [],
  scrub: ["lift", "drop"],

  create(root) {
    const N = +root.dataset.n || 11;
    const W = +root.dataset.width || 1150, H = +root.dataset.height || 470;
    const R = 330, cx = W / 2, cy = H - 90;
    const s = svg(root, W, H);
    const X = (x) => cx + R * x;

    // The x axis, and the row the equally spaced points sit on below it.
    const ghostY = cy + 44;
    el("line", { class: "axis", x1: X(-1.08), x2: X(1.08), y1: cy, y2: cy }, s);
    // Tick labels start as [−π, π], like the Equally spaced points slide, and
    // switch to [−1, 1] as the equally spaced points step down out of the way.
    const ticksPi = el("g", {}, s), ticksOne = el("g", {}, s);
    for (const [v, pi, one] of [[-1, "−π", "−1"], [0, "0", "0"], [1, "π", "1"]]) {
      el("line", { class: "axis", x1: X(v), x2: X(v), y1: cy - 6, y2: cy + 6 }, s);
      if (v === 0) {
        el("text", { class: "tick", x: X(v), y: cy + 28, text: "0" }, s);
        continue;
      }
      el("text", { class: "tick", x: X(v), y: cy + 28, text: pi }, ticksPi);
      el("text", { class: "tick", x: X(v), y: cy + 28, text: one }, ticksOne);
    }
    el("text", { class: "axis-name", x: X(1.08) + 8, y: cy + 6, text: "x" }, s);
    const equi = Array.from({ length: N }, (_, k) => -1 + (2 * k) / (N - 1));
    const ghostLine = el("line", { class: "grid", x1: X(-1), x2: X(1), y1: ghostY, y2: ghostY }, s);
    const ghostLab = el("text", { class: "pane-tag", x: X(-1.08), y: ghostY + 6, "text-anchor": "end", text: "equally spaced" }, s);
    const equiDots = pool(el("g", {}, s), "circle", { r: 6.5, class: "node-dot" })(N);

    // The half circle, its spokes, the falling points.
    const arc = el("path", { class: "unit-circle", d: `M${X(1)},${cy} A${R},${R} 0 0 0 ${X(-1)},${cy}` }, s);
    const th = Array.from({ length: N }, (_, k) => ((k + 0.5) * PI) / N);
    const spokes = pool(el("g", {}, s), "line", { class: "spoke" })(N);
    const drops = pool(el("g", {}, s), "line", { class: "drop" })(N);
    const dots = pool(el("g", {}, s), "circle", { r: 7, class: "node-dot" })(N);

    return {
      render({ lift, drop }) {
        lift = clamp01(lift);
        drop = clamp01(drop);
        set(ticksPi, { opacity: 1 - lift });
        set(ticksOne, { opacity: lift });
        set(arc, { opacity: lift });
        // Equally spaced: on the axis at first, then down to the ghost row.
        const ey = lerp(cy, ghostY, lift);
        equiDots.forEach((d, k) => set(d, { cx: X(equi[k]), cy: ey, opacity: lerp(1, 0.3, lift) }));
        set(ghostLine, { opacity: lift * 0.8 });
        set(ghostLab, { opacity: lift * 0.8 });
        th.forEach((t, k) => {
          const px = X(Math.cos(t)), py = cy - R * Math.sin(t);
          const y = lerp(py, cy, drop);
          set(spokes[k], { x1: cx, y1: cy, x2: px, y2: py, opacity: lift * (1 - 0.6 * drop) });
          set(drops[k], { x1: px, y1: py, x2: px, y2: y, opacity: drop > 0 ? lift : 0 });
          set(dots[k], { cx: px, cy: y, opacity: lift });
        });
      },
    };
  },
};
