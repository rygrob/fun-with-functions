// Move the octave (log2): a number line for m over a plot of log2 m, on the
// same x axis. The octave [1, 2) is shaded on the line and its piece of the
// curve drawn in blue. Its top, [1.5, 2), turns orange; then it halves onto
// [0.75, 1), on the line and on the curve, where log2 of it is exactly 1
// lower: the e + 1. What's left to approximate is log2 m on [0.75, 1.5),
// through zero at 1 instead of starting there.
//
// Data (generate.py "log2-fold-line"):
//   m, y              samples of log2 m on xlim
//   lo, fold, hi      the octave [lo, hi) and where it folds
//   xlim, ylim, xticks, yticks
//
// Parameters (each 0..1):
//   mark              [fold, hi) turns orange
//   fold              it moves to [fold/2, hi/2): ends halve, the curve drops 1
//
// <div class="figure" data-figure="foldline" data-src="log2-fold-line"></div>

import { el, set, svg, linear, pathD } from "../svg.js";

const M = { left: 48, right: 16 };
const LINE_Y = 46;     // the number line
const PLOT_TOP = 120;  // the plot under it
const BAR = 18;        // shaded bar height on the line

const lerp = (a, b, u) => a + (b - a) * u;

export default {
  defaults: { mark: 0, fold: 0 },
  sticky: [],
  scrub: [],

  create(root, d) {
    const W = +root.dataset.width || 680;
    const H = +root.dataset.height || 470;
    const s = svg(root, W, H);
    const sx = linear(d.xlim[0], d.xlim[1], M.left, W - M.right);
    const sy = linear(d.ylim[0], d.ylim[1], H - 40, PLOT_TOP);
    const [x0, x1, y0, y1] = [sx(d.xlim[0]), sx(d.xlim[1]), sy(d.ylim[0]), sy(d.ylim[1])];

    // Number line: axis, ticks, the name m.
    el("line", { class: "axis", x1: x0, x2: x1, y1: LINE_Y, y2: LINE_Y }, s);
    for (const t of d.xticks) {
      el("line", { class: "axis", x1: sx(t.v), x2: sx(t.v), y1: LINE_Y - 6, y2: LINE_Y + 6 }, s);
      el("text", { class: "tick", x: sx(t.v), y: LINE_Y + 32, text: t.label }, s);
    }
    el("text", { class: "axis-name", x: x0 - 30, y: LINE_Y + 7, text: "m" }, s);
    const bar = (cls) => el("rect", { class: `fold-bar ${cls}`, y: LINE_Y - BAR / 2, height: BAR, rx: 4 }, s);
    const keep = bar("keep");    // [lo, fold): stays
    const topBlue = bar("keep"); // [fold, hi) before it's marked
    const moving = bar("move");  // [fold, hi) marked, then moving
    const place = (r, a, b) => set(r, { x: sx(a), width: Math.max(0, sx(b) - sx(a)) });
    place(keep, d.lo, d.fold);
    place(topBlue, d.fold, d.hi);

    // Plot: grid, zero line, ticks, the full curve faint.
    for (const v of d.yticks) {
      el("line", { class: v === 0 ? "axis" : "grid", x1: x0, x2: x1, y1: sy(v), y2: sy(v) }, s);
      el("text", { class: "tick end", x: x0 - 10, y: sy(v) + 6, text: String(v).replace("-", "−") }, s);
    }
    for (const t of d.xticks) {
      el("line", { class: t.v === 1 ? "axis" : "grid", x1: sx(t.v), x2: sx(t.v), y1: y1, y2: y0 }, s);
      el("text", { class: "tick", x: sx(t.v), y: y0 + 28, text: t.label }, s);
    }
    el("text", { class: "axis-name", x: x1 - 70, y: sy(d.ylim[1]) + 22, text: "log₂ m" }, s);
    const pts = (keepIf, dx = (m) => m, dy = (y) => y) =>
      d.m.map((m, i) => (keepIf(m) ? [sx(dx(m)), sy(dy(d.y[i]))] : [NaN, NaN]));
    el("path", { class: "curve fold-all", d: pathD(pts(() => true)) }, s);
    el("path", { class: "curve fold-keep", d: pathD(pts((m) => m >= d.lo && m <= d.fold)) }, s);
    const topCurveBlue = el("path", { class: "curve fold-keep", d: pathD(pts((m) => m >= d.fold && m <= d.hi)) }, s);
    const movingCurve = el("path", { class: "curve fold-move" }, s);
    const inTop = (m) => m >= d.fold && m <= d.hi;

    return {
      render({ mark, fold }) {
        set(topBlue, { opacity: 1 - mark });
        set(topCurveBlue, { opacity: 1 - mark });
        set(moving, { opacity: mark });
        set(movingCurve, { opacity: mark });
        // The ends halve (m -> m/2) and log2 drops by 1, together.
        const k = lerp(1, 0.5, fold);
        place(moving, d.fold * k, d.hi * k);
        set(movingCurve, { d: pathD(pts(inTop, (m) => m * k, (y) => y - fold)) });
      },
    };
  },
};
