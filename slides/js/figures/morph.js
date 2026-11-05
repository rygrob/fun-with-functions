// A plot whose one moving curve morphs through a sequence of steps (Taylor
// polynomials of rising degree), over static curves (the function itself).
//
// Data (one generate.py item's "plot"): as plot.js, plus
//   steps             [{ y }], the moving curve at each step
//   color             the moving curve's colour
//   curves            instead of steps/color: [{ text or label, color, steps }],
//                     several curves moving together (with a legend)
//   points            optional fixed dots, as plot.js
//   stepPoints        optional [[{ x, y, color, r? }], ...]: dots per step
//                     (interpolation nodes), shown for the nearest step
//   curves[].dashed   a dashed moving curve; curves[].nolegend leaves it
//                     out of the legend (the lower half of an envelope)
//   stepXlabels       optional [[label per xtick], ...]: the x tick labels
//                     per step (a change of units), for the nearest step
//   stepYlims         optional [{ ylim, yticks }, ...]: the y axis per step
//                     (instead of ylim / yticks); fractional steps stretch
//                     it between neighbours, ticks from the nearest step
//
// Parameters:
//   step              0 .. steps.length − 1; fractional values blend the
//                     neighbouring steps pointwise, so a tween morphs one
//                     curve into the next
//   show              0..1, the moving curve's opacity
//
// <div class="figure" data-figure="morph" data-src="taylor-sin"></div>

import { el, set, linear, pathD } from "../svg.js";
import { frame, points, legend } from "./plot.js";

export default {
  defaults: { step: 0, show: 0 },
  sticky: [],
  scrub: [],

  create(root, data) {
    const d = data.plot;
    const ys = d.stepYlims;
    const { s, sx, sy, curves, xLabels } = frame(root, ys ? { ...d, ylim: ys[0].ylim, yticks: [] } : d);
    // A moving y axis: every step's ticks, under the curves, placed per render.
    const [py0, py1] = ys ? [sy(ys[0].ylim[0]), sy(ys[0].ylim[1])] : [];
    const [px0, px1] = [sx(d.xlim[0]), sx(d.xlim[1])];
    const yTicks = [];
    for (const { yticks } of ys || []) {
      for (const t of yticks) {
        const { v, label } = typeof t === "number" ? { v: t, label: String(t) } : t;
        if (yTicks.some((y) => y.v === v)) continue;
        const line = el("line", { class: v === 0 ? "axis" : "grid", x1: px0, x2: px1 }, s);
        const text = el("text", { class: "tick end", x: px0 - 10, text: label.replace("-", "−") }, s);
        s.insertBefore(line, curves);
        s.insertBefore(text, curves);
        yTicks.push({ v, line, text });
      }
    }
    for (const c of d.series) {
      const pts = d.x.map((x, i) => [sx(x), c.y[i] === null ? NaN : sy(c.y[i])]);
      el("path", { class: "curve" + (c.dashed ? " dashed" : ""), d: pathD(pts), stroke: c.color }, curves);
    }
    const movers = d.curves || [{ color: d.color, steps: d.steps }];
    const paths = movers.map((c) => el("path", { class: "curve" + (c.dashed ? " dashed" : ""), stroke: c.color }, curves));
    points(d, sx, sy, curves);
    const stepDots = (d.stepPoints || []).map((pts) => {
      const g = el("g", {}, curves);
      for (const p of pts) el("circle", { class: "dot", cx: sx(p.x), cy: sy(p.y), r: p.r || 6, fill: p.color }, g);
      return g;
    });
    if (d.curves) legend(root, d.curves.filter((c) => !c.nolegend));
    const last = movers[0].steps.length - 1;

    return {
      render({ step, show }) {
        const k = Math.min(last, Math.max(0, step));
        const i = Math.min(last - 1, Math.floor(k));
        const u = k - i;
        let y = sy;
        if (ys) {
          // Zoom geometrically (a 10x zoom moves evenly), centre linearly.
          const [a, b] = [ys[i].ylim, ys[i + 1].ylim];
          const mid = ((a[0] + a[1]) / 2) * (1 - u) + ((b[0] + b[1]) / 2) * u;
          const half = Math.pow(a[1] - a[0], 1 - u) * Math.pow(b[1] - b[0], u) / 2;
          const lim = [mid - half, mid + half];
          y = linear(lim[0], lim[1], py0, py1);
          // The nearest step's ticks, once they're far enough apart to read.
          const keep = ys[Math.round(k)].yticks.map((t) => (typeof t === "number" ? t : t.v));
          const roomy = keep.every((v, j) => j === 0 || Math.abs(y(v) - y(keep[j - 1])) >= 28);
          for (const t of yTicks) {
            const on = roomy && keep.includes(t.v) && t.v >= lim[0] - 1e-9 && t.v <= lim[1] + 1e-9 ? 1 : 0;
            set(t.line, { y1: y(t.v), y2: y(t.v), opacity: on });
            set(t.text, { y: y(t.v) + 6, opacity: on });
          }
        }
        movers.forEach((c, m) => {
          const [a, b] = [c.steps[i].y, c.steps[i + 1].y];
          const pts = d.x.map((x, j) => [sx(x), y(a[j] + (b[j] - a[j]) * u)]);
          set(paths[m], { d: pathD(pts), opacity: show });
        });
        stepDots.forEach((g, j) => set(g, { opacity: j === Math.round(k) ? show : 0 }));
        if (d.stepXlabels) xLabels.forEach((t, j) => set(t, { text: d.stepXlabels[Math.round(k)][j] }));
      },
    };
  },
};
