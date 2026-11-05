// Interpolating sin at points that slide from equally spaced to Chebyshev:
// the function and interpolant, and the signed error.
//
// Data (generate.py "nodes-sin"):
//   x, sin            the plotting grid and sin on it
//   frames            [{ nodes, p, err, max }] along q from 0 to 1, from
//                     example 15's barycentric interpolation; the figure
//                     blends neighbouring frames
//   err_lim           the error panel's half-height (the equally spaced
//                     error runs off it, on purpose)
//   xticks            [{ v, label }]
//
// Parameters:
//   q                 0 equally spaced .. 1 Chebyshev
//   err               0..1, the error panel's opacity
//   runge             0..1, shading on the outer two intervals at each end
//                     (it tracks the points)
//
// data-layout="side" puts the panels side by side (function left, error
// right); the default stacks them.
// <div class="figure" data-figure="nodes" data-src="nodes-sin"></div>

import { el, set, svg, linear, pathD, pool } from "../svg.js";
import { tex } from "../tex.js";

let instances = 0;  // unique clip-path ids: the figure appears on two slides

function sci(v) {
  const e = Math.floor(Math.log10(v));
  return String.raw`${(v / 10 ** e).toFixed(2)} \times 10^{${e}}`;
}

// Panel rectangles for each layout: { x0, x1, y0, y1 } in SVG pixels.
function layout(side, W, H) {
  if (side) {
    const mid = W / 2;
    return {
      fn: { x0: 20, x1: mid - 30, y0: 50, y1: H - 40 },
      er: { x0: mid + 80, x1: W - 10, y0: 50, y1: H - 40 },
    };
  }
  return {
    fn: { x0: 80, x1: W - 20, y0: 10, y1: 180 },
    er: { x0: 80, x1: W - 20, y0: 215, y1: H - 40 },
  };
}

export default {
  defaults: { q: 0, err: 0, runge: 0 },
  sticky: [],
  scrub: [],

  create(root, data) {
    const side = root.dataset.layout === "side";
    const W = +root.dataset.width || 1150;
    const H = +root.dataset.height || 450;
    const s = svg(root, W, H);
    const { fn, er } = layout(side, W, H);
    const lo = data.x[0], hi = data.x[data.x.length - 1];
    const fx = linear(lo, hi, fn.x0, fn.x1), ex = linear(lo, hi, er.x0, er.x1);
    const fy = linear(-1.25, 1.25, fn.y1, fn.y0);
    const ey = linear(-data.err_lim, data.err_lim, er.y1, er.y0);
    // Runge shading: from each end to the third point in (the outer two
    // gaps), on both panels. It follows the points, so it narrows as they
    // crowd toward the ends.
    const shade = el("g", {}, s);
    const bands = [[fx, fn], [ex, er]].map(([sx, p]) => [sx,
      el("rect", { class: "runge-band", y: p.y0, height: p.y1 - p.y0 }, shade),
      el("rect", { class: "runge-band", y: p.y0, height: p.y1 - p.y0 }, shade)]);

    // Function panel: sin, the interpolant, the points.
    el("line", { class: "axis", x1: fx(lo), x2: fx(hi), y1: fy(0), y2: fy(0) }, s);
    // Side layout: TeX over each panel, as in the interp figure.
    const label = (x, src) => {
      const div = document.createElement("div");
      div.className = "pane-label nodes-label";
      div.style.left = `${x}px`;
      tex(div, src);
      root.appendChild(div);
      return div;
    };
    if (side) label(fn.x0, String.raw`\sin x \text{ and } p(x)`);
    else el("text", { class: "pane-tag", x: 0, y: fy(0) + 6, text: "sin x" }, s);
    el("path", { class: "curve", stroke: "var(--ink)", d: pathD(data.x.map((x, i) => [fx(x), fy(data.sin[i])])) }, s);
    const interp = el("path", { class: "curve dashed", stroke: "var(--blue)" }, s);
    const topDots = pool(el("g", {}, s), "circle", { r: 6, class: "node-dot" });

    // Error panel, clipped.
    const panel = el("g", {}, s);
    const clipId = `nodes-clip-${instances++}`;
    const clip = el("clipPath", { id: clipId }, el("defs", {}, s));
    el("rect", { x: er.x0 - 2, y: er.y0, width: er.x1 - er.x0 + 4, height: er.y1 - er.y0 }, clip);
    el("line", { class: "grid", x1: er.x0, x2: er.x1, y1: er.y0, y2: er.y0 }, panel);
    el("line", { class: "grid", x1: er.x0, x2: er.x1, y1: er.y1, y2: er.y1 }, panel);
    el("line", { class: "axis", x1: er.x0, x2: er.x1, y1: ey(0), y2: ey(0) }, panel);
    const errLabel = side ? label(er.x0, String.raw`p(x) - \sin x`) : null;
    if (!side) el("text", { class: "pane-tag", x: 0, y: ey(0) + 6, text: "error" }, panel);
    const lim = data.err_lim.toExponential(1).replace("e-", "e−");
    el("text", { class: "tick end", x: er.x0 - 10, y: er.y0 + 6, text: `+${lim}` }, panel);
    el("text", { class: "tick end", x: er.x0 - 10, y: er.y1 + 6, text: `−${lim}` }, panel);
    const err = el("path", { class: "curve", stroke: "var(--blue)", "clip-path": `url(#${clipId})` }, panel);
    const botDots = pool(el("g", {}, panel), "circle", { r: 5, class: "node-dot" });
    for (const t of data.xticks) {
      el("text", { class: "tick", x: ex(t.v), y: H - 10, text: t.label }, panel);
      if (side) el("text", { class: "tick", x: fx(t.v), y: H - 10, text: t.label }, s);
    }
    if (!side) for (const t of data.xticks) el("text", { class: "tick", x: fx(t.v), y: H - 10, text: t.label }, s);

    const rungeLabel = el("text", { class: "runge-label", x: er.x1, y: side ? 24 : er.y0 - 8, "text-anchor": "end", text: "Runge's phenomenon" }, s);

    const readout = document.createElement("div");
    readout.className = "nodes-readout" + (side ? " side" : "");
    root.appendChild(readout);

    const F = data.frames.length;
    const big = 8 * data.err_lim;  // keep off-panel values finite for the path

    return {
      render({ q, err: show, runge }) {
        const k = Math.min(F - 1, Math.max(0, q * (F - 1)));
        const i = Math.min(F - 2, Math.floor(k)), u = k - i;
        const [a, b] = [data.frames[i], data.frames[i + 1]];
        const mix = (p, r, j) => p[j] + (r[j] - p[j]) * u;
        const nodes = a.nodes.map((_, j) => mix(a.nodes, b.nodes, j));

        set(interp, { d: pathD(data.x.map((x, j) => [fx(x), fy(Math.max(-1.3, Math.min(1.3, mix(a.p, b.p, j))))])) });
        set(err, { d: pathD(data.x.map((x, j) => [ex(x), ey(Math.max(-big, Math.min(big, mix(a.err, b.err, j))))])) });
        topDots(nodes.length).forEach((c, j) => set(c, { cx: fx(nodes[j]), cy: fy(Math.sin(nodes[j])) }));
        botDots(nodes.length).forEach((c, j) => set(c, { cx: ex(nodes[j]), cy: ey(0) }));
        set(panel, { opacity: show });
        if (errLabel) errLabel.style.opacity = show;
        set(shade, { opacity: runge });
        const [inL, inR] = [nodes[2], nodes[nodes.length - 3]];
        for (const [sx, left, right] of bands) {
          set(left, { x: sx(lo), width: sx(inL) - sx(lo) });
          set(right, { x: sx(inR), width: sx(hi) - sx(inR) });
        }
        set(rungeLabel, { opacity: runge * (1 - q) });  // the bands stay; the name goes with the points
        readout.style.opacity = show;
        tex(readout, String.raw`\max\,|p(x) - \sin x| = ${sci(a.max + (b.max - a.max) * u)}`);
      },
    };
  },
};
