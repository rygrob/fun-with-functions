// Interpolation, step by step: a function, N sample points on it, the
// polynomial through them, and, in a panel beside it, the error p - f, zero
// at every sample point and oscillating in between. Laid out like the nodes
// figure's side layout: function left, error right, a label over each.
//
// Data (generate.py's "interp-sin-5"):
//   x, f, p, err      the samples of each curve
//   nodes             [{ x, y }], the sample points
//   ylim, err_lim     the function panel's y range; the error panel's is ±err_lim
//   xticks, yticks    for both panels
//   label, err_label  TeX over each panel
//   color, err_color  the polynomial / points, and the error
//
// Parameters (each 0..1, an opacity):
//   pts               the sample points
//   poly              the polynomial through them
//   err               the error panel
//
// <div class="figure" data-figure="interp" data-src="interp-sin-5"></div>
// data-width / data-height size the whole figure (default 1150 × 440).

import { el, set, pathD } from "../svg.js";
import { tex } from "../tex.js";
import { frame } from "./plot.js";

export default {
  defaults: { pts: 0, poly: 0, err: 0 },
  sticky: [],
  scrub: [],

  create(root, data) {
    const W = +root.dataset.width || 1150;
    const H = +root.dataset.height || 440;
    const row = document.createElement("div");
    row.className = "interp-row";
    root.appendChild(row);
    const panel = (name, label) => {
      const col = document.createElement("div");
      const head = document.createElement("div");
      head.className = "pane-label";
      tex(head, label);
      const div = document.createElement("div");
      Object.assign(div.dataset, { width: Math.round(W / 2 - 20), height: H - 48, left: 64, src: `${root.dataset.src}-${name}` });
      col.append(head, div);
      row.appendChild(col);
      return { col, div };
    };
    const left = panel("fn", data.label);
    const right = panel("err", data.err_label);

    const xlim = [data.x[0], data.x[data.x.length - 1]];
    const T = frame(left.div, { xlim, ylim: data.ylim, xticks: data.xticks, yticks: data.yticks });
    const L = data.err_lim;
    const E = frame(right.div, {
      xlim, ylim: [-L, L], xticks: data.xticks,
      yticks: [{ v: -L, label: `−${L}` }, { v: 0, label: "0" }, { v: L, label: `${L}` }],
    });
    const line = (F, y, color, cls = "curve") =>
      el("path", { class: cls, d: pathD(data.x.map((x, i) => [F.sx(x), F.sy(y[i])])), stroke: color }, F.curves);

    line(T, data.f, "#0b0b0b");
    const poly = line(T, data.p, data.color, "curve dashed");
    line(E, data.err, data.err_color);
    const dots = data.nodes.map((n) => el("circle", { class: "dot", cx: T.sx(n.x), cy: T.sy(n.y), r: 8, fill: data.color }, T.curves));
    data.nodes.forEach((n) => el("circle", { class: "dot", cx: E.sx(n.x), cy: E.sy(0), r: 7, fill: data.err_color }, E.curves));

    return {
      render({ pts, poly: p, err: e }) {
        dots.forEach((d) => set(d, { opacity: pts }));
        set(poly, { opacity: p });
        right.col.style.opacity = e;
      },
    };
  },
};
