// Static line plot of curves from slides/generate.py.
//
// Data (one generate.py item's "plot"):
//   x                 shared sample positions
//   series            [{ label (TeX) or text (plain), color, dashed?, stems?, r?, dx?, gaps?, y }],
//                     drawn in order; stems draws a stem and dot (radius r,
//                     default 6) per sample, from the bottom of the plot,
//                     instead of a line, shifted by dx (two sets side by side);
//                     gaps breaks the line at null samples (poles)
//   xlim, ylim        axis ranges; curves are clipped to them
//   xticks            [{ v, label }]; yticks: [v, ...] or [{ v, label }]
//   band              optional [a, b], shaded (the fit interval)
//   vlines            optional [x, ...], dashed verticals (asymptotes)
//   points            optional [{ x, y, color, r? }], dots drawn over the curves
//   markers           optional [{ x, label: [line, ...] }], a solid vertical in
//                     blue with its label to the left, in the lower half, faded
//                     in by the `mark` parameter (a reference point, 20 kHz)
//   views             optional [{ xlim, xticks: [v, ...] }]: x ranges to move
//                     between (line series only); xticks then lists every
//                     tick any view uses, and xlim is ignored
//   fold              optional { N, keep, color, dx, text, tail: [{ k, y }] }:
//                     aliasing in a stem spectrum (series[0]), driven by the
//                     `fold`, `over` and `trunc` parameters (the retired fold.js,
//                     without its merge)
//
// Parameters:
//   shown             how many series are visible, in order (default all);
//                     fractional values fade the next one in, legend row too
//   mark              how many markers are visible, in order (default none;
//                     fractional values fade the next one in)
//   view              with views: 0 .. views.length − 1; fractional values
//                     stretch the x axis between neighbouring views (both
//                     plots of a stack move together: they share the click)
//   fold              with fold, 0 → 1: a dashed line at N fades in and the
//                     tail stems turn the fold colour (first third), then those
//                     copies slide from k to beside their mirror image 2N − k
//   over              with fold, 0 → 1: oversample. The fold line slides off
//                     to the right, taking the images with it
//   trunc             with fold, 0 → 1: truncate. series[0]'s stems past keep
//                     drop to the axis and fade
//
// <div class="figure" data-figure="plot" data-src="cheb-sin-5"></div>
// Size in slide pixels: data-width / data-height (default 1100 × 350);
// data-left widens the left margin for longer y tick labels.

import { el, set, svg, linear, pathD, lerp, clamp01, smooth } from "../svg.js";
import { tex } from "../tex.js";

const M = { left: 48, right: 8, top: 8, bottom: 40 };

export default {
  defaults: { shown: 99, view: 0, mark: 0, fold: 0, over: 0, trunc: 0 },
  sticky: [],
  scrub: [],

  create(root, data) {
    const d = data.plot;
    const { sx, sy, curves, setView } = frame(root, d);
    // Views: curves are drawn once at the first view and stretched by a
    // transform on an inner group (the clip stays put); strokes don't scale.
    const inner = d.views ? el("g", {}, curves) : curves;
    const groups = d.series.map((c) => {
      const g = el("g", {}, inner);
      if (c.stems) {
        g.stems = [];
        d.x.forEach((x0, i) => {
          if (c.y[i] === null) return;
          const x = x0 + (c.dx || 0);
          const line = el("line", { class: "stem", x1: sx(x), x2: sx(x), y1: sy(d.ylim[0]), y2: sy(c.y[i]), stroke: c.color }, g);
          const dot = el("circle", { class: "dot", cx: sx(x), cy: sy(c.y[i]), r: c.r || 6, fill: c.color }, g);
          g.stems.push({ k: x0, y: c.y[i], line, dot });
        });
        return g;
      }
      // Far-off values (a polynomial run off to 1e9) are pinned a plot height
      // outside the frame: hidden by the clip either way, but coordinates in
      // the billions can make a GPU renderer drop the whole stretched group.
      const [top, bottom] = [sy(d.ylim[1]), sy(d.ylim[0])];
      const pin = (v) => Math.min(bottom + (bottom - top), Math.max(top - (bottom - top), sy(v)));
      const pts = d.x.map((x, i) => [sx(x), c.y[i] === null ? NaN : pin(c.y[i])]);
      const path = el("path", { class: "curve" + (c.dashed ? " dashed" : ""), d: pathD(pts, c.gaps), stroke: c.color }, g);
      if (d.views) set(path, { "vector-effect": "non-scaling-stroke" });
      return g;
    });
    points(d, sx, sy, curves);
    const marks = (d.markers || []).map((m) => {
      const g = el("g", { class: "marker" }, frameSvg(root));
      el("line", { x1: sx(m.x), x2: sx(m.x), y1: sy(d.ylim[1]), y2: sy(d.ylim[0]) }, g);
      m.label.forEach((line, k) => el("text", { x: sx(m.x) - 10, y: sy(d.ylim[0] * 0.4) + 24 * k, text: line }, g));
      return g;
    });

    const rows = legend(root, d.fold ? [...d.series, { text: d.fold.text, color: d.fold.color }] : d.series);
    const folds = d.fold && foldOverlay(d, sx, sy, curves, frameSvg(root), rows.at(-1), groups[0].stems);

    return {
      render({ shown, view, mark, fold, over, trunc }) {
        if (folds) folds(fold, over, trunc);
        marks.forEach((g, i) => set(g, { opacity: Math.min(1, Math.max(0, mark - i)) }));
        if (d.views) set(inner, { transform: setView(view) });
        groups.forEach((g, i) => {
          const o = Math.min(1, Math.max(0, shown - i));
          set(g, { opacity: o });
          rows[i].style.opacity = o;
        });
      },
    };
  },
};

const frameSvg = (root) => root.querySelector("svg.fig");

// The fold: copies of the tail stems in the fold colour, which slide from k
// to their mirror image about N (T_(N+j) looks like −T_(N−j) at the N
// nodes), offset by dx so they sit beside the kept stem they land on.
function foldOverlay(d, sx, sy, curves, s, row, stems) {
  const f = d.fold;
  const bottom = sy(d.ylim[0]);
  const images = f.tail.map(({ k, y }) => {
    const g = el("g", {}, curves);
    el("line", { class: "stem", x1: 0, x2: 0, y1: bottom, y2: sy(y), stroke: f.color }, g);
    el("circle", { class: "dot", cx: 0, cy: sy(y), r: 5, fill: f.color }, g);
    return { k, g };
  });
  const line = el("line", { class: "fold-line", y1: sy(d.ylim[1]), y2: bottom }, s);
  const label = el("text", { class: "fold-label", y: sy(0) + 28, text: `N = ${f.N}` }, s);
  const far = d.xlim[1] + 4 * (d.xlim[1] - f.N);  // well off the right edge
  return (fold, over, trunc) => {
    const a = clamp01(3 * fold), b = smooth(clamp01((fold - 1 / 3) * 1.5));
    const o = smooth(clamp01(over)), t = smooth(clamp01(trunc));
    // Oversampling moves N (and so every mirror image) off to the right.
    const N = lerp(f.N, far, o);
    set(line, { x1: sx(N), x2: sx(N), opacity: a * (1 - o) });
    set(label, { x: sx(N) + 8, opacity: a * (1 - clamp01(3 * o)) });
    row.style.opacity = a * (1 - o);
    for (const im of images) {
      set(im.g, { transform: `translate(${sx(lerp(im.k, 2 * N - im.k + f.dx, b))},0)`, opacity: a });
    }
    // Truncation: the stems past keep drop to the axis and fade.
    for (const st of stems) {
      if (st.k <= f.keep) continue;
      const y = lerp(sy(st.y), bottom, t);
      set(st.line, { y2: y, opacity: 1 - t });
      set(st.dot, { cy: y, opacity: 1 - t });
    }
  };
}

// Axes, ticks, the optional band, and a clipped group for the curves. Shared
// with the other plot-style figures (morph.js).
export function frame(root, d) {
  const W = +root.dataset.width || 1100;
  const H = +root.dataset.height || 350;
  const s = svg(root, W, H);
  const left = +root.dataset.left || M.left;
  const xlim = d.views ? d.views[0].xlim : d.xlim;
  const sx = linear(xlim[0], xlim[1], left, W - M.right);
  const sy = linear(d.ylim[0], d.ylim[1], H - M.bottom, M.top);
  const [x0, x1, y0, y1] = [sx(xlim[0]), sx(xlim[1]), sy(d.ylim[0]), sy(d.ylim[1])];

  const clipId = `clip-${root.dataset.src}`;
  el("rect", { x: x0, y: y1, width: x1 - x0, height: y0 - y1 }, el("clipPath", { id: clipId }, el("defs", {}, s)));

  const band = d.band &&
    el("rect", { class: "band", x: sx(d.band[0]), y: y1, width: sx(d.band[1]) - sx(d.band[0]), height: y0 - y1 }, s);
  for (const t of d.yticks) {
    const { v, label } = typeof t === "number" ? { v: t, label: String(t) } : t;
    el("line", { class: v === 0 ? "axis" : "grid", x1: x0, x2: x1, y1: sy(v), y2: sy(v) }, s);
    el("text", { class: "tick end", x: x0 - 10, y: sy(v) + 6, text: label.replace("-", "−") }, s);
  }
  const xLines = [];
  const xLabels = d.xticks.map((t) => {
    xLines.push(el("line", { class: t.v === 0 ? "axis" : "grid", x1: sx(t.v), x2: sx(t.v), y1: y1, y2: y0 }, s));
    return el("text", { class: "tick", x: sx(t.v), y: y0 + 28, text: t.label }, s);
  });
  const vlines = (d.vlines || []).map((v) =>
    el("line", { class: "asymptote", x1: sx(v), x2: sx(v), y1: y1, y2: y0 }, s));

  const curves = el("g", { "clip-path": `url(#${clipId})` }, s);

  // With views: move everything that depends on x to view u (fractional:
  // between two), show the nearest view's ticks, and return the transform
  // that takes curves drawn at the first view to this one.
  const setView = (u) => {
    const k = Math.min(d.views.length - 1, Math.max(0, u));
    const i = Math.min(d.views.length - 2, Math.floor(k));
    const [a, b] = [d.views[i].xlim, d.views[i + 1].xlim];
    const f = k - i;
    const lim = [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
    const nx = linear(lim[0], lim[1], left, W - M.right);
    const keep = d.views[Math.round(k)].xticks;
    const inside = (v) => v >= lim[0] - 1e-9 && v <= lim[1] + 1e-9;
    d.xticks.forEach((t, j) => {
      const on = keep.includes(t.v) && inside(t.v) ? 1 : 0;
      set(xLines[j], { x1: nx(t.v), x2: nx(t.v), opacity: t.v === 0 && inside(0) ? 1 : on });
      set(xLabels[j], { x: nx(t.v), opacity: on });
    });
    vlines.forEach((line, j) => set(line, { x1: nx(d.vlines[j]), x2: nx(d.vlines[j]) }));
    if (band) {
      const [b0, b1] = [nx(d.band[0]), Math.min(nx(d.band[1]), x1)];
      set(band, { x: b0, width: Math.max(0, b1 - b0) });
    }
    // pixel p at the first view is data xlim[0] + (p − x0) / sx-scale; map it to nx.
    const scale = (nx(1) - nx(0)) / (sx(1) - sx(0));
    return `matrix(${scale},0,0,1,${nx(xlim[0]) - scale * x0},0)`;
  };
  return { s, sx, sy, curves, xLabels, setView };
}

// The optional fixed dots (interpolation nodes), over everything else.
export function points(d, sx, sy, parent) {
  for (const p of d.points || []) {
    el("circle", { class: "dot", cx: sx(p.x), cy: sy(p.y), r: p.r || 7, fill: p.color }, parent);
  }
}

// TeX labels with a line swatch in each series' colour; returns the rows.
export function legend(root, series) {
  const box = document.createElement("div");
  box.className = "legend";
  for (const c of series) {
    const row = document.createElement("div");
    row.innerHTML = `<svg width="36" height="12"><line x1="0" x2="36" y1="6" y2="6" stroke="${c.color}"
      stroke-width="3" ${c.dashed ? 'stroke-dasharray="7 5"' : ""}/></svg><span></span>`;
    const span = row.querySelector("span");
    if (c.text) span.textContent = c.text;
    else tex(span, c.label);
    box.appendChild(row);
  }
  // Never wider than the plot: a long legend wraps instead of widening the figure.
  const svgEl = root.querySelector("svg.fig");
  if (svgEl) box.style.maxWidth = `${svgEl.getAttribute("width")}px`;
  root.appendChild(box);
  return [...box.children];
}
