// Aliasing in the Chebyshev spectrum, the way audio people picture it.
//
// Stems of |a_k| (log axis) for the true series, a dashed fold line at N
// (the Chebyshev "Nyquist": interpolating at N nodes), and the kept terms
// (k < N) shaded. Parameters, each 0 → 1, one per click:
//   fold    copies of the dropped terms (N < k < 2N) slide from k to their
//           mirror image 2N − k, beside the kept terms they land on
//           (T_(N+j) looks like −T_(N−j) at the nodes)
//   merge   each image rises into its kept stem and is absorbed; that
//           stem's dot turns the image colour: a_k now carries the alias.
//           (The change in |a_k| is ~2% at most, invisible on a log axis,
//           so it's shown by colour, not height.)
//   over    oversample: the fold line slides off to the right and the
//           kept terms are clean again
//   trunc   truncate: the terms past the kept ones drop away, leaving the
//           projection
//
// An inset in the empty top right zooms on the top kept term's tip on a
// linear scale, the one coefficient that moves visibly: its dot climbs by
// the real interpolant's change as `merge` goes in, and settles back with
// `over`.
//
// Data (generate.py's "cheb-fold"): plot as plot.js (x, series[0] the true
// stems, xlim, ylim, band, ticks), plus N, tail [{ k, y }] (the terms that
// fold), imageColor, dx (the images' offset beside the kept stems), zoom
// { k, y (log10 of the true value), from, to (true and aliased |a_k|),
// text (the change) }.

import { el, set, lerp, clamp01, smooth } from "../svg.js";
import { frame, legend } from "./plot.js";

export default {
  defaults: { fold: 0, merge: 0, over: 0, trunc: 0 },
  sticky: [],
  scrub: ["fold", "merge", "over", "trunc"],

  create(root, data) {
    const d = data.plot;
    const { s, sx, sy, curves } = frame(root, d);
    const bottom = sy(d.ylim[0]);

    const stem = (color, y) => {
      const g = el("g", {}, curves);
      const line = el("line", { class: "stem", x1: 0, x2: 0, y1: bottom, y2: sy(y), stroke: color }, g);
      const dot = el("circle", { class: "dot", cx: 0, cy: sy(y), r: 5, fill: color }, g);
      return { g, line, dot, y };
    };
    const truth = d.series[0];
    const stems = new Map();  // k -> the true stem
    d.x.forEach((k, i) => {
      if (truth.y[i] === null) return;
      const st = stem(truth.color, truth.y[i]);
      set(st.g, { transform: `translate(${sx(k)},0)` });
      // A cap in the image colour, faded in when this stem absorbs an alias.
      st.cap = el("circle", { class: "dot", cx: 0, cy: sy(truth.y[i]), r: 5, fill: data.imageColor, opacity: 0 }, st.g);
      stems.set(k, st);
    });
    const images = data.tail.map(({ k, y }) => ({ k, ...stem(data.imageColor, y) }));

    // Zoom inset: a box in the top right, joined to a_k's tip.
    const z = data.zoom;
    const box = { x0: sx(z.k + 3.2), x1: sx(d.xlim[1]) - 10, y0: sy(d.ylim[1]) + 30, y1: sy(d.ylim[1]) + 150 };
    const zy = (v) => lerp(box.y1 - 22, box.y0 + 22, (v - z.from) / (z.to - z.from));
    const zx = sx(z.k + 7);  // between stems, so it can't read as one
    const zoom = el("g", { class: "zoom-inset" }, s);
    el("line", { class: "zoom-link", x1: sx(z.k), y1: sy(z.y), x2: box.x0, y2: box.y1 }, zoom);
    el("rect", { class: "zoom-frame", x: box.x0, y: box.y0, width: box.x1 - box.x0, height: box.y1 - box.y0, rx: 8 }, zoom);
    el("line", { class: "stem", x1: zx, x2: zx, y1: box.y1, y2: zy(z.from), stroke: truth.color }, zoom);
    el("circle", { class: "dot", cx: zx, cy: zy(z.from), r: 6, fill: truth.color }, zoom);
    const zStem = el("line", { class: "stem", x1: zx, x2: zx, y1: zy(z.from), stroke: data.imageColor }, zoom);
    const zDot = el("circle", { class: "dot", cx: zx, r: 6, fill: data.imageColor }, zoom);
    el("text", { class: "zoom-name", x: zx - 14, y: zy(z.from) + 6, text: `a${String.fromCharCode(0x2080 + z.k)}` }, zoom);
    const zText = el("text", { class: "zoom-change", x: zx + 14, y: zy(z.to) + 6, text: z.text }, zoom);

    const line = el("line", { class: "fold-line", y1: sy(d.ylim[1]), y2: bottom }, s);
    const label = el("text", { class: "fold-label", y: sy(d.ylim[1]) + 18 }, s);
    legend(root, [truth, { text: "folded images", color: data.imageColor }]);

    const far = d.xlim[1] + 4 * (d.xlim[1] - data.N);  // well off the right edge

    return {
      render({ fold, merge, over, trunc }) {
        const f = smooth(clamp01(fold)), m = smooth(clamp01(merge));
        const o = smooth(clamp01(over)), t = smooth(clamp01(trunc));
        const N = data.N;
        const L = lerp(N, far, o);
        set(line, { x1: sx(L), x2: sx(L), opacity: 1 - o });
        set(label, { x: sx(L) + 8, text: `N = ${N}`, opacity: 1 - clamp01(2 * o) });

        for (const im of images) {
          const target = stems.get(2 * N - im.k);
          const x = m > 0 ? lerp(2 * N - im.k + data.dx, 2 * N - im.k, m) : lerp(im.k, 2 * L - im.k + data.dx, f);
          const y = sy(lerp(im.y, target.y, m));
          set(im.g, { transform: `translate(${sx(x)},0)`, opacity: clamp01(3 * f) * (1 - m) });
          set(im.line, { y2: y });
          set(im.dot, { cy: y });
          set(target.cap, { opacity: m * (1 - o) });
        }

        const zm = m * (1 - o);
        set(zStem, { y2: zy(lerp(z.from, z.to, zm)), opacity: zm > 0 ? 1 : 0 });
        set(zDot, { cy: zy(lerp(z.from, z.to, zm)), opacity: zm > 0 ? 1 : 0 });
        set(zText, { opacity: zm });
        set(zoom, { opacity: 1 - t });

        // Truncation: everything at or past N drops to the axis and fades.
        for (const [k, st] of stems) {
          if (k < N) continue;
          const y = lerp(sy(st.y), bottom, t);
          set(st.line, { y2: y });
          set(st.dot, { cy: y });
          set(st.g, { opacity: 1 - t });
        }
      },
    };
  },
};
