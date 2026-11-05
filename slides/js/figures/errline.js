// Measuring error, on a number line: a toy float's values as ticks, the
// true f(x) between two of them, the computed p(x) on one. Absolute error is
// the gap between them; relative zooms out to 0 to set the gap against
// |f(x)|; ULP zooms back in and counts the gap in float spacings at f.
//
// Data (generate.py "error-line"):
//   values            every representable value, ascending
//   f, p              the true and computed values
//   ulp, ulp_lo       the float spacing at f, and the float just below it
//   n_ulp             |p - f| / ulp
//   near, wide        the view, zoomed in and out to 0
//   tex               labels: f, p, abs, rel, ulp, mag, gap
//
// Parameters (each 0..1):
//   abs               the gap and its size
//   rel               |f(x)| from 0, and the ratio
//   ulp               the spacing at f, the gap ruled off in it
//   zoom              near -> wide
//
// <div class="figure" data-figure="errline" data-src="error-line"></div>

import { el, set, svg, lerp } from "../svg.js";
import { tex } from "../tex.js";

const PAD = 20;
const LINE_Y = 120;  // the number line
const SPAN_Y = 72;   // the error, above it
const BAR_Y = 206;   // |f(x)| or ulp(f(x)), under it

export default {
  defaults: { abs: 0, rel: 0, ulp: 0, zoom: 0 },
  sticky: [],
  scrub: [],

  create(root, d) {
    const W = +root.dataset.width || 540;
    const H = +root.dataset.height || 270;
    const s = svg(root, W, H);
    const clip = el("clipPath", { id: "errline-clip" }, el("defs", {}, s));
    el("rect", { x: PAD - 6, y: 0, width: W - 2 * PAD + 12, height: H }, clip);
    const g = el("g", { "clip-path": "url(#errline-clip)" }, s);

    // A TeX label centred on x: HTML over the SVG, not a foreignObject
    // (Safari misplaces a foreignObject's contents once it fades).
    const label = (src, y, cls = "") => {
      const div = document.createElement("div");
      div.className = `err-label ${cls}`;
      div.style.top = `${y}px`;
      root.appendChild(div);
      tex(div, src);
      return (x, opacity) => {
        div.style.left = `${x}px`;
        div.style.opacity = opacity;
      };
    };
    const bracket = (cls) => {
      const gr = el("g", { class: cls }, g);
      return {
        g: gr,
        bar: el("line", {}, gr),
        a: el("line", {}, gr),
        b: el("line", {}, gr),
        place(x0, x1, y, h = 8) {
          set(this.bar, { x1: x0, x2: x1, y1: y, y2: y });
          set(this.a, { x1: x0, x2: x0, y1: y - h, y2: y + h });
          set(this.b, { x1: x1, x2: x1, y1: y - h, y2: y + h });
        },
      };
    };

    el("line", { class: "axis", x1: PAD, x2: W - PAD, y1: LINE_Y, y2: LINE_Y }, s);
    const ticks = d.values.map(() => el("line", { class: "float-tick", y1: LINE_Y - 12, y2: LINE_Y + 12 }, g));
    const zero = el("text", { class: "tick", y: LINE_Y + 40, text: "0" }, s);

    const guideF = el("line", { class: "err-guide", y1: SPAN_Y, y2: LINE_Y }, g);
    const guideP = el("line", { class: "err-guide", y1: SPAN_Y, y2: LINE_Y }, g);
    const span = bracket("err-span");
    const marks = Array.from({ length: Math.floor(d.n_ulp) }, () => el("line", { class: "err-mark" }, g));
    const mag = bracket("err-mag");
    const gap = bracket("err-gap");

    const dotF = el("circle", { class: "err-true", cy: LINE_Y, r: 7 }, s);
    const dotP = el("circle", { class: "err-approx", cy: LINE_Y, r: 7 }, s);
    const fLabel = label(d.tex.f, LINE_Y + 16);
    const pLabel = label(d.tex.p, LINE_Y + 16, "c-orange");
    const absLabel = label(d.tex.abs, SPAN_Y - 50, "c-orange");
    const relLabel = label(d.tex.rel, SPAN_Y - 50, "c-orange");
    const ulpLabel = label(d.tex.ulp, SPAN_Y - 50, "c-orange");
    const magLabel = label(d.tex.mag, BAR_Y + 10);
    const gapLabel = label(d.tex.gap, BAR_Y + 10, "c-blue");

    return {
      render({ abs, rel, ulp, zoom }) {
        const lo = lerp(d.near[0], d.wide[0], zoom);
        const hi = lerp(d.near[1], d.wide[1], zoom);
        const sx = (v) => PAD + ((v - lo) / (hi - lo)) * (W - 2 * PAD);
        const [xf, xp] = [sx(d.f), sx(d.p)];

        d.values.forEach((v, i) => set(ticks[i], { x1: sx(v), x2: sx(v) }));
        set(zero, { x: sx(0), opacity: zoom });
        set(dotF, { cx: xf });
        set(dotP, { cx: xp });
        fLabel(xf, 1);
        pLabel(xp, 1);

        // The error: the gap from f to p, with its size above.
        for (const [line, x] of [[guideF, xf], [guideP, xp]]) set(line, { x1: x, x2: x, opacity: abs });
        span.place(xf, xp, SPAN_Y);
        set(span.g, { opacity: abs });
        const mid = (xf + xp) / 2;
        absLabel(mid, abs * (1 - rel) * (1 - ulp));
        relLabel(mid, rel);
        ulpLabel(mid, ulp);

        // Relative: the distance from 0 to f underneath.
        mag.place(sx(0), xf, BAR_Y);
        set(mag.g, { opacity: rel });
        magLabel((sx(0) + xf) / 2, rel);

        // ULP: the float spacing at f, and the gap ruled off in it.
        gap.place(sx(d.ulp_lo), sx(d.ulp_lo + d.ulp), BAR_Y);
        set(gap.g, { opacity: ulp });
        gapLabel(sx(d.ulp_lo + d.ulp / 2), ulp);
        marks.forEach((m, k) => {
          const x = sx(d.f + (k + 1) * d.ulp);
          set(m, { x1: x, x2: x, y1: SPAN_Y - 8, y2: SPAN_Y + 8, opacity: ulp });
        });
      },
    };
  },
};
