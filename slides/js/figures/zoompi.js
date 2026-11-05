// Zoom from the whole of sin x into its zero at π, down to the last few
// floats, where f32(π) and π come apart; then each method's f32 outputs as
// dots there (vanilla, factored, split π), morphing like morph.js.
//
// The view zooms in log scale: its width and height shrink geometrically,
// and its centre moves in step with the shrinking, so the target stays put
// on screen. x is stored as d = x − π (radians) so the deep end keeps its
// digits in JSON.
//
// Data (generate.py's "cheb-near-pi"):
//   sin               { d, y }: sin x, sampled densely toward π (log spaced)
//   gap               f32 spacing near π; f32_off: f32(π) − π
//   floats            d of each float the dots sit on
//   full, near        views { x: [lo, hi], y: [lo, hi] }, in d and sin
//   box               the box drawn on the full view, { x, y }
//   steps             [{ y }], each method's f32 output at the floats
//   color             the dots' colour
//   approx            [{ y }], each method's polynomial as a curve on sin's
//                     samples (exact, with the code's π constants);
//                     approxColor its colour. Drawn throughout, blended by
//                     step like the dots
//
// Parameters:
//   box               0..1, the zoom box's opacity
//   zoom              0..1, full view -> near view
//   step, show        as morph.js: which method's dots, and their opacity
//
// <div class="figure" data-figure="zoompi" data-src="cheb-near-pi"></div>
// data-dots="off" leaves "float outputs" out of the legend (for a slide
// that never shows the dots).

import { el, set, linear, pathD, pool, clamp01, smooth } from "../svg.js";
import { legend } from "./plot.js";

const M = { left: 72, right: 8, top: 8, bottom: 40 };

// "1.5e-6" -> "1.5e−6"; plain decimals for big steps.
function label(v, step) {
  if (v === 0) return "0";
  if (step >= 0.1) return String(+v.toFixed(1)).replace("-", "−");
  return (+v.toPrecision(3)).toExponential().replace(/\.?0+e/, "e").replace("e+", "e").replace(/-/g, "−");
}

// About n ticks across [lo, hi]: steps of 1, 2 or 5 times a power of ten.
function niceTicks(lo, hi, n = 5) {
  const raw = (hi - lo) / n;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return { ticks: out, step };
}

// One axis of the view at zoom t: log-interpolated size, centre moving in
// proportion to how much the size has shrunk.
function along(a, b, t) {
  const [s0, s1] = [a[1] - a[0], b[1] - b[0]];
  const [c0, c1] = [(a[0] + a[1]) / 2, (b[0] + b[1]) / 2];
  const s = s0 * Math.pow(s1 / s0, t);
  const c = c0 + (c1 - c0) * (s0 - s) / (s0 - s1);
  return [c - s / 2, c + s / 2];
}

export default {
  defaults: { box: 0, zoom: 0, step: 0, show: 0 },
  sticky: [],
  scrub: [],
  tweenMs: 2400,

  create(root, data) {
    const W = +root.dataset.width || 680;
    const H = +root.dataset.height || 440;
    const s = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "fig" }, root);
    const [X0, X1, Y0, Y1] = [M.left, W - M.right, H - M.bottom, M.top];
    const clipId = "clip-zoompi-" + Math.random().toString(36).slice(2);
    el("rect", { x: X0, y: Y1, width: X1 - X0, height: Y0 - Y1 }, el("clipPath", { id: clipId }, el("defs", {}, s)));

    const grid = el("g", {}, s);
    const plot = el("g", { "clip-path": `url(#${clipId})` }, s);
    const piLine = el("line", { class: "axis" }, plot);
    const f32Line = el("line", { class: "f32-pi" }, plot);
    const curve = el("path", { class: "curve", stroke: "#0b0b0b" }, plot);
    const approx = el("path", { class: "curve", stroke: data.approxColor }, plot);
    const box = el("rect", { class: "zoom-box" }, plot);
    const dots = data.floats.map(() => el("circle", { class: "dot", r: 8, fill: data.color }, plot));
    const floatTicks = pool(el("g", {}, s), "line", { class: "zoom-float" });
    const piLabel = el("text", { class: "tick", text: "π" }, s);
    const f32Label = el("text", { class: "tick f32-label", text: "f32(π)" }, s);
    const last = data.steps.length - 1;
    const keys = [{ label: "\\sin x", color: "#0b0b0b" }, { text: "our approximation", color: data.approxColor }];
    if (root.dataset.dots !== "off") keys.push({ text: "float outputs", color: data.color });
    legend(root, keys);

    return {
      render({ box: boxOn, zoom, step, show }) {
        const t = clamp01(zoom);
        const vx = along(data.full.x, data.near.x, t);
        const vy = along(data.full.y, data.near.y, t);
        const sx = linear(vx[0], vx[1], X0, X1);
        const sy = linear(vy[0], vy[1], Y0, Y1);

        // Grid and y ticks for this view; x ticks at −π, 0 (π is its own line).
        grid.replaceChildren();
        const { ticks, step: ystep } = niceTicks(vy[0], vy[1]);
        for (const v of ticks) {
          el("line", { class: v === 0 ? "axis" : "grid", x1: X0, x2: X1, y1: sy(v), y2: sy(v) }, grid);
          el("text", { class: "tick end", x: X0 - 10, y: sy(v) + 6, text: label(v, ystep) }, grid);
        }
        for (const [d, text] of [[-2 * Math.PI, "−π"], [-Math.PI, "0"]]) {
          const x = sx(d);
          if (x < X0 - 1 || x > X1 + 1) continue;
          el("line", { class: d === -Math.PI ? "axis" : "grid", x1: x, x2: x, y1: Y1, y2: Y0 }, grid);
          el("text", { class: "tick", x, y: Y0 + 28, text }, grid);
        }

        // Which method, blended between steps (the curve and the dots).
        const k = Math.min(last, Math.max(0, step));
        const i = Math.min(last - 1, Math.floor(k));
        const u = k - i;

        // sin x and the approximation: only the samples near the view, so
        // nothing is drawn at 1e9 px.
        const w = vx[1] - vx[0];
        const pts = [], apts = [];
        const [aa, ab] = [data.approx[i].y, data.approx[i + 1].y];
        data.sin.d.forEach((d, j) => {
          if (d < vx[0] - w || d > vx[1] + w) return;
          pts.push([sx(d), sy(data.sin.y[j])]);
          apts.push([sx(d), sy(aa[j] + (ab[j] - aa[j]) * u)]);
        });
        set(curve, { d: pathD(pts) });
        set(approx, { d: pathD(apts) });

        // π and f32(π): one line until the zoom pulls them apart; f32(π)'s label
        // sits at the top, clear of π's.
        const xPi = sx(0), xF = sx(data.f32_off);
        set(piLine, { x1: xPi, x2: xPi, y1: Y1, y2: Y0 });
        set(piLabel, { x: xPi, y: Y0 + 28 });
        const apart = clamp01((xF - xPi - 6) / 16);
        set(f32Line, { x1: xF, x2: xF, y1: Y1, y2: Y0, opacity: apart });
        set(f32Label, { x: xF + 8, y: Y1 + 20, opacity: apart });

        // Every float, once they're far enough apart to see.
        const gapPx = sx(data.gap) - sx(0);
        const seen = clamp01((gapPx - 10) / 30);
        const first = Math.ceil((vx[0] - data.f32_off) / data.gap);
        const n = seen > 0 ? Math.max(0, Math.floor((vx[1] - data.f32_off) / data.gap) - first + 1) : 0;
        floatTicks(n).forEach((line, i) => {
          const x = sx(data.f32_off + (first + i) * data.gap);
          set(line, { x1: x, x2: x, y1: Y0 - 10, y2: Y0, opacity: seen });
        });

        // The box: drawn in data coordinates, so it grows past the frame and
        // fades. Its edges are clamped to just outside the frame, and it's
        // hidden once faded: a dashed stroke millions of pixels long stalls
        // the renderer (Safari especially) on every frame of the zoom.
        const b = data.box;
        const bw = sx(b.x[1]) - sx(b.x[0]);
        const boxAlpha = boxOn * (1 - smooth(clamp01((bw / (X1 - X0) - 0.25) / 0.5)));
        const pad = 20;
        const bx0 = Math.max(sx(b.x[0]), X0 - pad), bx1 = Math.min(sx(b.x[1]), X1 + pad);
        const by0 = Math.max(sy(b.y[1]), Y1 - pad), by1 = Math.min(sy(b.y[0]), Y0 + pad);
        set(box, {
          x: bx0, y: by0, width: Math.max(0, bx1 - bx0), height: Math.max(0, by1 - by0),
          opacity: boxAlpha, display: boxAlpha > 0 ? "inline" : "none",
        });

        // Each method's dots, blended between steps; they need the floats resolved.
        const [ya, yb] = [data.steps[i].y, data.steps[i + 1].y];
        dots.forEach((dot, j) => set(dot, {
          cx: sx(data.floats[j]), cy: sy(ya[j] + (yb[j] - ya[j]) * u), opacity: show * seen,
        }));
      },
    };
  },
};
