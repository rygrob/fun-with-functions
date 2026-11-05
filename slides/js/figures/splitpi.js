// Split π on two number lines at the same scale. Top: the floats near π,
// π between two of them, π_hi = f32(π) the one above, and x the float just
// below π. Bottom: near 0, where the floats are so dense the line looks
// solid. Subtracting π_hi drops x straight down onto the bottom line,
// exactly, with 0 under π_hi; π drops to π − π_hi, which π_lo carries;
// subtracting π_lo moves 0 along π_lo onto π. x and π never move: only
// where we measure from does.
//
// Until the last step the names are f32(π) and δ (the correction); as π_lo
// is subtracted they cross-fade to π_hi and π_lo.
//
// Data (generate.py "split-pi-line"): offsets from π, in f64:
//   gap               f32 spacing near π
//   hi, x             π_hi − π and x − π
//   lo                π_lo (f32)
//   tex_ulp_top, tex_ulp_bottom   each line's float spacing, for its caption
//
// Parameters (each 0..1):
//   want              the brackets x − π (ink) and x − π_hi (blue), top
//   drop              x − π_hi: x falls to the bottom line, its bracket there
//   miss              π_lo on the top line; π falls to its image, x − π there
//   lo                subtract π_lo: the bottom line's 0 moves onto π, and
//                     the names change to π_hi, π_lo
//
// <div class="figure" data-figure="splitpi" data-src="split-pi-line"></div>

import { el, set, linear, lerp, clamp01, smooth } from "../svg.js";
import { tex } from "../tex.js";

const U = [-1.85, 1.5]; // the view, in gaps from π

// TeX in the SVG, centred on (x, y) unless align is "start". A caption is
// text with TeX only for the spacing: "floats 2^-22 apart".
function label(parent, src, cls, align = "center") {
  const w = 300;
  const fo = el("foreignObject", { width: w, height: 44 }, parent);
  const div = document.createElement("div");
  div.className = `splitpi-label ${cls}`;
  div.style.textAlign = align;
  fo.appendChild(div);
  if (cls === "caption") {
    const t = document.createElement("span");
    tex(t, src);
    div.append("floats ", t, " apart");
  } else tex(div, src);
  return (x, y, opacity = 1) => set(fo, {
    x: align === "start" ? x : x - w / 2, y: y - 22,
    opacity, display: opacity > 0.001 ? "inline" : "none",
  });
}

// Two labels at one place, cross-fading from the first to the second as t
// goes 0 -> 1.
function swap(parent, src0, src1, cls) {
  const [a, b] = [label(parent, src0, cls), label(parent, src1, cls)];
  return (x, y, opacity, t) => { a(x, y, opacity * (1 - t)); b(x, y, opacity * t); };
}

// A bracket over [x0, x1] at height y: a line with end caps pointing down.
function bracket(parent, cls) {
  const path = el("path", { class: `bracket ${cls}` }, parent);
  return (x0, x1, y, opacity = 1) => set(path, {
    d: `M${x0},${y + 10}V${y}H${x1}V${y + 10}`, opacity,
  });
}

export default {
  defaults: { want: 0, drop: 0, miss: 0, lo: 0 },
  sticky: [],
  scrub: ["want", "drop", "miss", "lo"],

  create(root, data) {
    const W = +root.dataset.width || 700;
    const H = +root.dataset.height || 480;
    const s = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "fig" }, root);
    const [X0, X1] = [16, W - 16];
    const [yT, yB] = [162, 420];
    const sx = linear(U[0], U[1], X0, X1);
    const g = data.gap;
    const [uHi, uX, uLo] = [data.hi / g, data.x / g, data.lo / g];

    // --- top line: the floats near π
    el("line", { class: "axis", x1: X0, x2: X1, y1: yT, y2: yT }, s);
    for (let k = -4; k <= 3; k++) {
      const u = uHi + k;
      if (u < U[0] || u > U[1]) continue;
      el("line", { class: "float-tick", x1: sx(u), x2: sx(u), y1: yT - 14, y2: yT + 14 }, s);
    }
    el("line", { class: "float-tick hi", x1: sx(uHi), x2: sx(uHi), y1: yT - 18, y2: yT + 18 }, s);
    const loSeg = el("line", { class: "lo-seg", x1: sx(0), x2: sx(uHi), y1: yT, y2: yT }, s);
    el("line", { class: "pi-marker", x1: sx(0), x2: sx(0), y1: yT - 24, y2: yT + 24 }, s);
    el("circle", { class: "x-dot", cx: sx(uX), cy: yT, r: 8 }, s);
    label(s, data.tex_ulp_top, "caption", "start")(X0, yT - 52);
    label(s, "x", "")(sx(uX), yT + 40);
    label(s, "\\pi", "")(sx(0), yT + 40);
    const HI = ["\\mathrm{f32}(\\pi)", "\\pi_{hi}"], LO = ["\\delta", "\\pi_{lo}"];
    const hiName = swap(s, ...HI, "c-blue");
    const loTop = swap(s, ...LO, "c-orange");
    const wantTop = bracket(s, "ink");
    const wantTopLabel = label(s, "x - \\pi", "");
    const hiTop = bracket(s, "blue");
    const hiTopLabel = swap(s, `x - ${HI[0]}`, `x - ${HI[1]}`, "c-blue");

    // --- bottom line: near 0, floats everywhere at this scale
    const bottom = el("g", {}, s);
    el("line", { class: "axis", x1: X0, x2: X1, y1: yB, y2: yB }, bottom);
    let dense = "";
    for (let x = X0; x <= X1; x += 3) dense += `M${x},${yB - 9}V${yB + 9}`;
    el("path", { class: "dense-ticks", d: dense }, bottom);
    const bottomCaption = label(s, data.tex_ulp_bottom, "caption", "start");
    const zeroLabel = label(s, "0", "");

    // Falling copies and their guides.
    const guideX = el("line", { class: "guide" }, s);
    const guidePi = el("line", { class: "guide" }, s);
    const loSegBottom = el("line", { class: "lo-seg" }, s);
    const piImage = el("line", { class: "pi-marker" }, s);
    const zero = el("line", { class: "zero-marker" }, s);
    const xDot = el("circle", { class: "x-dot", r: 8 }, s);
    const loBottom = label(s, LO[0], "c-orange");
    const hiBottom = bracket(s, "blue");
    const hiBottomLabel = label(s, `x - ${HI[0]}`, "c-blue");
    const wantBottom = bracket(s, "ink");
    const wantBottomLabel = label(s, "x - \\pi", "");

    return {
      render({ want, drop, miss, lo }) {
        // Top: the two distances, then π_lo between π and π_hi.
        const rows = [yT - 126, yT - 74];  // x − π clear of x − f32(π)'s label
        wantTop(sx(uX), sx(0), rows[0], want);
        wantTopLabel(sx((uX + 0) / 2), rows[0] - 20, want);
        hiTop(sx(uX), sx(uHi), rows[1], want);
        const named = smooth(clamp01(lo));
        hiName(sx(uHi) + 10, yT + 42, 1, named);
        hiTopLabel(sx((uX + uHi) / 2), rows[1] - 20, want, named);
        set(loSeg, { opacity: miss });
        loTop(sx(uHi / 2), yT + 78, miss, named);

        // Bottom line fades in as x starts to fall. Its 0 starts under π_hi
        // and, subtracting π_lo, moves along it onto π.
        const shown = smooth(clamp01(drop * 3));
        const z = sx(uHi + uLo * lo);
        set(bottom, { opacity: shown });
        bottomCaption(X0, yB - 52, shown);
        set(zero, { x1: z, x2: z, y1: yB - 24, y2: yB + 24, opacity: shown });
        zeroLabel(z, yB + 40, shown);

        // x falls straight down: subtracting π_hi is exact.
        const xb = sx(uX);
        const fall = smooth(clamp01(drop));
        set(guideX, { x1: xb, x2: xb, y1: yT + 60, y2: lerp(yT, yB, fall) - 12, opacity: drop > 0 ? 0.9 : 0 });
        set(xDot, { cx: xb, cy: lerp(yT, yB, fall), opacity: drop > 0 ? 1 : 0 });
        const landed = smooth(clamp01((drop - 0.75) / 0.25));
        const merged = smooth(clamp01((lo - 0.8) / 0.2));
        hiBottom(z, xb, yB - 82, landed * (1 - merged));
        hiBottomLabel((z + xb) / 2, yB - 102, landed * (1 - merged));

        // π falls to π − π_hi: π_lo short of 0, as on the top line.
        const pfall = smooth(clamp01(miss));
        const pb = sx(0);
        const py = lerp(yT, yB, pfall);
        set(guidePi, { x1: pb, x2: pb, y1: yT + 60, y2: py - 24, opacity: miss > 0 ? 0.9 : 0 });
        set(piImage, { x1: pb, x2: pb, y1: py - 24, y2: py + 24, opacity: miss > 0 ? 1 : 0 });
        const piLanded = smooth(clamp01((miss - 0.75) / 0.25));
        set(loSegBottom, { x1: pb, x2: z, y1: yB, y2: yB, opacity: piLanded * (1 - merged) });
        loBottom((pb + z) / 2, yB + 40, piLanded * (1 - smooth(clamp01(lo * 2))));
        wantBottom(pb, xb, yB - 134, piLanded);
        wantBottomLabel((pb + xb) / 2, yB - 154, piLanded);
      },
    };
  },
};
