// θ wraps onto the unit circle; cos(Nθ) becomes T_N(x).
//
// Right pane: the θ axis [0, π], with N equally spaced angles
// θ_k = (k + ½)π/N marked on it. As `wrap` goes 0 → 1 the axis curls
// counterclockwise onto the upper unit semicircle, 0 pinned at (1, 0), π
// ending at (−1, 0). Arc length is preserved throughout -- θ is distance
// along the axis before, angle after. As `project` goes 0 → 1 the angles
// drop onto the x axis: x_k = cos θ_k, the Chebyshev points.
//
// Left pane: y = cos(Nθ), whose zeros are exactly the θ_k. As θ is replaced
// by x = cos θ the same curve becomes y = T_N(x).
//
// x = cos θ runs backwards (θ = 0 is x = 1), so with the θ axis heading
// right the left pane has to turn around somewhere: its θ axis turns over
// during the wrap (squeezing through its midpoint, like a card), then
// θ → cos θ is a smooth stretch. (Two other ways, tracking the right pane's
// x exactly and bending the axis left like a rainbow, are in git history.)

import { C, el, set, svg, linear, pathD, pool, lerp, clamp01 } from "../svg.js";
import { tex } from "../tex.js";

const PI = Math.PI;
const SAMPLES = 400;

// Point at arc length u along the wrapping axis, and its direction angle.
// Constant curvature s, initial heading s·π/2, starting at (1, 0): a straight
// line heading right at s = 0, the upper unit semicircle at s = 1.
function axisPoint(u, s) {
  const phi0 = (s * PI) / 2;
  const phi = phi0 + s * u;
  if (s < 1e-6) return [1 + u * Math.cos(phi0), u * Math.sin(phi0), phi];
  return [1 + (Math.sin(phi) - Math.sin(phi0)) / s, (Math.cos(phi0) - Math.cos(phi)) / s, phi];
}

// World-space bounds over the whole animation (the curl rises to y ≈ 2.8
// midway).
const BOUNDS = { x0: -1.25, x1: 1 + PI + 0.35, y0: -0.6, y1: 2.95 };

const nodeAngles = (N) => Array.from({ length: N }, (_, k) => ((k + 0.5) * PI) / N);

export default {
  defaults: { wrap: 0, project: 0, N: 5 },
  sticky: ["N"],
  scrub: ["wrap", "project"],
  ints: { N: [1, 16] },

  create(root) {
    root.classList.add("wrap-figure");
    const panes = document.createElement("div");
    panes.className = "panes";
    root.prepend(panes);

    // --- left: the graph -------------------------------------------------
    const left = document.createElement("div");
    left.className = "pane";
    panes.appendChild(left);
    const lLabels = document.createElement("div");
    lLabels.className = "pane-label stack";
    left.appendChild(lLabels);
    const lCos = document.createElement("div");
    const lT = document.createElement("div");
    lLabels.append(lCos, lT);

    const LW = 560, LH = 400;
    const ls = svg(left, LW, LH);
    const hx = linear(-1, 1, 50, LW - 20);
    const vy = linear(-1, 1, LH - 45, 25);
    el("line", { x1: hx(-1.04), x2: hx(1.04), y1: vy(0), y2: vy(0), class: "axis" }, ls);
    for (const v of [-1, 1]) {
      el("line", { x1: hx(-1.04), x2: hx(1.04), y1: vy(v), y2: vy(v), class: "grid" }, ls);
      el("text", { x: hx(-1) - 30, y: vy(v) + 5, class: "tick", text: v > 0 ? "1" : "−1" }, ls);
    }
    const curve = el("path", { class: "curve", stroke: C.blue }, ls);
    const lDots = pool(el("g", {}, ls), "circle", { r: 5.5, fill: C.orange });
    const thetaTicks = [
      [0, "0"], [PI / 2, "π/2"], [PI, "π"],
    ].map(([t, s]) => [t, el("text", { y: vy(0) + 26, class: "tick", text: s }, ls)]);
    const xTicks = [-1, 0, 1].map((v) => el("text", { x: hx(v), y: vy(0) + 26, class: "tick", text: v < 0 ? "−1" : String(v) }, ls));
    const lAxisName = el("text", { x: hx(1.04) + 4, y: vy(0) + 5, class: "axis-name" }, ls);

    // --- right: the wrapping axis ----------------------------------------
    const right = document.createElement("div");
    right.className = "pane";
    panes.appendChild(right);
    const rLabels = document.createElement("div");
    rLabels.className = "pane-label stack";
    right.appendChild(rLabels);
    const rTheta = document.createElement("div");
    const rX = document.createElement("div");
    rLabels.append(rTheta, rX);
    tex(rX, String.raw`\text{zeros at } x_k = \cos\theta_k`);

    const RW = 640, RH = LH;
    const rs = svg(right, RW, RH);
    // Camera: fit the whole animation's bounds, centred.
    const b = BOUNDS;
    const k = Math.min(RW / (b.x1 - b.x0), RH / (b.y1 - b.y0));
    const ox = (RW - k * (b.x1 - b.x0)) / 2, oy = (RH - k * (b.y1 - b.y0)) / 2;
    const wx = (x) => ox + (x - b.x0) * k;
    const wy = (y) => RH - oy - (y - b.y0) * k;

    const xAxis = el("g", {}, rs);
    const xAxisLine = el("line", { class: "axis" }, xAxis);
    const xAxisTicks = [-1, 0, 1].map((v) => [
      v,
      el("line", { class: "axis" }, xAxis),
      el("text", { class: "tick", text: v < 0 ? "−1" : String(v) }, xAxis),
    ]);
    const xAxisName = el("text", { class: "axis-name", text: "x" }, xAxis);
    const circle = el("path", { class: "guide" }, rs);
    function placeStatics() {
      set(xAxisLine, { x1: wx(-1.08), x2: wx(1.08), y1: wy(0), y2: wy(0) });
      for (const [v, ln, t] of xAxisTicks) {
        set(ln, { x1: wx(v), x2: wx(v), y1: wy(0) - 5, y2: wy(0) + 5 });
        set(t, { x: wx(v), y: wy(0) + 26 });
      }
      set(xAxisName, { x: wx(1.08) + 6, y: wy(0) + 5 });
      circle.setAttribute("d", pathD(Array.from({ length: 121 }, (_, i) => [wx(Math.cos((i * PI) / 120)), wy(Math.sin((i * PI) / 120))])));
    }
    const spokes = pool(el("g", {}, rs), "line", { class: "spoke" });
    const drops = pool(el("g", {}, rs), "line", { class: "drop" });
    const axis = el("path", { class: "theta-axis" }, rs);
    const cellTicks = pool(el("g", {}, rs), "line", { class: "axis" });
    const rDots = pool(el("g", {}, rs), "circle", { r: 5.5, fill: C.orange });
    const fallDots = pool(el("g", {}, rs), "circle", { r: 5.5, fill: C.orange });
    const axisLabels = [
      [0, "0"], [PI / 2, "π/2"], [PI, "π"],
    ].map(([u, s]) => [u, el("text", { class: "tick", text: s }, rs)]);
    const thetaName = el("text", { class: "axis-name", text: "θ" }, rs);
    placeStatics();

    let lastN = null;

    return {
      render({ wrap: s, project: p, N }) {
        s = clamp01(s);
        p = clamp01(p);
        const th = nodeAngles(N);
        const at = axisPoint;
        // Labels swap in sequence (old out, then new in), never overlapping.
        const outA = clamp01(1 - 2 * p), inB = clamp01(2 * p - 1);

        if (N !== lastN) {
          tex(lCos, String.raw`y = \cos(${N}\theta)`);
          tex(lT, String.raw`y = T_{${N}}(x)`);
          tex(rTheta, String.raw`\text{zeros at } \theta_k = \tfrac{(k + \frac12)\,\pi}{${N}}`);
          lastN = N;
        }
        lCos.style.opacity = outA;
        lT.style.opacity = inB;
        rTheta.style.opacity = outA;
        rX.style.opacity = inB;

        // Left pane's horizontal coordinate for angle θ, in [-1, 1].
        const h = (t) => lerp(lerp((2 * t) / PI - 1, 1 - (2 * t) / PI, s), Math.cos(t), p);

        const pts = [];
        for (let i = 0; i <= SAMPLES; i++) {
          const t = (i * PI) / SAMPLES;
          pts.push([hx(h(t)), vy(Math.cos(N * t))]);
        }
        curve.setAttribute("d", pathD(pts));
        lDots(N).forEach((d, i) => set(d, { cx: hx(h(th[i])), cy: vy(0) }));
        for (const [t, node] of thetaTicks) set(node, { x: hx(h(t)), opacity: outA });
        xTicks.forEach((n) => n.setAttribute("opacity", inB));
        set(lAxisName, { text: p > 0.5 ? "x" : "θ", opacity: Math.max(outA, inB) });

        // Right pane.
        const axisPts = [];
        for (let i = 0; i <= SAMPLES; i++) {
          const [X, Y] = at((i * PI) / SAMPLES, s);
          axisPts.push([wx(X), wy(Y)]);
        }
        axis.setAttribute("d", pathD(axisPts));
        circle.setAttribute("opacity", 0.9 * s);
        xAxis.setAttribute("opacity", clamp01(3 * p));

        cellTicks(N + 1).forEach((ln, i) => {
          const [X, Y, phi] = at((i * PI) / N, s);
          const nx = Math.sin(phi) * 6, ny = -Math.cos(phi) * 6; // outward normal
          set(ln, { x1: wx(X) - nx, y1: wy(Y) + ny, x2: wx(X) + nx, y2: wy(Y) - ny });
        });
        const spokeEls = spokes(N), dropEls = drops(N), fall = fallDots(N);
        rDots(N).forEach((d, i) => {
          const [X, Y] = at(th[i], s);
          set(d, { cx: wx(X), cy: wy(Y), opacity: lerp(1, 0.35, p) });
          set(spokeEls[i], { x1: wx(0), y1: wy(0), x2: wx(X), y2: wy(Y), opacity: s * s * (1 - p) });
          const yDrop = Y * (1 - p);
          set(dropEls[i], { x1: wx(X), y1: wy(Y), x2: wx(X), y2: wy(yDrop), opacity: p > 0 ? 1 : 0 });
          set(fall[i], { cx: wx(X), cy: wy(yDrop), opacity: p > 0 ? 1 : 0 });
        });
        for (const [u, node] of axisLabels) {
          const [X, Y, phi] = at(u, s);
          const off = 24;
          set(node, { x: wx(X) + Math.sin(phi) * off, y: wy(Y) + Math.cos(phi) * off + 5, opacity: outA });
        }
        const [X, Y, phi] = at(PI, s);
        set(thetaName, {
          x: wx(X) + Math.cos(phi) * 22, y: wy(Y) - Math.sin(phi) * 22 + 6, opacity: outA,
        });
      },
    };
  },
};
