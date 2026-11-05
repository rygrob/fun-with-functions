// Tiny SVG helpers for the hand-rolled figures: elements, scales, paths,
// and the palette shared with examples/common/plots.py.

const NS = "http://www.w3.org/2000/svg";

// plots.SERIES, in the same fixed order, plus the neutral inks.
export const C = {
  blue: "#2a78d6",
  orange: "#eb6834",
  green: "#1baf7a",
  yellow: "#eda100",
  pink: "#e87ba4",
  ink: "#0b0b0b",
  muted: "#8a8985",
  faint: "#d9d8d4",
};

export function el(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(NS, tag);
  set(node, attrs);
  if (parent) parent.appendChild(node);
  return node;
}

export function set(node, attrs) {
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  return node;
}

export function svg(parent, width, height) {
  return el("svg", { viewBox: `0 0 ${width} ${height}`, width, height, class: "fig" }, parent);
}

// Maps [d0, d1] onto [r0, r1]; works for flipped ranges (SVG y grows down).
export function linear(d0, d1, r0, r1) {
  const k = (r1 - r0) / (d1 - d0);
  return (v) => r0 + (v - d0) * k;
}

// Missing points are skipped; with gaps, the line breaks there instead
// (tan's branches either side of a pole).
export function pathD(points, gaps = false) {
  let d = "";
  let pen = false;
  for (let i = 0; i < points.length; i++) {
    const [x, y] = points[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      if (gaps) pen = false;
      continue;
    }
    d += (pen ? "L" : "M") + x.toFixed(2) + "," + y.toFixed(2);
    pen = true;
  }
  return d;
}

// A pool of same-kind elements, grown or hidden as the count changes (N).
export function pool(parent, tag, attrs = {}) {
  const items = [];
  return (n) => {
    while (items.length < n) items.push(el(tag, attrs, parent));
    items.forEach((it, i) => it.setAttribute("display", i < n ? "inline" : "none"));
    return items.slice(0, n);
  };
}

export const lerp = (a, b, t) => a + (b - a) * t;
export const clamp01 = (t) => Math.min(1, Math.max(0, t));
export const smooth = (t) => t * t * (3 - 2 * t);
