// Deck runtime: reveal.js, KaTeX, equation highlights, and parameterized
// figures driven by fragments.
//
// Figures. A slide holds <div class="figure" data-figure="wrap"></div>.
// Each figure module exports { defaults, sticky, scrub, ints, create }:
//   defaults  every parameter's starting value
//   sticky    parameters fragments don't reset (N, mode): yours until a
//             fragment names them
//   scrub     numeric parameters, in order, that the scrub slider walks
//             through (each 0..1)
//   ints      { name: [min, max] } for the - / = keys and the stepper
//   tweenMs   optional: this figure's tween length (default TWEEN_MS)
//   create(root) -> { render(params) }
//
// A figure element's data-defaults='{"err": 1}' overrides its defaults on
// that slide only.
//
// Fragments set parameters. An empty <span class="fragment"
// data-params='{"wrap": 1}'></span> is an invisible step: the figure's
// target is defaults overlaid with every visible fragment's params, in
// order, and the figure tweens there on each click (jumps on slide change).
//
// Highlights. Wrap TeX in \htmlClass{hl name}{...}. A fragment with
// data-hl="name" lights every .hl.name on its slide; the last visible
// fragment with data-hl wins, so stepping on moves the emphasis along;
// data-hl="" turns it off again.
//
// Data. Numbers on slides come from slides/generate.py, which writes
// data/<name>.json. An element with data-src="<name>" gets that data: a .tex
// element with data-key="tex" takes its TeX from the JSON, a <code> with
// data-key takes its source (highlighted by reveal's plugin), a .text element
// with data-key takes plain text, and a figure is
// created with create(root, data). A dotted key reaches into lists and
// objects: data-key="constants.0.code".
//
// Animated algebra: in a <div class="swap flip"> stack, stepping forward
// slides the pieces tagged with \htmlData from one line to the next (flip.js).
//
// Debug: ?wrap=0.5&mode=track overrides parameters (for screenshots);
// ?still turns tweens off; ?contents opens the contents overlay; ?frag puts
// the click in the URL (#/18/0/3) so a screenshot can open on it.

import Reveal from "../node_modules/reveal.js/dist/reveal.mjs";
import Notes from "../node_modules/reveal.js/dist/plugin/notes.mjs";
import Highlight from "../node_modules/reveal.js/dist/plugin/highlight.mjs";
import Search from "../node_modules/reveal.js/dist/plugin/search.mjs";
import { tex } from "./tex.js";
import { flipStep } from "./flip.js";
import { applyBudgets, contents } from "./rehearse.js";
import plot from "./figures/plot.js";
import morph from "./figures/morph.js";
import numberline from "./figures/numberline.js";
import table from "./figures/table.js";
import floatbits, { floatzoom, floatfields } from "./figures/floatbits.js";
import nodes from "./figures/nodes.js";
import chebcircle from "./figures/chebcircle.js";
import wrap from "./figures/wrap.js";
import angles from "./figures/angles.js";
import interp from "./figures/interp.js";
import zoompi from "./figures/zoompi.js";
import fold from "./figures/fold.js";
import splitpi from "./figures/splitpi.js";
import foldline from "./figures/foldline.js";
import errline from "./figures/errline.js";

// Figure modules from js/figures/, by data-figure name.
const FIGURES = { plot, morph, numberline, table, floatbits, floatzoom, floatfields, nodes, chebcircle, wrap, angles, zoompi, interp, fold, splitpi, foldline, errline };
const TWEEN_MS = 1400;
const query = new URLSearchParams(location.search);
const still = query.has("still");

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

const DATA = {};

async function loadData(root) {
  const names = new Set([...root.querySelectorAll("[data-src]")].map((n) => n.dataset.src));
  await Promise.all([...names].map(async (name) => {
    const r = await fetch(`data/${name}.json`);
    if (!r.ok) throw new Error(`data/${name}.json: ${r.status} (run slides/generate.py)`);
    DATA[name] = await r.json();
  }));
  root.querySelectorAll(".tex[data-src][data-key], code[data-src][data-key], .text[data-src][data-key]").forEach((n) => {
    n.textContent = n.dataset.key.split(".").reduce((v, k) => v[k], DATA[n.dataset.src]);
  });
}

// ---------------------------------------------------------------------------
// TeX
// ---------------------------------------------------------------------------

function renderTex(root) {
  root.querySelectorAll(".tex").forEach((n) => tex(n, n.textContent, n.classList.contains("display")));
}

// ---------------------------------------------------------------------------
// Highlights
// ---------------------------------------------------------------------------

function updateHighlights(slide) {
  if (!slide) return;
  const lit = [...slide.querySelectorAll(".fragment.visible[data-hl]")].pop();
  const names = lit ? lit.dataset.hl.split(/\s+/).filter(Boolean) : [];
  slide.querySelectorAll(".hl").forEach((n) => {
    n.classList.toggle("on", names.some((name) => n.classList.contains(name)));
  });
  slide.classList.toggle("has-hl", names.length > 0);
}

// Callouts: a label under a highlighted piece of maths, <div class="callout"
// data-to="name"> inside a .callouts box, centred under the slide's .hl.name
// in the TeX. Measured on the live slide (hidden slides have no layout).
function placeCallouts(slide) {
  if (!slide) return;
  const scale = deck.getScale();
  slide.querySelectorAll(".callout[data-to]").forEach((c) => {
    const target = slide.querySelector(`.katex .hl.${c.dataset.to}`);
    if (!target) return;
    const t = target.getBoundingClientRect();
    const box = c.parentElement.getBoundingClientRect();
    c.style.left = `${(t.left + t.width / 2 - box.left) / scale}px`;
  });
}

// ---------------------------------------------------------------------------
// Figures
// ---------------------------------------------------------------------------

const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

class Figure {
  constructor(root, spec, data) {
    // data-defaults on the element overrides the module's defaults for this
    // instance (where the slide starts, before any fragment).
    this.spec = { ...spec, defaults: { ...spec.defaults, ...JSON.parse(root.dataset.defaults || "{}") } };
    spec = this.spec;
    this.params = { ...spec.defaults };
    this.view = spec.create(root, data);
    this.anim = null;
    this.buildControls(root);
    this.render();
  }

  render() {
    this.view.render(this.params);
    if (this.scrubInput) this.scrubInput.value = this.scrubValue();
    if (this.readouts) for (const [k, n] of Object.entries(this.readouts)) n.textContent = this.params[k];
  }

  // The fragment-driven target: defaults, then each visible fragment's params.
  // A sticky parameter keeps its manual value (- / =), unless one of the
  // slide's fragments sets it: then the clicks own it, forwards and back.
  target(slide) {
    const t = {};
    const clicked = new Set();
    slide.querySelectorAll(".fragment[data-params]").forEach((f) => Object.keys(JSON.parse(f.dataset.params)).forEach((k) => clicked.add(k)));
    for (const [k, v] of Object.entries(this.spec.defaults)) {
      t[k] = this.spec.sticky.includes(k) && !clicked.has(k) ? this.params[k] : v;
    }
    slide.querySelectorAll(".fragment.visible[data-params]").forEach((f) => Object.assign(t, JSON.parse(f.dataset.params)));
    for (const k of Object.keys(t)) {
      if (query.has(k)) t[k] = isNaN(+query.get(k)) ? query.get(k) : +query.get(k);
    }
    return t;
  }

  go(target, animate) {
    cancelAnimationFrame(this.anim);
    const from = { ...this.params };
    const numeric = Object.keys(target).filter((k) => typeof target[k] === "number" && !this.spec.sticky.includes(k));
    // Sticky and non-numeric parameters switch at once.
    for (const k of Object.keys(target)) if (!numeric.includes(k)) this.params[k] = target[k];
    if (!animate || still || numeric.every((k) => from[k] === target[k])) {
      Object.assign(this.params, target);
      this.render();
      return;
    }
    const t0 = performance.now();
    const step = (now) => {
      const u = ease(Math.min(1, (now - t0) / (this.spec.tweenMs || TWEEN_MS)));
      for (const k of numeric) this.params[k] = from[k] + (target[k] - from[k]) * u;
      this.render();
      if (u < 1) this.anim = requestAnimationFrame(step);
    };
    this.anim = requestAnimationFrame(step);
  }

  set(changes) {
    cancelAnimationFrame(this.anim);
    Object.assign(this.params, changes);
    this.render();
  }

  // --- manual controls: a scrub slider over spec.scrub, and steppers for ints

  scrubValue() {
    return this.spec.scrub.reduce((s, k) => s + this.params[k], 0);
  }

  setScrub(v) {
    const changes = {};
    this.spec.scrub.forEach((k, i) => (changes[k] = Math.min(1, Math.max(0, v - i))));
    this.set(changes);
  }

  stepInt(key, delta) {
    const [lo, hi] = this.spec.ints[key];
    this.set({ [key]: Math.min(hi, Math.max(lo, this.params[key] + delta)) });
  }

  buildControls(root) {
    const bar = document.createElement("div");
    bar.className = "fig-controls";
    this.readouts = {};
    for (const key of Object.keys(this.spec.ints || {})) {
      const g = document.createElement("span");
      g.className = "stepper";
      g.innerHTML = `<button>−</button><span class="name">${key}</span><span class="val"></span><button>+</button>`;
      const [minus, plus] = g.querySelectorAll("button");
      minus.onclick = () => this.stepInt(key, -1);
      plus.onclick = () => this.stepInt(key, +1);
      this.readouts[key] = g.querySelector(".val");
      bar.appendChild(g);
    }
    if (this.spec.scrub.length) {
      const input = document.createElement("input");
      Object.assign(input, { type: "range", min: 0, max: this.spec.scrub.length, step: 0.001 });
      input.oninput = () => this.setScrub(+input.value);
      this.scrubInput = input;
      bar.appendChild(input);
    }
    for (const [key, values] of Object.entries(this.spec.choices || {})) {
      const b = document.createElement("button");
      b.className = "choice";
      b.onclick = () => {
        this.set({ [key]: values[(values.indexOf(this.params[key]) + 1) % values.length] });
      };
      this.readouts[key] = b;
      bar.appendChild(b);
    }
    // Keep reveal from treating clicks and drags on the controls as navigation.
    bar.addEventListener("pointerdown", (e) => e.stopPropagation());
    root.appendChild(bar);
  }
}

const figures = new Map(); // slide -> [Figure], in document order

// The slide's first figure: the one the - / = / M keys drive.
function figureFor(slide) {
  return slide && figures.has(slide) ? figures.get(slide)[0] : null;
}

function sync(slide, animate) {
  updateHighlights(slide);
  placeCallouts(slide);
  for (const fig of (slide && figures.get(slide)) || []) fig.go(fig.target(slide), animate);
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

await loadData(document);
renderTex(document);
document.querySelectorAll(".figure[data-figure]").forEach((root) => {
  const data = DATA[root.dataset.src];
  const slide = root.closest("section");
  if (!figures.has(slide)) figures.set(slide, []);
  figures.get(slide).push(new Figure(root, FIGURES[root.dataset.figure], data));
});

applyBudgets(document); // section budgets -> data-timing, for the speaker view's pacing (rehearse.js)

// Slide numbers for review: the 0-based index, as in the URL hash and the T
// contents. I toggles them and the choice persists (turn off for the talk).
const numberOf = (slide) => [String(deck.getIndices(slide).h)];
function numbers(on) {
  try {
    if (on !== undefined) localStorage.setItem("slideNumbers", on ? "1" : "0");
    return localStorage.getItem("slideNumbers") !== "0";
  } catch { return true; }
}

const deck = new Reveal({
  width: 1280,
  height: 720,
  margin: 0.04,
  center: false,
  hash: true,
  fragmentInUrl: query.has("frag"),
  controls: false,
  progress: true,
  slideNumber: numbers() ? numberOf : false,
  transition: "fade",
  transitionSpeed: "fast",
  plugins: [Notes, Highlight, Search],
  // A line break inside a span (highlight.js runs a function's signature on
  // to its "{" when it doesn't know the leading macro, FP_FORCEINLINE) leaves
  // a stray space before the brace once lines are numbered: end it first.
  highlight: {
    beforeHighlight: (hljs) => hljs.addPlugin({
      "after:highlight": (r) => { r.value = r.value.replace(/(\n[ \t]*)<\/span>/g, "</span>$1"); },
    }),
  },
});

deck.on("ready", (e) => sync(e.currentSlide, false));
deck.on("slidechanged", (e) => sync(e.currentSlide, false));
deck.on("fragmentshown", (e) => {
  sync(deck.getCurrentSlide(), true);
  const f = e.fragment;
  const prev = f && f.previousElementSibling;
  if (!still && f.parentElement.classList.contains("flip") && prev && prev.classList.contains("fragment")) {
    flipStep(prev, f);
  }
});
deck.on("fragmenthidden", () => sync(deck.getCurrentSlide(), true));

// - / = step the current figure's first integer parameter (N); m cycles its
// first choice (mode).
const current = () => figureFor(deck.getCurrentSlide());
deck.addKeyBinding({ keyCode: 189, key: "-", description: "Figure: N − 1" }, () => {
  const f = current();
  if (f && f.spec.ints) f.stepInt(Object.keys(f.spec.ints)[0], -1);
});
deck.addKeyBinding({ keyCode: 187, key: "=", description: "Figure: N + 1" }, () => {
  const f = current();
  if (f && f.spec.ints) f.stepInt(Object.keys(f.spec.ints)[0], +1);
});
deck.addKeyBinding({ keyCode: 77, key: "M", description: "Figure: cycle mode" }, () => {
  const f = current();
  const key = f && f.spec.choices && Object.keys(f.spec.choices)[0];
  if (key) f.readouts[key].click();
});

// T: the contents overlay, every slide by section with its budget (rehearse.js).
const toggleContents = contents(deck, document);
deck.addKeyBinding({ keyCode: 84, key: "T", description: "Contents: slides by section, with budgets" }, toggleContents);
if (query.has("contents")) deck.on("ready", toggleContents); // ?contents opens it (screenshots)

// I: slide numbers on / off (N is Reveal's next slide).
deck.addKeyBinding({ keyCode: 73, key: "I", description: "Slide numbers on / off" }, () => {
  const on = !numbers();
  numbers(on);
  deck.configure({ slideNumber: on ? numberOf : false });
});

deck.initialize();
