// Animated algebra for a <div class="swap flip"> stack of equations: when a
// click steps from one line to the next, the pieces tagged in the TeX glide
// from where they were to where they are now, instead of the two lines just
// crossfading. Pieces keep their colour in flight (orange, say), rest in
// place, then hand over to the new line, in its colour.
//
// Tags, with KaTeX's \htmlData (trust is on):
//   m=key      the same piece in both lines. Two copies travel together
//              from the old position to the new: the old form fading out,
//              the new form fading in (u^2/5! becomes u/5! on the way). The
//              new form ends exactly on the new line, so the handover is a
//              pure colour fade with nothing shifting
//   fly=key    the new line's m=key flies in from this piece, growing or
//              shrinking from its size: a factor pulled out to the front
//   drop=1     with fly: the piece leaves its old place (the u of u/3!),
//              rather than being copied (one u out of u^2)
// Pieces are matched by the bottom centre of their boxes. Untagged parts
// crossfade as usual (reveal's own fragment fade). A forward step where
// nothing moves (marking terms) switches instantly. Only forward steps
// animate; going back just crossfades.
//
// main.js calls flipStep(previous line, new line) on fragmentshown.

const MOVE_MS = 1100;
const HOLD_MS = 700;   // pieces rest in place, still in their colour, before handing over
const LAND_MS = 500;
const MORPH = 0.45;    // the old form has faded into the new by this fraction of the move
const EASE = "cubic-bezier(0.65, 0, 0.35, 1)";

let finish = null; // ends the running step at once (a fast click)

export function flipStep(from, to) {
  if (finish) finish();
  const box = to.parentElement;
  const origin = box.getBoundingClientRect();
  const scale = origin.width / box.offsetWidth || 1; // reveal scales the slide
  const corner = (n) => {
    const r = n.getBoundingClientRect();
    return { x: (r.left - origin.left) / scale, y: (r.top - origin.top) / scale };
  };
  const anchor = (n) => {
    const r = n.getBoundingClientRect();
    return { x: ((r.left + r.right) / 2 - origin.left) / scale, y: (r.bottom - origin.top) / scale };
  };

  const targets = {};
  to.querySelectorAll("[data-m]").forEach((n) => (targets[n.dataset.m] = n));

  // A step where nothing moves (marking terms, say) switches at once, so the
  // click lands as you say it.
  const moves = [...from.querySelectorAll("[data-m]")].some((src) => {
    const dst = targets[src.dataset.m];
    if (!dst) return false;
    const a = anchor(src), b = anchor(dst);
    return Math.hypot(b.x - a.x, b.y - a.y) > 0.5;
  }) || [...from.querySelectorAll("[data-fly]")].some((src) => targets[src.dataset.fly]);
  if (!moves) {
    snap(from, to);
    return;
  }

  const layer = document.createElement("div");
  layer.className = "flip-layer";
  box.appendChild(layer);
  const landing = new Set();
  const anims = [];

  // A copy of n, laid exactly over it, in the colour pieces travel in.
  const place = (n, travel) => {
    const piece = document.createElement("span");
    piece.className = "katex flip-piece";
    piece.style.fontSize = getComputedStyle(n).fontSize;
    piece.style.color = travel;
    const copy = n.cloneNode(true);
    copy.style.opacity = "";
    piece.appendChild(copy);
    layer.appendChild(piece);
    const a = corner(n);
    piece.style.left = `${a.x}px`;
    piece.style.top = `${a.y}px`;
    const c = corner(copy); // correct for the copy's own baseline offset
    piece.style.left = `${2 * a.x - c.x}px`;
    piece.style.top = `${2 * a.y - c.y}px`;
    return { piece, copy };
  };
  const move = (piece, frames) => anims.push(piece.animate(frames, { duration: MOVE_MS, easing: EASE, fill: "forwards" }));

  from.querySelectorAll("[data-m]").forEach((src) => {
    const dst = targets[src.dataset.m];
    if (!dst) return;
    const travel = getComputedStyle(src).color;
    const a = anchor(src), b = anchor(dst);
    const d = `${b.x - a.x}px, ${b.y - a.y}px`;
    // The old form: slides off and fades out; anything that flies away has
    // its own copy, so it leaves this one at once.
    const old = place(src, travel);
    old.copy.querySelectorAll("[data-fly][data-drop]").forEach((n) => (n.style.visibility = "hidden"));
    move(old.piece, [
      { transform: "translate(0, 0)", opacity: 1 },
      { opacity: 0, offset: MORPH },
      { transform: `translate(${d})`, opacity: 0 },
    ]);
    // The new form: starts where the old one is, ends exactly in place.
    const fresh = place(dst, travel);
    move(fresh.piece, [
      { transform: `translate(${a.x - b.x}px, ${a.y - b.y}px)`, opacity: 0 },
      { opacity: 1, offset: MORPH },
      { transform: "translate(0, 0)", opacity: 1 },
    ]);
    landing.add(dst);
  });

  const arrived = {}; // fly copies per target: all but the first fade as they land
  from.querySelectorAll("[data-fly]").forEach((src) => {
    const dst = targets[src.dataset.fly];
    if (!dst) return;
    const extra = (arrived[src.dataset.fly] = (arrived[src.dataset.fly] || 0) + 1) > 1;
    // A copy of the landing piece (the front u), starting over src at src's
    // size, so it ends exactly in place. Scale about the bottom centre,
    // which is what's matched.
    const { piece, copy } = place(dst, getComputedStyle(src).color);
    const p = corner(piece), a = anchor(src), b = anchor(copy);
    piece.style.transformOrigin = `${b.x - p.x}px ${b.y - p.y}px`;
    const k = parseFloat(getComputedStyle(src).fontSize) / parseFloat(getComputedStyle(dst).fontSize);
    const start = `translate(${a.x - b.x}px, ${a.y - b.y}px) scale(${k})`;
    const end = "translate(0, 0) scale(1)";
    move(piece, extra
      ? [{ transform: start, opacity: 1 }, { opacity: 1, offset: 0.75 }, { transform: end, opacity: 0 }]
      : [{ transform: start }, { transform: end }]);
    landing.add(dst);
  });

  // The new line's landed pieces stay hidden until their copies hand over.
  landing.forEach((n) => (n.style.opacity = "0"));

  let timer = setTimeout(land, MOVE_MS + HOLD_MS);
  function land() {
    clearTimeout(timer);
    landing.forEach((n) => {
      n.style.transition = `opacity ${LAND_MS}ms`;
      n.style.opacity = "";
    });
    const out = layer.animate([{ opacity: 1 }, { opacity: 0 }], { duration: LAND_MS, fill: "forwards" });
    timer = setTimeout(done, LAND_MS);
    out.onfinish = done;
  }
  function done() {
    clearTimeout(timer);
    anims.forEach((a) => a.cancel());
    layer.remove();
    landing.forEach((n) => {
      n.style.transition = "";
      n.style.opacity = "";
    });
    finish = null;
  }
  finish = done;
}

// Cut reveal's fragment fade short: both lines jump to their end state.
function snap(...lines) {
  lines.forEach((n) => {
    n.style.transition = "none";
    void n.offsetWidth; // apply now, ending the fade in progress
  });
  requestAnimationFrame(() => lines.forEach((n) => (n.style.transition = "")));
}
