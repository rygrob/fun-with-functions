# Slides

reveal.js + KaTeX + hand-rolled SVG figures. No build step; everything is
local (`npm install` once), so it runs without wifi.

```sh
cd slides && npm install && npm run serve    # http://localhost:8026
python3 slides/generate.py                   # from the repo root: data/*.json
```

## Keys

- → / space: next step (fragments drive the animations)
- `-` / `=`: change N on the current figure; `M`: cycle the figure's mode
- `S`: speaker view, `F`: fullscreen, `Esc`: overview
- Hover the bottom-right of a figure for its controls (N stepper, scrub
  slider, mode button)

## Authoring

- **Numbers come from Python.** Anything computed (coefficients, curves,
  error figures) is produced by `generate.py`, from the same
  `examples/common` code as the examples, into `data/<name>.json`, and
  committed. Run `python3 slides/generate.py` after changing it. On a slide:
  `<div class="tex display" data-src="cheb-sin-5" data-key="tex"></div>`
  for generated TeX, `<div class="figure" data-figure="plot"
  data-src="cheb-sin-5"></div>` for a plot of its curves.
- **Math:** `<span class="tex">...</span>`, or add `display` for a block.
- **Code:** `<code>sin</code>` for function names as you'd call them.
- **Plots:** `data-figure="plot"` draws a generated item's `plot` (curves,
  ticks, shaded band, TeX legend); size with `data-width` / `data-height`.
- **Layout:** `split` (list beside a picture or math), `cols` (two lists),
  `sub` (subtitle under the title).
- **Highlights:** wrap TeX in `\htmlClass{hl name c-blue}{...}`; a fragment
  with `data-hl="name"` lights it and grays the rest of the displayed math.
  The last visible fragment with `data-hl` wins.
- **Animations:** `<div class="figure" data-figure="name"></div>`, then empty
  fragments that set parameters: `<span class="fragment"
  data-params='{"wrap": 1}'></span>`. Each click tweens to defaults overlaid
  with every visible fragment's params. Figures live in `js/figures/`; see
  the contract at the top of `js/main.js`.
- **Debug / screenshots:** `?still` disables tweens; any parameter can be
  forced from the URL, e.g. `?still&wrap=0.5#/1`.
- **Serving:** `serve.py` sends `Cache-Control: no-store`; plain
  `http.server` let Safari keep stale CSS.

Colors match `examples/common/plots.py`.

## Publishing

`python3 slides/build.py` copies the deck into `slides/dist/`, with only the
library files it uses, in `vendor/` rather than `node_modules/` (hosts tend to
skip anything called that). `dist/` is static: serve it from anywhere.
