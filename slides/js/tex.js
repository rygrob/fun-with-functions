// KaTeX with \htmlClass allowed (trust), for the highlight mechanism.

import katex from "../node_modules/katex/dist/katex.mjs";

export function tex(node, src, display = false) {
  katex.render(src, node, { displayMode: display, trust: true, strict: false, throwOnError: false });
}
