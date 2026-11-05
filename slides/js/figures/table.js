// A results table from a generate.py item: one row per method.
//
// Data: { rows: [{ key, ...fields }] }. Columns and rows come from the element:
//   data-cols='[["label", "Method"], ["max_ulp", "Max ULP", "sci"], ...]'
//   data-rows='["std::sin(x)", "taylor_sin_13"]'   (keys, in order; default all)
//   data-new='["pade_sin_7_6"]'                    (rows to highlight)
// Parameter: shown, how many rows have their contents in, in order (default
// all); the header and the rules are there from the start, so the table's
// shape is. A fragment's data-params='{"shown": 2}' fills it a click at a time.
// Formats: "sci" (2.976e9 -> 3.0 × 10⁹, 0.00025 -> 2.5 × 10⁻⁴), "x"
// (15.9 -> 15.9×, 3 -> 3.0×), "dB", "dBc", "ns" (two decimals), "code", or plain. A null
// value shows as a dash. Plain text that starts with a std:: call sets the call
// in code ("std::sqrt (compiler)" -> <code>std::sqrt</code> (compiler)).
//
// <div class="figure" data-figure="table" data-src="sin-scoreboard" data-cols='...'></div>

function cell(v, fmt) {
  if (v === null || v === undefined) return { text: "—" };
  if (fmt === "sci") {
    if (v < 1000 && v >= 0.01) return { text: String(v) };
    const [m, e] = v.toExponential(1).split("e");
    return { html: `${m} × 10<sup>${String(+e).replace("-", "−")}</sup>` };
  }
  if (fmt === "x") return { text: `${v.toFixed(1)}×` };
  if (fmt === "dB") return { text: `${String(v).replace("-", "−")} dB` };
  if (fmt === "dBc") return { text: `${String(v).replace("-", "−")} dBc` };
  if (fmt === "code") return { code: v };
  if (fmt === "ns") return { text: v.toFixed(2) };
  const std = String(v).match(/^(std::\w+(?:\([^)]*\))?)(.*)$/);
  if (std) return { code: std[1], text: std[2] };
  return { text: String(v) };
}

export default {
  defaults: { shown: 99 },
  sticky: [],
  scrub: [],

  create(root, data) {
    const cols = JSON.parse(root.dataset.cols);
    const table = document.createElement("table");
    table.className = "results";
    const head = table.createTHead().insertRow();
    for (const [, title] of cols) {
      const th = document.createElement("th");
      th.textContent = title;
      head.appendChild(th);
    }
    const body = table.createTBody();
    const keys = root.dataset.rows ? JSON.parse(root.dataset.rows) : data.rows.map((r) => r.key);
    const fresh = JSON.parse(root.dataset.new || "[]");
    for (const key of keys) {
      const row = data.rows.find((r) => r.key === key);
      const tr = body.insertRow();
      if (fresh.includes(key)) tr.className = "new";
      for (const [key, , fmt] of cols) {
        const td = tr.insertCell().appendChild(document.createElement("span"));
        const c = cell(row[key], fmt);
        if (c.html) td.innerHTML = c.html;
        else if (c.code) {
          const code = document.createElement("code");
          code.textContent = c.code;
          td.append(code, c.text || "");
        } else td.textContent = c.text;
      }
    }
    root.appendChild(table);
    const fade = (n, o) => (n.style.opacity = Math.min(1, Math.max(0, o)));
    return {
      render({ shown }) {
        [...body.rows].forEach((tr, i) => {
          for (const td of tr.cells) fade(td.firstChild, shown - i);
        });
      },
    };
  },
};
