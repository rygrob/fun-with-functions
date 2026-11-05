# Tables vs. math as tables multiply

Written before the bench moved to projection fits and to log₂ degree 7, 2^x degree
5; the polynomials' timings are close, but rerun rather than quote.

`bench/build/adc_bench tables` (src/tables.hpp), then
`python bench/plot_tables.py` (writes `bench/out/tables.png` and
`tables_slowdown.png`). Apple M2 Pro: 128 KB L1D and 16 MB L2 per P-cluster,
128-byte lines. On AC power; one machine, so indicative. Full sweep ≈ 13 s.

## The workload

- **K instances** (voices or plug-in instances), each running a mixed chain
  per sample: **sin oscillator → tanh waveshaper → dB-to-gain envelope →
  sum into a bus**.
- **Table:** every instance owns three float LUTs of N + 1 entries, read with
  linear interpolation. Footprint = K × 3 × (N + 1) × 4 bytes.
- **Math:** the same chain with `cheb_turns_sin_13`, `sigmoid_tanh_9` and
  `db_to_gain_3`, either vectorized across the block (as the compiler
  likes) or scalar (vectorization disabled).
- **Host loop:** a 128-sample block per instance in turn.
- **Swept:**
  - K from 1 to 1024 and N from 256 to 64K (footprint up to 200 MB)
  - oscillator access: sequential, or a hashed phase (unpredictable reads)
  - 0, 2 or 16 MB of other work read and written between rounds

## Results (ns per sample per instance, whole chain)

| | warm, small | footprint > L2 (sequential) | > L2 (random) | 16 MB of other work, random |
|---|---|---|---|---|
| polynomials, vectorized | **1.3** (random 1.6) | 1.3 | 1.6 | 1.6 |
| polynomials, scalar | 4.5 | 4.5 | 4.8 | 4.8 |
| tables | **3.6–4.0** | 4–11 | 7–20 | 5–22 |

- **Warm tables beat scalar math (3.6 vs 4.5 ns) but lose to vectorized
  math (1.3 ns) by about 3×.** The lookups can't vectorize (no gather on
  NEON); the polynomials do. In per-sample feedback code, where nothing
  vectorizes, the warm table wins by about 20%.
- **Tables hold up far better than "cache misses kill tables" suggests:**
  - Footprint barely matters until it passes the **16 MB L2**. 1 KB of
    tables and 12 MB cost the same.
  - With sequential (oscillator) reads, even 200 MB of 64K-entry tables
    costs only about 4.8 ns: the prefetchers follow each voice's phase.
- **The collapse needs two things together:**
  1. **A footprint beyond L2 plus unpredictable reads.** N = 64K at 256
     instances with random phase: 20 ns, 5× warm.
  2. **Other work evicting the tables between callbacks.** With 16 MB
     between rounds, random reads from a 16K table cost **9.8 ns even for
     one instance** (192 KB of tables). Every block starts cold, and a
     128-sample block can't amortise the refill. The worst case measured is
     **22 ns** (K = 256, N = 64K, random, 16 MB between): 17× vectorized
     math, 4.6× scalar.
- **Ratio at a typical size** (N = 4096, `tables_slowdown.png`, table ÷
  scalar math):
  - The table stays at 0.7–0.9 up to K = 256 with nothing in between.
  - It crosses 1 at K ≈ 6 with random reads and 16 MB of other work, and
    reaches 1.9 at K = 1024.

## The talk takeaway

- **Prefer the polynomial when the code vectorizes.** It's 3× faster than
  even a warm table, and flat whatever else the host is doing.
- **When it can't vectorize, a small table is fine and slightly faster.**
  The cost shows up as *variance*: the same code is 3.6 ns or 10–20 ns
  depending on what ran before it (other plug-ins, other voices). That's
  the real argument for math in a realtime context: predictable worst
  case, not average speed.
- **The danger zone:**
  - many distinct tables (big wavetables per voice, per-instance LUTs)
  - data-dependent indices (waveshapers driven by audio, FM, noise)
  - small blocks after the host has run other work

## Not done

- **Aligned vs padded tables** (set conflicts with power-of-two strides),
  i16 tables, and mipmapped wavetables.
- **x86** (smaller L2, different prefetchers): the knee should move left.
- **Hardware counters** (Linux `perf stat`) to confirm miss counts; this is
  timing only.
- **Table accuracy** at each N isn't compared here.
