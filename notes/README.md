# Notes

Longer write-ups behind the talk, written as the work went along. They're
working notes rather than polished articles, and some quote numbers from before
later changes; those say so at the top. Run the code for current numbers.

**Start here**

- [`error-metrics.md`](error-metrics.md): absolute, relative, ULP and SINAD, and when each one matters
- [`factoring-zeros.md`](factoring-zeros.md): why building in the zeros fixes relative error
- [`fit-projection.md`](fit-projection.md): Chebyshev projection against interpolation
- [`degree-effects.md`](degree-effects.md): what raising the degree buys, and when it stops
- [`reading-list.md`](reading-list.md): independent sources for the background

**Floating point and compilers**

- [`fp-contraction.md`](fp-contraction.md): when `a * b + c` becomes one instruction
- [`fast-math-split-pi.md`](fast-math-split-pi.md), [`fast-math-guard.md`](fast-math-guard.md): keeping split π alive under fast-math

**Speed**

- [`latency-horner-estrin.md`](latency-horner-estrin.md): latency vs throughput, Horner vs Estrin, three loops
- [`sinf-latency.md`](sinf-latency.md): why `std::sin` keeps up in a feedback loop
- [`tables-vs-math.md`](tables-vs-math.md): what happens as tables multiply
- [`lut-reproduction.md`](lut-reproduction.md): reproducing "Lookup Tables Are Bad Approximators"

**Other functions**

- [`tan-filter-cutoff.md`](tan-filter-cutoff.md): tan(πx) for filter cutoffs
- [`exp2-log2-tanh.md`](exp2-log2-tanh.md): bits for the integer part, polynomials for the rest
- [`ebay-log2.md`](ebay-log2.md): what Goldberg's "Fast Approximate Logarithms" adds
- [`db-gain.md`](db-gain.md): dB ↔ gain
- [`tanh-asymptotes.md`](tanh-asymptotes.md), [`tanh-feedback.md`](tanh-feedback.md): tanh's asymptotes, and tanh in a feedback loop
