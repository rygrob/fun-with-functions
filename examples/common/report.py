"""Terminal output shared by every example."""

from .metrics import N_FFT


def fmt_ulp(u):
    return f"{u:,.2f}" if u < 1e4 else f"{u:.2e}"


def print_header(title, fields):
    """title line, then aligned "key : value" fields. Chebyshev fits by
    interpolation add a line (projection, the default, doesn't)."""
    from . import fit
    if fit.get_method() != "projection":
        fields = list(fields) + [("cheb fit", f"{fit.get_method()} (--fit)")]
    print(title)
    width = max(len(k) for k, _ in fields)
    for k, v in fields:
        print(f"  {k:<{width}} : {v}")
    print()


def print_result(result):
    stats, spec = result.stats, result.spec
    print("Error:")
    print(f"  max abs error : {stats['max_abs']:.3e}  ({stats['abs_bits']:4.1f} bits)")
    print(f"  max rel error : {stats['max_rel']:.3e}  ({stats['rel_bits']:4.1f} bits)")
    print(f"  max ULP error : {fmt_ulp(stats['max_ulp'])}")
    print(f"  mean ULP error: {fmt_ulp(stats['mean_ulp'])}")
    print()
    print(f"Spectrum ({N_FFT:,}-point FFT, one cycle):")
    print(f"  SINAD         : {spec['sinad']:.1f} dB  ({spec['enob']:4.1f} ENOB)")
    print(f"  worst harmonic: h{spec['worst_h']} at {spec['worst_dbc']:.1f} dBc")
    print()


def print_compare(results, key_header, extra_cols=()):
    """One row per result. extra_cols is a sequence of (header, width,
    fn(result) -> str) inserted after the key column."""
    header = f"{key_header:>6}"
    for name, width, _ in extra_cols:
        header += f" {name:>{width}}"
    header += (f"  {'max abs':>9} {'bits':>5}  {'max rel':>9} {'bits':>5}"
               f"  {'max ULP':>9}  {'mean ULP':>9}  {'SINAD dB':>8} {'ENOB':>5}")
    print(header)
    print("-" * len(header))
    for r in results:
        stats, spec = r.stats, r.spec
        row = f"{r.key:>6}"
        for _, width, fn in extra_cols:
            row += f" {fn(r):>{width}}"
        row += (f"  {stats['max_abs']:>9.2e} {stats['abs_bits']:>5.1f}"
                f"  {stats['max_rel']:>9.2e} {stats['rel_bits']:>5.1f}"
                f"  {fmt_ulp(stats['max_ulp']):>9}  {fmt_ulp(stats['mean_ulp']):>9}"
                f"  {spec['sinad']:>8.1f} {spec['enob']:>5.1f}")
        print(row)
    print()


def print_reference(ctx):
    print(f"f32(sin) reference: SINAD {ctx.ref['sinad']:.1f} dB ({ctx.ref['enob']:.1f} ENOB)")
    print()


def print_progression(rows, title):
    """rows of (name, method, result) -- the same degree across methods.
    method needs .ncoef, .mul and .add."""
    print(title)
    header = (f"  {'method':<18} {'coeffs':>6} {'mul':>4} {'add':>4}  {'max abs':>9}  {'max rel':>9}"
              f"  {'max ULP':>9}  {'SINAD dB':>8}")
    print(header)
    for name, m, r in rows:
        s = r.stats
        print(f"  {name:<18} {m.ncoef:>6} {m.mul:>4} {m.add:>4}  {s['max_abs']:>9.2e}  {s['max_rel']:>9.2e}"
              f"  {fmt_ulp(s['max_ulp']):>9}  {r.spec['sinad']:>8.1f}")
    print()
