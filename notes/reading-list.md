# Reading list: independent sources for the background

Third-party sources (textbooks, papers, Wikipedia, blogs) for the background
to the talk. They're listed by title rather than URL:
search for them. Editions are worth double-checking.

## Floating point: ULPs, rounding, Sterbenz, f32(π) ≠ π

- **David Goldberg, "What Every Computer Scientist Should Know About
  Floating-Point Arithmetic"** (ACM Computing Surveys, 1991). Free online
  (Oracle hosts a copy). The classic: ULPs vs relative error, exact
  subtraction, cancellation. *Read first.*
- **Nicholas Higham, *Accuracy and Stability of Numerical Algorithms***
  (SIAM, 2nd ed. 2002). Rigorous rounding-error analysis: Horner's error
  bounds, Sterbenz's lemma.
- **Jean-Michel Muller et al., *Handbook of Floating-Point Arithmetic***
  (Birkhäuser, 2nd ed. 2018). The reference: FMA, exact operations, the
  several definitions of ULP.
- **Bruce Dawson, "Comparing Floating Point Numbers, 2012 Edition"**
  (randomascii blog). Practical integer ULP distance from bit patterns.
- **Wikipedia:** *Unit in the last place*, *Floating-point arithmetic*,
  *Sterbenz lemma*, *Catastrophic cancellation*, *Fused multiply–add*.

## Approximating elementary functions

- **Jean-Michel Muller, *Elementary Functions: Algorithms and
  Implementation*** (Birkhäuser, 3rd ed. 2016). *The* book for this talk:
  minimax and Remez, range reduction (Cody–Waite, Payne–Hanek), table-based
  methods, polynomials with machine-number coefficients.
- **W. J. Cody & W. Waite, *Software Manual for the Elementary Functions***
  (Prentice-Hall, 1980). Where split-constant reduction comes from (split-π,
  Cody–Waite exp).
- **N. Brisebarre, J.-M. Muller, A. Tisserand, "Computing Machine-Efficient
  Polynomial Approximations"** (ACM TOMS, 2006). Coefficients that are exact
  floats (cf. 08).
- **Sollya** (tool and docs). Minimax with float coefficients, relative or
  absolute weights, and fixed terms.

## Chebyshev: projection vs interpolation, near-minimax, aliasing

- **Lloyd N. Trefethen, *Approximation Theory and Approximation Practice***
  (SIAM, 2013; early chapters free on his Oxford page). Chebyshev series vs
  interpolants, aliasing, the factor-of-2 relation, near-best. *Read for
  `fit-projection.md`.*
- **Chebfun** (chebfun.org). The guide and examples.
- **Wikipedia:** *Chebyshev polynomials*, *Chebyshev nodes*,
  *Equioscillation theorem*, *Remez algorithm*, *Padé approximant*,
  *Horner's method*.

## Bit tricks (exp2 / log2)

- **N. N. Schraudolph, "A Fast, Compact Approximation of the Exponential
  Function,"** *Neural Computation* 11(4), 1999.
- **J. N. Mitchell, "Computer Multiplication and Division Using Binary
  Logarithms,"** *IRE Trans. Electronic Computers* EC-11(4), 1962.
- **D. Goldberg, "Fast Approximate Logarithms," parts I–III** (eBay, 2015).
  Links are in `ebay-log2.md`, with notes.
- **Wikipedia:** *Fast inverse square root*. Same idea, good history.
- **ARM optimized-routines** (GitHub). Production `exp2f` / `logf`: a small
  table plus a low-degree polynomial, with commented error bounds.

## tan and the filter prewarp

- **Vadim Zavalishin, *The Art of VA Filter Design*** (free PDF, Native
  Instruments). TPT filters, g = tan(π fc/fs), bilinear-transform warping.
- **Wikipedia:** *Bilinear transform* (the frequency-warping section).

## Compiler behaviour (FMA contraction, fast-math)

- **The GCC wiki, "FloatingPointMath",** plus the GCC and Clang manuals on
  `-ffp-contract` and `-ffast-math`.
- **Muller's *Handbook*** (above) for FMA semantics.

## Tables vs computation

- **Nima Badizadegan, "You (Probably) Shouldn't Use a Lookup Table"**
  (Speculative Branches blog).
- **Jatin Chowdhury, "Lookup Tables Are Bad Approximators"** (2025). Notes in
  `lut-reproduction.md`.

## Suggested order

1. Goldberg (1991): floats and ULPs.
2. Trefethen, ATAP: the chapters on Chebyshev series and interpolation.
3. Muller, *Elementary Functions*: polynomial approximation and range
   reduction.
