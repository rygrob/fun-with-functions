# Fun With Functions

**Building a Library of Custom Math Approximations.** The code, data and slides
from Ryan Robinson's talk at the Audio Developer Conference 2026.

**Slides:** [functions.rygrob.com](https://functions.rygrob.com).

The talk designs fast approximations of `sin`, `tan`, `log2`, `2^x` and `tanh` for
audio, using Chebyshev polynomials.

## Just want the functions?

Copy [`bench/generated/`](bench/generated): plain `template<typename T>` functions in
headers, using only the standard library. Most of the headers include
`bit_helpers.hpp`, and `tanh_approx.hpp` includes `exp2_approx.hpp`, so keep the
directory together.

| Function | From the header | Max ULP | Max error | Speedup, sample loop |
|---|---|---:|---:|---:|
| sin x | `cheb_split_guarded_sin_13` (`sin_approx.hpp`) | 4.44 | −132 dB | 2.0× |
| sin 2πx | `cheb_turns_sin_13` (`sin_approx.hpp`) | 4.97 | −133 dB | 2.3× |
| tan πx | `cheb_turns_tan_pi_9` (`tan_approx.hpp`) | 4.08 | 0.0005 cents up to 20 kHz | 2.8× |
| log₂ x | `fold15_log2_7` (`log2_approx.hpp`) | 24.5 | −116 dB | 2.1× |
| 2^x | `round_unit_exp2_5` (`exp2_approx.hpp`) | 3.2 | 2.7e-7 relative | 1.9× |
| tanh x | `sigmoid_cheb_clamp_tanh_9` (`tanh_approx.hpp`) | 24.1 | −117 dB | 1.8× |

Each error is the maximum over every normal `float` in the function's range. Speedup
is against the matching `std::` function, on sorted inputs, measured on one machine:
an M2 Pro MacBook Pro on power, Apple clang 21, `-O3`, no fast-math. Treat the timings
as indicative and rerun them on your own hardware. Read the docs above each function
for its input range.

## What's here

| Path | What it is |
|---|---|
| `examples/` | One approximation each in Python, numbered in talk order: Taylor, tables, Padé, Chebyshev, Remez, then each function. Most take `-d`, `--compare`, `--eval f32\|f64`, `--no-show` and `--emit-cpp` |
| `examples/common/` | The shared fits, metrics and C++ generation, all hand-rolled |
| `bench/` | The C++ bench: timing, and the exhaustive check over every `float` |
| `slides/` | The deck, with `generate.py`, which reproduces every number and curve on it |
| `notes/` | Longer write-ups of what we found along the way; start with [`notes/README.md`](notes/README.md) |

## Running it

```sh
pip install -r requirements.txt

python3 examples/06-chebyshev-turns/cheb_turns_sin.py --no-show
python3 examples/18-degree-effects/degree_effects.py --no-show

python3 bench/generate.py                      # examples -> bench/generated
cmake -S bench -B bench/build && cmake --build bench/build
bench/build/adc_bench ulp f32 --domain turns   # every float, on half your cores (--threads n)
bench/build/adc_simple_loops                   # the benchmark on the slides

python3 slides/generate.py                     # slide data -> slides/data
cd slides && npm install && npm run serve      # the deck at localhost:8026
```

CMake fetches [`audio_bench`](https://github.com/sketch-audio/audio_bench) for the
timer. Everything here has been run on an Apple Silicon Mac; Linux and Windows
builds are untested, though
[adc-msvc](https://github.com/rygrob/adc-msvc) checks the split π guard on MSVC.

## Related

Claude helped write much of the code and notes in this repo, as the talk mentions.

## License

The code and notes are [MIT](LICENSE). The images in `slides/img/` aren't included.
