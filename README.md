# Timbrogenesis

Timbrogenesis is a C++20 command-line application by
[Packard Bell 95](https://packardbell95.com/) for progressively transforming a
source WAV toward a target WAV. It supports seeded evolutionary search and
deterministic complex-spectral morphing.

## Project Status

Timbrogenesis is a personal experimental project shared for demonstration and
educational purposes. It is not intended as production-ready software and is
provided without warranties or ongoing support commitments.

The project uses FFTW3, which is licensed under the GNU General Public License.
See [FFTW's licensing information](https://www.fftw.org/doc/License-and-Copyright.html)
for details.

## Requirements

- CMake 3.16 or newer
- A C++20 compiler
- libsndfile development files (`libsndfile1-dev` on Debian/Ubuntu)
- FFTW3 development files (`libfftw3-dev` on Debian/Ubuntu)

## Build and Test

```bash
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Code Formatting

C++ source, headers, and tests use the root `.clang-format` configuration.
Use clang-format 18.1.8 for consistent results:

```bash
clang-format -i src/*.cpp src/*.hpp tests/*.cpp
```

To check formatting without changing files:

```bash
clang-format --dry-run --Werror src/*.cpp src/*.hpp tests/*.cpp
```

## Run

Both inputs must be mono WAV files with identical sample rates and sample
counts. If you don't want to do this manually, see
[the Automatic audio preparation section](#automatic-audio-preparation)
below.

```bash
./build/timbrogenesis \
    source.wav target.wav [run-directory] \
    [--output PATH] [--mode evolution|spectral-morph] \
    [--morph-strategy STRATEGY] [--morph-steps N] \
    [--replacement-order ORDER] [--seed N] \
    [--candidates-per-generation N] [--maximum-generations N] \
    [--maximum-attempts N | --maximum-successful-writes N] \
    [--max-accepted-mutations N] \
    [--fitness-dynamic-range DB] [--guided-mutation-ratio X] \
    [--refinement-ratio X] \
    [--target-fitness X] [--stagnation-generations N] \
    [--pre-conversion-source PATH] [--pre-conversion-target PATH]
```

Evolution is the default mode. Select `--mode spectral-morph` for the direct,
deterministic path described below.

### Spectral Morph Mode

Spectral morph analyzes the source and target as complex short-time Fourier
transforms (STFTs), constructs intermediate spectra at absolute positions
between them, and resynthesizes those spectra. It does not generate mutation
candidates, calculate evolutionary fitness, or maintain a genome.

`--morph-steps N` specifies the number of transitions between the endpoints.
For example, ten steps create eleven generation pairs at 0%, 10%, ..., 100%:

```bash
./build/timbrogenesis \
    source.wav target.wav \
    --mode spectral-morph \
    --morph-strategy interpolate \
    --morph-steps 10
```

`generation-000.wav` contains the original source audio and
`generation-010.wav` contains the original target audio in this example. The
endpoints bypass STFT reconstruction, so their decoded samples, sample rate,
and sample count are preserved exactly. Only the nine intermediate files are
resynthesized.

The available strategies are:

| Strategy | Intermediate spectrum |
| --- | --- |
| `interpolate` | Linearly interpolates each complex coefficient. |
| `polar-interpolate` | Interpolates log magnitude and moves phase along the shortest angular path. |
| `magnitude-source-phase` | Interpolates log magnitude while retaining source phase. |
| `magnitude-target-phase` | Interpolates log magnitude while using target phase. |
| `replace` | Progressively copies complete target time-frequency cells into the source spectrum. |

All progress values are absolute: generation 4 of a ten-step run is always
40% along the selected path. The engine analyzes each input once. Its core
transform cost for `N` steps is two forward STFTs plus `N - 1` inverse STFTs;
the two endpoint files are written directly. Replacement strategies also build
one cell ordering for the whole run.

The replacement strategy progressively copies complete target time-frequency
cells into the source spectrum. Its default `error` order introduces cells with
the largest source/target log-magnitude difference first:

```bash
./build/timbrogenesis \
    source.wav target.wav \
    --mode spectral-morph \
    --morph-strategy replace \
    --replacement-order error \
    --morph-steps 10
```

Replacement order can be `error`, `target-energy`, `source-energy`,
`low-to-high`, `high-to-low`, `early-to-late`, `late-to-early`, or `random`.
Energy orderings transform the strongest source or target cells first;
frequency and time orderings create directional sweeps. Random ordering
shuffles the cell list once. Use `--seed N` to reproduce that shuffle, or omit
the seed to generate and record one automatically in `run.json`.

Magnitude and phase can be morphed separately with `polar-interpolate`,
`magnitude-source-phase`, or `magnitude-target-phase`. All three interpolate
magnitude logarithmically. Polar interpolation moves phase along the shortest
angular path; the other strategies retain source or target phase throughout
the intermediate generations.

Each generation has a matching JSON file with its generation number, progress,
mode, and strategy. Replacement metadata also records the order and replaced
cell count. `run.json` records the source and target, strategy, step count, FFT
and hop sizes, sample rate, sample count, and random seed when applicable.
Evolution-only options are rejected in spectral morph mode, and morph-only
options are rejected in evolution mode.

### Evolution Mode

Runs default to 100 generations with 1,000 candidates per generation. When
`--seed` is omitted, the program generates and records one.

Use `--maximum-generations N` and `--candidates-per-generation N` to configure
those limits from the command line. The shorter existing aliases
`--generations N` and `--candidates N` remain supported.

By default, the generation limit counts attempted generations.
`--maximum-attempts N` is an explicit alias for that behavior. Alternatively,
use `--maximum-successful-writes N` to keep attempting generations until N
generation WAV/JSON pairs have been recorded. The generation-000 baseline
counts as the first successful write; attempts without an improvement do not
increment the count. The generation-limit options are mutually exclusive.
Target-fitness and stagnation limits, when supplied, can still stop the run
earlier.

Each generation may accept up to eight frame-disjoint mutations by default;
use `--max-accepted-mutations` to change that cap. Accepted sine atoms form an
editable genome, so later mutations can remove, move, or retune earlier atoms.

Fitness uses a target-relative spectral floor with an 80 dB dynamic range by
default. `--fitness-dynamic-range` changes that range. Evolution uses signed
spectral residual hotspots for 80% of broad proposals by default, retains 20%
random exploration, and reserves 40% of the candidate budget for local
refinement and bounded amplitude/phase searches. Use
`--guided-mutation-ratio` and `--refinement-ratio` to tune those proportions;
both accept values from 0 to 1, with the refinement ratio required to be less
than 1.

## Output

There are two equivalent ways to select the exact run directory: provide it as
the third positional argument, or use `--output PATH`. Do not use both forms in
the same command. A supplied path is used directly; the application does not
create another timestamped directory inside it. The directory must either not
exist or already be empty.

When neither output form is provided, the application creates a new timestamped
run directory under `output/`, such as `output/2026-09-11_143700/`. A run
directory contains:

- `generation-000.wav`: the unchanged source
- `generation-NNN.wav`: generation 0 and each improved evolutionary champion
- `generation-NNN.json`: metadata for generation 0 and improved generations
- `run.json`: settings, seed, final fitness, stop reason, and aggregate search statistics
- `genome.json`: the complete final set of active sine atoms with stable IDs

In spectral morph mode, every absolute step is written. The run contains
`morphSteps + 1` WAV/JSON generation pairs and a mode-specific `run.json`; it
does not create a genome file.

Generations that find no improving candidate are still counted for stopping and
stagnation behavior, but do not create redundant WAV or generation JSON files.

When invoking the executable directly, the optional `--pre-conversion-source`
and `--pre-conversion-target` arguments record the original input paths used to
produce the prepared WAV files. They appear in `run.json` as
`preConversionSource` and `preConversionTarget`; when omitted, those fields are
`null`.

## Automatic Audio Preparation

The `scripts/run-timbrogenesis.sh` wrapper can prepare arbitrary audio files before
running either application mode. This is useful when the source and target are
not already compatible mono WAV files.

The wrapper automatically records the original source and target paths in the
run metadata, alongside the paths of the prepared WAV files.

The script uses `ffmpeg` and `ffprobe` to:

- select the first audio stream from each input
- convert both inputs to mono
- extract the first channel by default for stereo or multichannel audio
- resample both inputs to an identical sample rate
- convert both inputs to 32-bit floating-point WAV files
- pad or trim them to an identical sample count
- verify the resulting channel count, sample rate, and sample count
- invoke `timbrogenesis` with the prepared files

By default, the lower of the two input sample rates is used. This avoids
unnecessary upsampling and reduces the amount of audio data processed.

The shorter recording is padded with silence to match the longer recording by
default. Use `--length shortest` to trim both recordings to the shorter length
instead.

Prepared files are retained under `prepared/` so the WAV files referenced by
the evolutionary run remain available after processing.

### Requirements

In addition to the main application dependencies, the preparation script
requires:

- `ffmpeg`
- `ffprobe`

On Debian/Ubuntu systems, both are provided by the `ffmpeg` package.

### Usage

Make the script executable:

```bash
chmod +x scripts/run-timbrogenesis.sh
```

Basic usage:

```bash
./scripts/run-timbrogenesis.sh source.mp3 target.flac
```

Options intended for `timbrogenesis` can be placed after `--`:

```bash
./scripts/run-timbrogenesis.sh \
    source.mp3 target.flac \
    -- \
    --maximum-generations 50 \
    --candidates-per-generation 2000
```

The same wrapper can launch a spectral morph:

```bash
./scripts/run-timbrogenesis.sh \
    source.mp3 target.flac \
    -- \
    --mode spectral-morph \
    --morph-strategy polar-interpolate \
    --morph-steps 10
```

A specific sample rate can be requested:

```bash
./scripts/run-timbrogenesis.sh \
    --sample-rate 44100 \
    source.mp3 target.flac
```

To trim both files to the shorter input instead of padding the shorter one:

```bash
./scripts/run-timbrogenesis.sh \
    --length shortest \
    source.mp3 target.flac
```

By default, multichannel audio uses only its first channel. To perform a mono
mixdown of all channels instead:

```bash
./scripts/run-timbrogenesis.sh \
    --mixdown \
    source.mp3 target.flac
```

Wrapper options can be combined with application options:

```bash
./scripts/run-timbrogenesis.sh \
    --sample-rate 48000 \
    --length shortest \
    source.flac target.mp3 \
    -- \
    --output output/test-run \
    --seed 123456789 \
    --maximum-generations 100 \
    --candidates-per-generation 1000
```

The available wrapper options are:

```text
--program PATH
    Path to the timbrogenesis executable.
    Default: ./build/timbrogenesis

--sample-rate HZ
    Force both prepared files to use this sample rate.

--length longest
    Pad the shorter file with silence to match the longer file.
    This is the default.

--length shortest
    Trim both files to the shorter file's length.

--mixdown
    Mix all channels to mono instead of extracting the first channel.

--prepared-root PATH
    Directory under which prepared WAV files are retained.
    Default: prepared

-h, --help
    Show the wrapper's usage information.
```

## A Bit of Fun with Sound Evolution

The evolution mode is quite slow and generally does not produce a satisfying
result all by itself. However, it is possible to get some interesting results
out of it. One application is to take the difference between the input and one
of the output sounds and pitch shift it to get some gnarly effects. Here's how
you can do it with an audio editor such as
[Audacity](https://www.audacityteam.org/):

1. Take two files from the same evolutionary run: `generation-001.wav` as the
   starting sound (rather than `generation-000.wav` which is the raw,
   unnormalized input) and some later generation (`generation-008.wav` in this
   example's case).
2. Align the two sounds at the same start time on separate tracks.
3. Invert the polarity of one of the tracks.
4. Mix the two tracks together. This subtracts the later generation from the
   starting sound, exposing their differences.
4. The end result may be inaudible, so pitch shift it. In this example, the
   resulting sound was shifted up by two octaves.
5. Enjoy your new and strange sound effect 🎶

<video controls preload="metadata" src="misc/evo-fun.webm">
    <a href="misc/evo-fun.webm">Watch the sound effects demonstration</a>
</video>

## Links
- [Post by MySpace Tom on Agora Road's Macintosh Café explaining the spectral morph operations of this software](https://forum.agoraroad.com/index.php?threads/agora-roads-tech-weekly.9276/#post-205806)
