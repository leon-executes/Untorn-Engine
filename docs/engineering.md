# Engineering reference

## Repository boundaries

`source/untorn_engine.py` is the sectioned desktop, CLI and numerical engine.
`source/native/` contains the two native helper build inputs and component hashes.
`source/create_icon.py` generates the original vector mark and multi-size ICO.
`tests/test_engine.py` exercises the engine and failure contracts.
`docs/` contains the architecture schematics, interface capture and release checks.
`samples/` contains five photographic inputs: coffee, the supplied feather crop,
green grass, busy ferns and dry foliage. Their provenance and separate image
licenses are recorded in `NOTICE`. They are input fixtures, not generated outputs.

Runtime job workspaces, compiler intermediates, developer tools and old experiments
are not repository content. Release ZIPs are generated into `dist/`; Git ignores
them. A clean source ZIP is built from an explicit project-file allowlist.

## Build contract

Use Windows x64 and CPython 3.12. Create a virtual environment and install
`requirements-build.txt`. Runtime dependencies are NumPy 2.5.4, Pillow 12.3.0 and
wxPython 4.3.2. Packaging uses PyInstaller 6.22.3 and hooks 2026.8.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\.venv\Scripts\python.exe build_portable.py
```

The builder runs the tests before producing GUI and console executables from the
same source. It checks the packaged CLI version and seals the portable archive
with a per-file SHA-256 manifest. It also creates a source ZIP with no user jobs,
intermediate files or local absolute paths. The package uses an onedir runtime;
keep `_internal` beside both executables.

## Native helpers

Prebuilt x64 helpers are intentionally tracked under `source/bin`. Their hashes
and source descriptions are recorded in `source/native/build.json`. To rebuild,
place MinGW-w64 `gcc` and `g++` on PATH and run:

```powershell
python source/native/build_helpers.py
```

The source archive contains the required GCS C/header files and its BSD license;
upstream demo images and unrelated utilities are excluded. The script safely
extracts `contours.tar.gz` into `build/native`, compiles the
original GCS interpolation sources with the BSD-licensed BMP-only I/O adapter,
and links the executable with its compiler runtime. The process boundary uses
BMP, avoiding additional image-codec dependencies in that helper.

The OpenCL helper is compiled as C++17 with static compiler runtimes. It loads
`OpenCL.dll` from Windows System32 and requires no OpenCL SDK/import library at
build time. Installed GPU drivers supply the runtime. The Python engine retains
its CPU reference path for enumeration, validation and execution failures.

Native executable bytes may vary with compiler version, path and build metadata.
Numerical equivalence must be tested; matching source is not proof of a
bit-reproducible executable build.

## Numerical and persistence contracts

- The native detail grid is 4×. Destination rounding occurs in global coordinates.
- Input-area policy is evaluated before tiling; 262144 source pixels is inclusive.
- Cache signatures include source hash, size, numerical version and mode/recipe.
- Different numerical recipes must never reuse unfinished tile output.
- Tile receipts record validated outputs; global accumulation is owned by the
  supervisor thread rather than independently mutated by parallel workers.
- GPU output must pass shape, finiteness and reference checks before use.
- Exports use atomic replacement; interrupted work must not destroy a prior result.
- Stage-duration sums can overlap across threads; active job time is separate.

When changing a numerical contract, update its cache identity and add behavior
checks. Keep desktop and CLI entry points on the same implementation. Use structured
subprocess arguments and retain original inputs. Do not introduce unbounded worker
counts or silently convert unsupported colour/bit-depth formats.

## Validation and code style

```powershell
python tests/test_engine.py
ruff check source tests build_portable.py
ruff format --check source tests build_portable.py
```

Ruff 0.14.1 is a development-only formatter/linter, configured in `pyproject.toml`.
The original OpenCL C++ source is formatted with clang-format 21.1.0. Upstream
license headers and reference-source archives are retained.

Numerical reviews should include smooth fields, directional edges and dense
textures. Pixel error alone is insufficient to assess visible detail. Check
cache compatibility, fractional sampling, exact integer-scale centre registration,
colour limits and CPU/GPU fallback separately from perceptual comparisons.

The Windows CI workflow builds on pushes and pull requests with read-only
repository permissions. It uploads build artifacts; it does not publish a release.

## Security reports

For input-decoding, process-execution or workspace-integrity issues, use GitHub
private vulnerability reporting if enabled. Otherwise contact the maintainer
through the profile linked in README before sharing sensitive details. Include
the application version, minimal non-sensitive input and reproduction steps.
Do not attach credentials, private photographs or complete workspaces to public
issues. Untorn processes local files; it is not a sandbox for untrusted binaries.

## Current release and cache identity

Application **2.3.0**; numerical cache version **1.3.0**. The desktop and CLI
default to Max. The `xray-plus` spelling remains an alias for X-Ray; cache reuse
still requires an exact numerical identity.

| Modes | Fitted shading | Recipe |
| :--- | :--- | :--- |
| Fast / Max | Enabled | `source-grain-spatial-blend-v3` |
| Fast / Max | Disabled by input area | `source-grain-spatial-blend-no-fitted-v3` |
| X-Ray | Input-area policy recorded in the job identity | `xray-shared80-spatial-blend-v8` |

Unfinished jobs with older numerical identities are rejected. Completed exports
are retained. The fixed structural weights are 0.77 unaligned/FXAA and 0.23
aligned/no-FXAA; their float blend precedes destination sampling and quantization.

`spatial_envelope` owns shared neighbourhood bounds. `spatial_branch` accepts
explicit alignment and FXAA flags. `spatial_cleanup` owns composition and uses
the same precomputed contributor fields for both branches. Tile workers do not
run independent full upscales for the two branch results.

[Release checks](release-checks.json) record the validated source and executable
identities, numerical contracts and runtime checks. The five bundled samples
are a reproducible inspection set; they do not replace validation on additional
Windows systems, image content and GPU drivers.

## Numerical stage contracts

The following parameters describe the shipping implementation. Stage order and
dependencies are shown in [the signal schematic](architecture-signal.svg).
All working-pixel radii refer to the native 4× grid unless stated otherwise.

### Structure and shading

**DCCI** supplies the core structural field through repeated 2× directional
interpolation. Direction-dependent estimates reduce cross-edge mixing while
retaining the input's geometry. A fitted colour/shading contributor supplies
broad spatial colour variation for smaller inputs; DCCI retains the fine
structural component of the base.

The fitted contributor is enabled for input area **≤ 262144 pixels**. Thus both
512 × 512 and 1024 × 256 qualify. Above that total area it is skipped, reducing
cost; DCCI supplies the broad field and the remaining grain contributors are
renormalized. The policy depends on the complete input area, not tile dimensions.

### Middle detail

**Fast** evaluates Local Laplacian detail at **32 intensity references**, retaining
the complete directional base, fitted-shading policy, middle-band correction and
source-derived grain ensemble. Its reduced remapping density lowers processing
cost; it can change both middle detail and the Local Laplacian grain contribution.

**Max** uses **64 intensity references** for the middle band and grain donor.
The denser remapping places greater emphasis on detail precision while keeping
the source-led structure and tonal balance. It is the quality-and-faithfulness
formula for general photographic enlargement.

**X-Ray** evaluates Local Laplacian detail at **80 intensity references** and
reuses that same field for both middle detail and grain. The grain donor uses the same evaluated field. Reference count controls sampling of the local
tonal remapping, rather than the number of sharpening passes.

In X-Ray, contour-stencil interpolation (GCS) and Jinc4 contribute a bounded middle-band
correction. Their nominal weights are **12%** and **6%**; Jinc4's detail contrast
is further attenuated, and the combined middle correction has a **0.60** strength.
These are band-limited corrections relative to DCCI, not full-image opacity layers.
GCS uses a stronger contour interpolation configuration in this band. Jinc4 uses
90% anti-ringing plus a local source-envelope limit of at most one 8-bit code
value (or 2% of local range, whichever is smaller). X-Ray's GCS middle donor uses
25% suppression of excursions beyond a local DCCI envelope. DCCI and Local Laplacian receive no additional global unsharp pass.

### Fine texture

With fitted shading enabled, donor gains are **2.5 / 2.5 / 1.75 / 1.75** for
DCCI / fitted / Local Laplacian / GCS, followed by equal averaging. Without
fitting, the remaining gains are **2.5 / 1.75 / 1.75**, averaged across three donors.

All three modes use a complete grain ensemble of compact, image-derived residuals from DCCI,
fitted shading when enabled, Local Laplacian and GCS. Texture and structure masks
limit the correction around prominent boundaries. Per-donor gain, a fine-band
unsharp response and a smooth luminance limiter determine the final amplitude.
Fast uses its 32-reference Local Laplacian donor; Max uses its 64-reference donor.
Both preserve the same grain gating and amplitude controls. When fitted shading
is omitted, the remaining contributors are renormalized.

X-Ray guards GCS grain against unsupported two-axis periodic patterns at native-grid
periods of 2, 4 and 8 pixels. A continuous local confidence field blends suspect
fine responses toward DCCI's fine response, by at most 85%. Source-agreeing texture
and one-axis structure are retained; grain masks and contributor weights do not change.
The guard does not blur the output or introduce source-cell partitions. Patterns already
present in DCCI can remain; this is not a guarantee of removing every grid artifact.

X-Ray refines **its Local Laplacian grain contribution** with four coordinated
adjustments. The fine-band unsharp amount is **2.6**, versus 2.0 for the ordinary
ensemble, while its amplitude is **0.90×**. Source-guided neutral-veil compensation
acts on that fine residual with transmission bounded at **0.82**. A **12%**
muted-weighted source-chroma residual supplies grain vibrance; its luminance is
removed before composition. This is a grain-scale visibility adjustment, not a
whole-image dehaze filter or an estimate of the scene's true atmospheric depth.
Positive Laplacian grain peaks receive up to **4%** additional lift in upper
source tones, before support gating, cell-mean removal and gamut limits.

The additional RGB correction is confined to the Laplacian size/shape
mask and texture gate. Its mean is removed within each **4 × 4 source-pixel cell**
on the working grid. One common gamut bound per cell preserves that mean and
support, with an additional amplitude limit of **3/255 per channel**. A local
3×3 envelope limits Laplacian grain overshoot with at most one code value of
headroom (2% of local range in low-contrast regions). These
contracts hold before final sampling and 8-bit quantization; quantization can
remove very small changes. The refinement does not apply a new
broad colour, exposure or lighting adjustment.

X-Ray adds two restrained **internal source-matching** residuals: translated
examples and translated/rotated examples. Both search the same input; neither
uses an external dictionary or learned weights. Their contribution is confined
to fine grain, filtered at a 0.8-working-pixel Gaussian radius and gated with the
texture mask. Each residual contributes **1/6**, followed by local
4×4-cell mean removal. This preserves the local mean under that box model before
subsequent composition and quantization. Gaussian filtering attenuates broader
frequencies; it is not an ideal frequency cutoff.

Matching uses 3×3 queries, 12×12 donors, a four-source-pixel local search radius,
27 candidates including rotations of −20°, 0° and +20°, and batches of 128 queries.
Low-confidence or out-of-bounds donors are rejected. Overlapping estimates use
smooth window weights. The two searches share work and reuse the main X-Ray render.

### Tone and colour

Fast and Max retain the source-led tonal field from DCCI and eligible fitted
shading, with bounded luminance corrections for detail. X-Ray additionally
applies controlled shadow visibility, highlight shaping and a modest white
lift predominantly through the weighted GCS/Jinc4 contributions. The lighting
correction is spatially smoothed to avoid turning a tonal adjustment into another
fine-grain sharpening pass. Jinc4 uses a **0.31** white-lift coefficient;
GCS retains **0.28**. This upper-tone lift preserves the white endpoint.
DCCI supplies a slight vibrance correction, capped at
**2% additional chroma** and weighted toward muted colours. Shared RGB headroom
limits the correction without independent channel clipping.

### Artifact control and static anti-aliasing

All three modes apply compact-detail protection before final composition. A
continuous DCCI detail field reduces repeated grain emphasis around already
pronounced compact features. A short Gaussian grain-rounding pass then blends
35% of the original response with 65% of its rounded response; local energy
compensation is capped at 1.35×. This addresses coarse grid-like emphasis without
applying a broad blur to the photograph. It is a confidence-based adjustment,
not a classifier that can distinguish every real texture from every artifact.

The structural field is processed through two complementary branches:

| Contribution | Realignment | FXAA | Purpose |
| :--- | :--- | :--- | :--- |
| **77%** | Off | On | Retain the source geometry while reducing small stair-step discontinuities |
| **23%** | On | Off | Introduce a restrained amount of coherent contour correction without a second FXAA pass |

Both branches use the selected mode's tangent integration and bounded
normal-direction definition recovery. Only the 23% branch tracks ridge, valley
and edge centres and fits their trajectories. Only the 77% branch runs FXAA
3.11 Quality on the intermediate grid. A shared five-pixel structural envelope
limits each branch's correction with one RGB factor, then their outputs are
combined at **0.77 / 0.23** in encoded RGB. The texture residual is preserved
through both structural branches before final gamut clipping and quantization.

The costly DCCI, fitted, Local Laplacian, GCS and X-Ray contributors are evaluated
once per tile. The two branches share these fields and their neighbourhood
bounds; this is not two complete upscale jobs. Production combines floating-point fields
before final output rounding, avoiding an intermediate PNG encode/decode cycle.

| Spatial setting, native output pixels | Fast | Max | X-Ray |
| :--- | ---: | ---: | ---: |
| Trajectory half-window | 10 | 16 | 16 |
| Maximum coordinate correction, 23% branch | 0.9 | 1.1 | 1.6 |
| Tangent integration radius | 3.2 | 4.0 | 4.5 |
| Tangent correction strength | 0.65 | 0.75 | 0.90 |
| Bounded normal recovery | 0.25 | 0.25 | 0.50 |

The portable CPU implementation adds no runtime dependency. Spatial hashing limits
ridge correspondence work to nearby candidates, while overlapping tiles
bound image memory. Residual waviness or source aliasing can remain; these controls
reduce artifacts without claiming to recover unavailable photographic evidence.
