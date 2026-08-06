---
type: experiment
id: 2026-08-05_warp-webgpu-fluid-surface
created: 2026-08-05T14:45:00 CEST
author: @kunalpuri-prediqt
agent: codex
aspect: validation-benchmarks
status: complete
last_checked: 2026-08-06T11:09:00 CEST
plan: 2026-08-05_warp-webgpu-fluid-surface-renderer
---

# Experiment: client WebGPU fluid surface

## Purpose

Validate the gameplay-only client screen-space water renderer separately from
the PBF solver and from the Adaptive/Uniform VTK scientific views. Measure the
transport and browser GPU path at the approved browser size, and inspect final,
depth, thickness, and normal targets on real hardware.

## Configuration

- server GPU: NVIDIA GeForce RTX 5090;
- browser: Edge 151 headless hardware session on Windows 10/11 compatibility
  user agent;
- WebGPU adapter: `vendor=nvidia`, `architecture=blackwell`, not a fallback;
- browser viewport: `1280x720`; render canvas after studio chrome:
  `1265x656`;
- fluid: 1,000 particles, 250 gameplay frames, three PBF projections;
- surface: two bilateral iterations, radius four, one float16 depth target,
  one additive float16 thickness target, reconstructed normals, final water
  composite;
- transport: 16 float32 values / particle, base64 Trame state.

## Results

The completed 23-frame buffered run was replayed twice after warm-up at the
fixed browser size. Timings use browser `performance.now`; render completion
awaits `GPUQueue.onSubmittedWorkDone` and is not the server solver timer.

| Measurement | Result | Gate |
|---|---:|---:|
| Packed wire bytes / frame | `85,336 B` | `<=98,304 B` pass |
| Server pack median / p95 | `0.182 / 0.224 ms` | median `<=2 ms` pass |
| Browser upload median / p95 | `0.000 / 0.100 ms` | report |
| WebGPU completion median / p95 | `3.300 / 5.800 ms` | `<=16.7 / <=33.3 ms` pass |
| Dropped / stale frames | `0 / 0` | pass |

Hardware WGSL compilation succeeded through `getCompilationInfo`, all four
render pipelines were created asynchronously, and the app reported `READY`.
The final composite showed a continuous body through the initial column,
impact sheet, and settled pool. The orange quaternion box was visible and
occluded the water when closer to the camera. Raw depth, accumulated
thickness, and reconstructed-normal views all produced coherent fluid
silhouettes. Orbit, wheel zoom, reset, 1280x720 resize, replay, and
Surface-to-Particles-to-Surface switching completed without an error.

An additional default-backend Edge rerun covered the editor browser's stricter
WGSL validation. It caught and then passed the fix for an invalid compound
assignment to a writable `xy` swizzle; the corrected renderer again reported
`READY` after a complete 250-step gameplay run.

The Particles switch hid the WebGPU canvas, immediately produced a retained
17,527-character JPEG data URI through the existing VTK scene, and switching
back republished the same replay step. Surface live frames do not invoke VTK
scene update or JPEG encoding; this routing also has a direct Python oracle.

## Validation

- Node decoder/matrix/resize/shader/fallback contracts: `6 passed`;
- studio plus transport tests: `34 passed`;
- complete Warp separation regression: `79 passed`, two installed-Warp
  Python-future deprecation warnings;
- 250-frame browser-launched gameplay run: completed, finite;
- `git diff --check`: clean;
- implementation memory validation: recorded in the review after closeout.

## Interpretation limits

The renderer is view-dependent and display-only. Timings are local browser
evidence on this RTX/Edge setup, not a network-deployment throughput claim.
Screen-space refraction cannot sample offscreen objects; bright specular and
normal detail remain quality-tuning concerns. No result here validates PBF as
scientific SPH or changes the unresolved APR physics gates.

## Hands-on visual follow-up

The first browser tuning over-preserved individual particle depth, producing
a glossy chain of capsules rather than water. The follow-up uses gentler
bilateral depth blending, a wider normal footprint, thickness-contoured
silhouettes, wavelength-dependent absorption, restrained Fresnel/specular,
procedural tank context, and small simulation-time normal ripples. This is a
material/reconstruction correction only; no display quantity feeds back into
the PBF state.

At the original 1,000-particle Fast resolution, the result is less neon and
less capsule-like but remains visibly under-resolved because the pool is only
a few particles deep. The 7,020-particle Rich preset produced a substantially
more continuous early impact and pool at approximately `4.0 ms` browser GPU
completion. Its `599,040 B` base64 payload exceeds the Fast transport gate,
so Rich is currently a local presentation preset rather than evidence for a
remote deployment. Temporary Edge captures were inspected interactively and
were intentionally not added to the repository.
