# Known Issues - warp-backend

- [2026-08-05] Adaptive variable-resolution equations can now request exact
  destination/source smoothing-length gradients while uniform equations keep
  their averaged-`HIJ` source unchanged. The exact form passes local oracles
  but still fails the manufactured hydrostatic interface gate, so it remains
  experimental rather than a production consistency claim.
