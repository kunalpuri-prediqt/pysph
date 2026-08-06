# Open Questions - warp-backend

- [open 2026-07-31] Active environment imports Warp `1.15.0`; what minimum
  Warp version and NVIDIA documentation set should be authoritative?
- [open] What minimum NVIDIA GPU/driver/CUDA assumptions are acceptable?
- [open] Should Warp be exposed as `backend='warp'` or as a CUDA backend implementation detail?
- [open] Which Warp primitive should be implemented first: push/pull, alignment, or add/remove/extract kernels?
- [closed 2026-08-05] Generated neighbor loops expose opt-in scalar `gradi`
  and `gradj` factors evaluated at destination/source `h`; equations not
  requesting them retain the prior averaged-`HIJ` generated source.
- [open 2026-08-06] What fixed-boundary sampling/correction and refinement
  study should gate a procedural terrain surface beyond this filled-particle
  interaction smoke?
