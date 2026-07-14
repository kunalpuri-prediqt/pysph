# Known Issues - gpu-nnps

- [2026-07-14] Dense per-level grids scale with each level's AABB, including
  empty space between disconnected refinement patches. The two-patch kill case
  allocated 64,343 cells for 694 particles and used 26.8x the saved WCSPH state.
  ADR-0007 remains Proposed pending sparse or hybrid level storage.
