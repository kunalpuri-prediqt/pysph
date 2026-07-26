# Known Issues - gpu-nnps

- [2026-07-14] Dense per-level grids scale with each level's AABB, including
  empty space between disconnected refinement patches. The two-patch kill case
  allocated 64,343 cells for 694 particles and used 26.8x the saved WCSPH state.
  Resolved by the 2026-07-26 per-level dense/sparse hybrid; retained here as the
  regression case.
- [2026-07-26] Sorted sparse lookup fixes memory but is slower when a query
  spans many empty logical cells: the two-patch fused consumer was 3.65x slower
  than forced dense. Sparse is currently a memory-safety fallback.
