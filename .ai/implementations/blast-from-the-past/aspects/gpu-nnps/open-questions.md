# Open Questions - gpu-nnps

- [open] Which existing GPU NNPS path is the baseline for the first Warp comparison?
- [open] What neighbor-list correctness metric will gate performance claims?
- [closed 2026-07-26] Use a per-level hybrid: dense for
  `logical_cells / occupied_cells <= 4`, otherwise sorted int32 cell keys with
  device lower-bound lookup. ADR-0007 records the measured tradeoff.
- [open 2026-07-26] Can sparse traversal iterate occupied key ranges instead of
  performing a lower-bound lookup for every empty logical cell in a wide query?
