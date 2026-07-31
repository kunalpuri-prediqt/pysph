# Known Issues - particle-memory

- [2026-07-31] The first dynamic two-level APR smoke mutates and reconstructs
  the fluid ParticleArray on the host at explicit adaptation checkpoints, then
  rebuilds all dependent Warp state. This validates lifecycle correctness but
  is not device-resident allocation/compaction or a performance design.
