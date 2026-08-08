# Excluded material

The source workspace contained GNN and PINN experiments that are outside this paper-aligned public release. The release excludes graph neural networks, graph convolutions, message-passing networks, watershed-context experiments, hybrid GNN-ConvLSTM code, PINN models, physics-informed training, physics-loss and conservation-loss experiments, their configurations, checkpoints, outputs, plots, tests, caches, logs, raw datasets, and superseded runs.

Mixed hyperparameter files contained both supervised ConvLSTM and excluded branches. Those originals were not copied. The supervised architectures, training rules, and six selected configurations were extracted into clean modules under `src/seed` and `configs/models`; consequently, the original per-file history could not be retained. The source workspace had no usable Git history in any event.

Also excluded:

- raw NLDAS and full ELM archives;
- cluster restarts, case directories, logs, caches, and temporary outputs;
- `.env` files and machine-specific paths;
- figures 1-4 from the reviewed manuscript because those versions describe excluded model material;
- legacy and exploratory model checkpoints;
- files larger than needed for inference or documented reproduction.

Required mentions of the excluded scope appear only in `README.md`, this manifest, and `docs/paper-code-map.md`. No excluded implementation or artifact is present.
