# Config Layout

The root of this directory keeps the original workspace-compatible YAML files because the sweep and evaluation scripts refer to paths such as `refocus_vo/configs/...`.

The subdirectories are curated views of the thesis experiments:

- `architecture_sweep/`: generated configs for the architecture-family sweep.
- `ratio_ablation/`: generated configs for the native/DINO ratio ablation across the three reported architecture families.
- `architecture_family_ablation/`: configs for the additional family ablation runs.
- `final_method/`: final method and baseline configs that are most useful to inspect directly.
- `data/`: dataset subset manifests used by the training and validation setup.
