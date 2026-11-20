# MoReG multi-seed ablation study

This repository contains Jupyter notebooks for a multi-seed ablation study of MoReG, a 3D brain-tumour segmentation model evaluated on the BraTS 2021 Task 1 dataset.

The experiments use four MRI modalities (`t1`, `t1ce`, `t2`, and `flair`) and predict three nested tumour regions:

- **WT** - whole tumour
- **TC** - tumour core
- **ET** - enhancing tumour

## Experiments

Each experiment is run with seeds 7, 42, and 256.

| Variant | Architectural change | Parameters | Notebooks |
| --- | --- | ---: | --- |
| Full MoReG | Modality-aware fusion, squeeze-and-excitation residual blocks, dilated context, and gated skip connections | 4,656,916 | [`moreg-full(7).ipynb`](./moreg-full%287%29.ipynb), [`moreg-full(42).ipynb`](./moreg-full%2842%29.ipynb), [`moreg-full(256).ipynb`](./moreg-full%28256%29.ipynb) |
| Without skip gates | Removes the learned gates applied to encoder-to-decoder skip features; other attention blocks remain enabled | 4,632,385 | [`moreg-without-skip-gates(7).ipynb`](./moreg-without-skip-gates%287%29.ipynb), [`moreg-without-skip-gates(42).ipynb`](./moreg-without-skip-gates%2842%29.ipynb), [`moreg-without-skip-gates(256).ipynb`](./moreg-without-skip-gates%28256%29.ipynb) |
| Without dilated context | Replaces the three-branch dilated bottleneck context block with an identity mapping | 3,107,668 | [`without_dilation_context(7).ipynb`](./without_dilation_context%287%29.ipynb), [`without_dilation_context(42).ipynb`](./without_dilation_context%2842%29.ipynb), [`without_dilation_context(256).ipynb`](./without_dilation_context%28256%29.ipynb) |

## Common pipeline

The notebooks implement the complete training and evaluation pipeline:

1. Load and validate the four aligned NIfTI modalities and segmentation mask.
2. Reorient images to canonical orientation, crop to the non-zero brain region, clip each modality to its 0.5-99.5 percentile range, and z-normalize non-zero voxels.
3. Split 1,251 patients into 875 training, 188 validation, and 188 test cases. The split is saved and reused by the experiment outputs.
4. Sample `96 x 96 x 96` patches, preferentially centred on foreground, with flips, right-angle rotations, intensity perturbations, and noise augmentation.
5. Train a residual 3D encoder-decoder with random affine augmentation, automatic mixed precision, gradient accumulation, AdamW, EMA weights, and deep supervision. The code also supports modality dropout for the optional teacher/student path; the reported main runs use all four modalities.
6. Evaluate full volumes with MONAI sliding-window inference. Predictions are hierarchically projected so ET is a subset of TC, and TC is a subset of WT.

The loss combines soft Dice, binary cross-entropy, a boundary-gradient term, and a hierarchy penalty. The two auxiliary output scales use weights `0.5` and `0.25` relative to the full-resolution output.

## Default configuration

The full runs use:

| Setting | Value |
| --- | ---: |
| Patch size | `96 x 96 x 96` |
| Base channels | `24` |
| Batch size | `1` |
| Gradient accumulation | `2` |
| Epochs | `125` |
| Maximum training batches per epoch | `250` |
| Initial learning rate | `3e-4` |
| Minimum learning rate | `1e-6` |
| Weight decay | `1e-4` |
| Warm-up | `10` epochs |
| Validation frequency | Every `5` epochs |
| Foreground patch probability | `0.75` |
| Modality dropout probability | `0.35` |
| Inference threshold | `0.5` |
| Test bootstrap samples | `2,000` |

## Reported results

The table below records the `mean_patient_macro_dice` and 95% bootstrap interval printed by each notebook. All test runs contain 188 patients. Values are copied from the saved notebook outputs, not recomputed by this README.

| Variant | Configured seed | Seed recorded in output | Validation Dice | Test macro Dice | 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| Full MoReG | 7 | 7 | 0.8997 | 0.8943 | 0.8778-0.9093 |
| Full MoReG | 42 | 42 | 0.8852 | 0.8904 | 0.8740-0.9061 |
| Full MoReG | 256 | 256 | 0.8849 | 0.8951 | 0.8779-0.9105 |
| Without skip gates | 7 | 7 | 0.8994 | 0.8914 | 0.8745-0.9065 |
| Without skip gates | 42 | **256** | 0.8861 | 0.8933 | 0.8761-0.9091 |
| Without skip gates | 256 | 256 | 0.8861 | 0.8933 | 0.8761-0.9091 |
| Without dilated context | 7 | 7 | 0.8968 | 0.8931 | 0.8772-0.9075 |
| Without dilated context | 42 | **7** | 0.8968 | 0.8931 | 0.8772-0.9075 |
| Without dilated context | 256 | 256 | 0.8843 | 0.8905 | 0.8731-0.9069 |

The seed-42 outputs marked in bold are inconsistent with their notebook filenames/configuration and should be re-run before being used as seed-42 measurements.

## Running the notebooks

These notebooks are written for a Kaggle-style Jupyter environment. They expect the BraTS archive at:

```text
/kaggle/input/datasets/dschettler8845/brats-2021-task1
```

To run an experiment:

1. Open the notebook matching the desired variant and seed.
2. Make sure the BraTS 2021 Task 1 data is available at the expected input path.
3. Run the cells from top to bottom on a CUDA-enabled runtime.
4. Change `Config.output_root` or set `MOREG_OUTPUT` if the default output location is unsuitable.

The first cell installs MONAI. The notebooks also use PyTorch, NumPy, SciPy, scikit-learn, pandas, Matplotlib, NiBabel, and tqdm. A full run trains for up to 125 epochs and performs full-volume evaluation on 188 test patients, so it is substantially more expensive than the smoke-test configuration.

Each run writes checkpoints and metrics below the selected output directory, including:

```text
<output_root>/full/<variant>/best.pt
<output_root>/full/<variant>/history.csv
<output_root>/full/<variant>/best_validation_patients.csv
<output_root>/full/<variant>/test/
<output_root>/full/exported_models/<variant>_model.pth
```

Set `smoke_test = True` in `Config` for a small end-to-end validation run. This reduces the patch size, epochs, batches, workers, cache size, and bootstrap count; its metrics must not be compared with the full experiment results.

## Reproducibility notes

- Patient splits are persisted in `patient_split.csv` and validated when checkpoints are resumed or loaded.
- The configured seed controls Python, NumPy, and PyTorch RNGs, epoch-specific sampling, and data-loader workers.
- `deterministic` defaults to `False`, so exact bitwise reproducibility is not guaranteed across hardware or CUDA/PyTorch versions.
- The test summary uses a threshold of `0.5`, no test-time augmentation, a 1 mm surface tolerance, and 2,000 bootstrap samples.
- The notebooks currently embed their environment and dataset paths; there is no separate lockfile or standalone training script in this repository.

## Data

The experiments use the BraTS 2021 Task 1 data. Obtain the dataset through its official distribution, follow its licence and usage conditions, and cite the BraTS 2021 dataset publication when publishing results based on these notebooks.
