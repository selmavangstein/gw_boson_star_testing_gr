# gw_boson_star_testing_gr

Boson-star injection generation and ringdown analysis. The repo is organized around active notebooks, active scripts, preserved results, legacy investigations, and ignored local state.

## Structure

- `notebooks/` contains the maintained injection, ringdown, and plotting notebooks.
- `scripts/` contains automation helpers for running the maintained notebooks.
- `inputs/` contains tracked waveform inputs such as `inputs/chombo/`.
- `results/` contains preserved in-repo scientific outputs and archived reference plots.
- `legacy/` contains old investigations, taper experiments, and older plotting notebooks.
- `local/` is ignored and holds environments, large dependencies, and scratch/generated files.

## Maintained Workflow

### 1. Generate injections

Run [barebones_injection.ipynb](/home/selmavangstein/mastersproject/gw_boson_star_testing_gr/notebooks/barebones_injection.ipynb).

Shared output location:

- `../datadir/bilby_output/`

Typical files written there:

- `<prefix>_injection_data.csv`
- `<prefix>_injection_metadata.csv`
- `<prefix>_H1_injection_data.h5`
- `<prefix>_L1_injection_data.h5`
- `hp.csv`
- `hc.csv`

### 2. Run ringdown fits

Run [bs_ringdown.ipynb](/home/selmavangstein/mastersproject/gw_boson_star_testing_gr/notebooks/bs_ringdown.ipynb).

Inputs:

- `../datadir/bilby_output/<prefix>_injection_data.csv`
- `../datadir/bilby_output/<prefix>_injection_metadata.csv`

Ringdown result output location:

- `../datadir/ringdown_output/`
- `results/ringdown/plots/`

### 3. Plot products

Maintained plotting notebooks:

- [plot_ringdown_results.ipynb](/home/selmavangstein/mastersproject/gw_boson_star_testing_gr/notebooks/plot_ringdown_results.ipynb)
- [plot_ringdown_mode_corner_plots.ipynb](/home/selmavangstein/mastersproject/gw_boson_star_testing_gr/notebooks/plot_ringdown_mode_corner_plots.ipynb)
- [plot_barebones_injection_outputs.ipynb](/home/selmavangstein/mastersproject/gw_boson_star_testing_gr/notebooks/plot_barebones_injection_outputs.ipynb)

Ringdown plot outputs are preserved under `results/ringdown/plots/`. Scratch or ad hoc exports can still go under `local/generated/`.

## Preserved Material

- `results/archive/` holds preserved result-like plots from earlier work.
- `legacy/taper/plots/` holds preserved taper-method plots.
- `legacy/taper/notebooks/`, `legacy/ringdown_tests/notebooks/`, and `legacy/old_plots/notebooks/` hold older notebook references.

## Local-Only Files

Keep these under ignored `local/` only:

- `local/env/`
- `local/deps/`
- `local/generated/`

## Handoff To `tdinf_example`

The maintained `tdinf_example/code/script.sh` reads shared bilby files from:

- `../datadir/bilby_output/`

## Notes

- The shared external `datadir/` remains the canonical interface between bilby and tdinf.
- The reorganization is structural only; it does not change the intended analysis workflow.
