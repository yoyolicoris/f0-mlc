<div align="center">

# f0-mlf: Robust and Lightweight F0 Estimation Through Mid-Level Fusion of DSP-Informed Features

<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

This is a Pytorch code repository accompanying the following paper. If you use this code in your research, please cite:

```bibtex
@inproceedings{StrahlM26_MidLevelFusionF0_ICASSP,
  author    = {Sebastian Strahl and Meinard M{\"u}ller},
  title     = {Robust And Lightweight {F0} Estimation Through Mid-Level Fusion of {DSP}-Informed Features},
  booktitle = {Proceedings of the {IEEE} International Conference on Acoustics, Speech, and Signal Processing ({ICASSP})},
  address   = {Barcelona, Spain},
  year      = {2026},
  pages     = {16087--16091},
  doi       = {10.1109/ICASSP55912.2026.11463185},
}
```

For details and references, please check out this paper.

## Installation (Conda)

```bash
# clone project
git clone https://github.com/groupmm/f0-mlf
cd f0-mlf

# create conda environment and install dependencies
conda env create -f environment.yaml

# activate conda environment
conda activate f0-mlf
```

## Dataset preparation

1. Copy `.env.example` to `.env` and set `DATA_DIR` to your data base path.

2. Training is done on MIR-1K. Download the dataset from [here](http://mirlab.org/dataset/public/) and place it under `$DATA_DIR/MIR-1K/` with the following structure:

   ```
   $DATA_DIR
   └── MIR-1K
       ├── PitchLabel
       └── Wavfile
   ```

3. Convert the F0 annotations from MIDI pitch to Hz (and add a time column). This creates a new `PitchLabel_csv` folder alongside `PitchLabel`:

   ```bash
   python scripts/prepare_mir1k.py
   ```


## How to use

Train model with default configuration

```bash
python src/train.py
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

To evaluate a trained model (e.g. on MIR-1K test subset, 0 dB SNR, last checkpoint):

```bash
python src/eval.py model=mlf data=mir1k model.snr_range=0 ckpt_path=logs/train/runs/YYYY-MM-DD_HH-MM-SS/checkpoints/last.ckpt
```


## Contribution
Automated code style checks via [pre-commit](https://pre-commit.com/):

```bash
pre-commit install
pre-commit run --all-files
```


## License
This code is published under an [MIT license](LICENSE).


## Acknowledgments

This codebase builds upon the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) and [mir_eval](https://github.com/mir-evaluation/mir_eval), parts of which are re-distributed here. We thank the authors and maintainers of both projects for their work.

This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Grant No. 500643750 (MU 2686/15-1). The authors are with the [International Audio Laboratories Erlangen](https://audiolabs-erlangen.de/), a joint institution of the [Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)](https://www.fau.eu/) and [Fraunhofer Institute for Integrated Circuits IIS](https://www.iis.fraunhofer.de/en.html).
