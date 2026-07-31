# WRTNet-EFWI

Official PyTorch implementation of **WRTNet-EFWI**, a wavelet-refined Transformer framework for physics-informed unsupervised elastic full-waveform inversion.

This repository accompanies the manuscript:

> **A Wavelet-Refined Transformer Network for Physics-Informed Unsupervised Elastic Full-Waveform Inversion**

WRTNet combines a Transformer encoder, parallel CNN decoders, a differentiable wavelet refinement module, and elastic-wave forward modeling. It jointly reconstructs P-wave velocity (`Vp`), S-wave velocity (`Vs`), and density (`ρ`) from multicomponent seismic observations.

## Repository Structure

* `WRTNet.py`: WRTNet architecture with wavelet-based multiscale refinement.
* `networks_transformer.py`: conventional Transformer baseline with plain multicomponent fusion and no wavelet refinement.
* `train_wrtnet.py`: data preparation, forward modeling, model training, evaluation, and result saving.
* `data/model/`: input elastic models used by the training script.
* `datasets/`: dataset-related files and utilities.
* `LICENSE`: repository license.

## Requirements

The implementation requires Python and the following main packages:

* PyTorch
* NumPy
* SciPy
* Matplotlib
* Deepwave
* PyWavelets
* ptwt
* TensorBoard

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Data Preparation

Place the SEAM Arid model files in `data/model/`:

```text
data/model/
├── vp_arid_shallow.npy
├── vs_arid_shallow.npy
└── rho_arid_shallow.npy
```

The file paths, acquisition geometry, sampling parameters, and training settings can be changed in `train_wrtnet.py`.

## Model Selection

Only one model import should be enabled in the training script.

For WRTNet:

```python
from WRTNet import Physics_deepwave, Transfomerdecoder
```

For the conventional Transformer baseline:

```python
from networks_transformer import Physics_deepwave, Transfomerdecoder
```

When using the conventional Transformer baseline, remove the following WRTNet-specific arguments from the model initialization:

```python
use_wave_refine=True
wave_base_ch=24
```

## Training

Run the inversion experiment with:

```bash
python train_wrtnet.py
```

The current configuration uses:

* `transddepth = 12`
* `embed_dim = 256`
* `num_heads = 8`
* `n_blocks_decoder = 4`
* `learning_rate = 1e-4`
* `iterations = 2000`

Checkpoints, estimated elastic models, TensorBoard logs, and figures are saved automatically under the `result/` directory.

## Code Availability

The repository provides the WRTNet implementation, the conventional Transformer baseline, differentiable elastic-wave forward modeling, and the main training workflow for academic research and reproducibility.

## Citation

If you use this repository, please cite:

> Yan, B., Pan, R., and Wang, Y. “A Wavelet-Refined Transformer Network for Physics-Informed Unsupervised Elastic Full-Waveform Inversion.”

The complete bibliographic information will be added after publication.

## License

This project is released under the license provided in the `LICENSE` file.
