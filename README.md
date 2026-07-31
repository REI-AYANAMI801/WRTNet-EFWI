# WRTNet-EFWI

Official PyTorch implementation of **WRTNet-EFWI**, a wavelet-refined Transformer framework for physics-informed unsupervised elastic full-waveform inversion.

This repository accompanies the manuscript:

> **A Wavelet-Refined Transformer Network for Physics-Informed Unsupervised Elastic Full-Waveform Inversion**

WRTNet combines a Transformer encoder, parallel CNN decoders, a differentiable wavelet refinement module, and elastic-wave forward modeling. It jointly reconstructs P-wave velocity (`Vp`), S-wave velocity (`Vs`), and density (`ρ`) from multicomponent seismic observations.

## Repository Structure

* `WRTNet.py`: WRTNet architecture with wavelet-based multiscale refinement.
* `networks_transformer.py`: conventional Transformer baseline with plain multicomponent fusion and no wavelet refinement.
* `main.py`: data preparation, forward modeling, model training, evaluation, and result saving.
* `data/model/`: SEAM Arid elastic model files used by the training script.
* `requirements.txt`: Python dependencies required to run the project.
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

Install the required packages with:

```bash
pip install -r requirements.txt
```

For GPU execution, ensure that the installed PyTorch version is compatible with the local CUDA environment.

## Data Preparation

The SEAM Arid shallow elastic model files are provided in:

```text
data/model/
├── vp_arid_shallow.npy
├── vs_arid_shallow.npy
└── rho_arid_shallow.npy
```

The model files contain P-wave velocity, S-wave velocity, and density, respectively. The density model is converted from `g/cm³` to `kg/m³` within the training script.

The data paths, acquisition geometry, temporal and spatial sampling parameters, and training settings can be modified in `main.py`.

## Model Selection

Only one model import should be enabled in `main.py`.

To run WRTNet:

```python
from WRTNet import Physics_deepwave, Transfomerdecoder
```

To run the conventional Transformer baseline:

```python
from networks_transformer import Physics_deepwave, Transfomerdecoder
```

When using the conventional Transformer baseline, comment out the WRTNet import and remove the following WRTNet-specific arguments from the model initialization:

```python
use_wave_refine=True,
wave_base_ch=24,
```

The conventional Transformer baseline uses plain multicomponent fusion and does not include the wavelet refinement module.

## Training

Run the inversion experiment with:

```bash
python main.py
```

The current training configuration uses:

* Transformer depth: `transddepth = 12`
* Embedding dimension: `embed_dim = 256`
* Number of attention heads: `num_heads = 8`
* Number of CNN decoder blocks: `n_blocks_decoder = 4`
* Learning rate: `1e-4`
* Number of iterations: `2000`

The default configuration performs inversion on the SEAM Arid shallow model using multicomponent elastic seismic data generated through differentiable forward modeling.

## Outputs

The following outputs are saved automatically under the `result/` directory:

* Estimated `Vp`, `Vs`, and density models
* Training and evaluation losses
* Model checkpoints
* TensorBoard logs
* Initial and inverted model figures
* Held-out receiver loss curves, when receiver masking is enabled

TensorBoard logs can be viewed with:

```bash
tensorboard --logdir result
```

## Code Availability

This repository provides:

* The complete WRTNet architecture
* A conventional Transformer baseline
* Differentiable elastic-wave forward modeling
* Multiscale wavelet-domain data misfit
* Joint `Vp`, `Vs`, and density inversion
* Training, evaluation, checkpointing, and visualization workflows

The source code is provided for academic research and reproducibility.

## Citation

If you use this repository, please cite:

> Yan, B., Pan, R., and Wang, Y. “A Wavelet-Refined Transformer Network for Physics-Informed Unsupervised Elastic Full-Waveform Inversion.”

The complete bibliographic information and DOI will be added after publication.

## License

This project is released under the MIT License. See the `LICENSE` file for details.
