# LaSyNet — Latent Symbiosis for Joint Medical Image Enhancement and Segmentation
!(https://github.com/Reconsider80/LaSyNet/blob/main/LaSyNet.png)

This repository contains a reference PyTorch implementation of **LaSyNet** from the paper:

> **Latent Symbiosis: Adapter-Guided Diffusion Interaction with Gated Cross-Task Routing for Joint Medical Image Enhancement and Segmentation** (AAAI 2027 / under review).

LaSyNet unifies medical image enhancement and segmentation into a single **Rectified-Flow** generative framework.  A frozen latent UNet backbone is adapted via lightweight, task-specific **Adapters**, and a timestep-aware **Gated Symbiotic Information Interaction (G-SII)** module dynamically routes bidirectional features between the two tasks.

---

## Key features

- **Rectified-Flow backbone** — straight-line ODE trajectory between noise and clean latent data instead of curved diffusion paths.
- **Adapter-guided PEFT** — only the task-specific adapters and projection layers are trained; the generative backbone can be kept frozen.
- **G-SII module** — time-conditioned cross-attention gates that exchange structural priors (`Seg → Enh`) and texture details (`Enh → Seg`).
- **Joint multi-task training** — simultaneous enhancement and segmentation with RF velocity-field losses plus pixel-space supervision.
- **Modality-specific degradation** — motion + Rician noise (ACDC/MRI), Poisson photon starvation (KiTS23/CT), and Gamma speckle (TN3K/ultrasound).

---

## Repository structure

```
LaSyNet/
├── configs/
│   └── default.yaml              # example configuration
├── lasynet/
│   ├── models/
│   │   ├── vae.py                # first-stage VAE/AE
│   │   ├── seg_encoder.py        # Flq encoder + mask encoder + segmentation decoder
│   │   ├── unet.py               # time-conditional UNet backbone
│   │   ├── adapter.py            # bottleneck adapters
│   │   ├── gsii.py               # Gated Symbiotic Information Interaction
│   │   └── lasynet.py            # full model + RF training/inference logic
│   ├── data/
│   │   ├── datasets.py           # ACDC, KiTS23, TN3K data loaders
│   │   └── degradation.py        # modality-specific artifact injection
│   └── utils/
│       ├── config.py             # YAML helpers
│       └── metrics.py            # PSNR, SSIM, Dice, mIoU
├── train.py                      # training script
├── eval.py                       # evaluation script
├── requirements.txt
└── README.md
```

---

## Installation

### Option 1: use the existing `py310` conda environment

The current server already has a `py310` environment with all required packages (PyTorch 2.0.1+cu117, torchvision, scikit-image, nibabel, etc.).  Just activate it:

```bash
cd /data2/chenying/LaSyNet
conda activate py310
```

Then run the demo or training scripts directly.

### Option 2: create a fresh `lasynet` environment

```bash
git clone <repo-url>
cd LaSyNet
conda env create -f environment.yml
conda activate lasynet
```

Or, if you prefer pip:

```bash
conda create -n lasynet python=3.10
conda activate lasynet
pip install -r requirements.txt
```

> The implementation uses PyTorch only.  `nibabel` is optional and only required for 3D NIfTI datasets (ACDC / KiTS23).

---

## Data preparation

### TN3K (2D thyroid ultrasound)

Organize the dataset as:

```
data/TN3K/
├── train/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Masks should be binary PNGs (`0` = background, `255` or `>0` = foreground).

### ACDC / KiTS23 (3D volumes)

Provide text files listing absolute paths to the NIfTI images and masks, then reference them in the config:

```yaml
train_images: /path/to/acdc_train_images.txt
train_masks:  /path/to/acdc_train_masks.txt
test_images:  /path/to/acdc_test_images.txt
test_masks:   /path/to/acdc_test_masks.txt
cache_volumes: true
```

Each line of a split file should contain one `.nii` or `.nii.gz` path.

---

## Training

```bash
python train.py --config configs/default.yaml --output_dir checkpoints/tn3k
```

Resume from a checkpoint:

```bash
python train.py --config configs/default.yaml --output_dir checkpoints/tn3k --resume checkpoints/tn3k/checkpoint_epoch_010.pt
```

Key hyperparameters in `configs/default.yaml`:

| Parameter | Meaning |
|-----------|---------|
| `beta` | Weight of the segmentation RF loss |
| `lambda_seg` | Weight of the pixel-space segmentation loss |
| `freeze_backbone` | If `true`, only adapters / G-SII / decoders are trained |
| `num_steps` | Euler ODE steps at inference (paper uses 25) |
| `seg_downsample_steps` | Must equal `len(vae_channel_mult) - 1` (VAE spatial factor) |

---

## Evaluation

```bash
python eval.py --config configs/default.yaml --checkpoint checkpoints/tn3k/checkpoint_epoch_100.pt --output_dir outputs/tn3k
```

Override the number of ODE sampling steps:

```bash
python eval.py --config configs/default.yaml --checkpoint ... --output_dir outputs/tn3k --num_steps 10
```

Metrics are saved to `outputs/tn3k/metrics.json`.

---

## Using a pretrained diffusion backbone

The default VAE and UNet are small trainable networks for demonstration.  To match the paper more closely, replace the default `Autoencoder` with a pretrained medical/stable-diffusion VAE and set `freeze_backbone: true` after loading a pretrained UNet into `models/lasynet.py`.  The interface is:

```python
z = model.vae.get_latent(x)      # deterministic latent encoding
x_hat = model.vae.decode(z)      # decode to image space
v = model.backbone(z_t, t, adapters)  # velocity field
```

The `LaSyNet` constructor will automatically create the right number of adapters for every ResBlock of the supplied UNet.

---

## Implementation notes / limitations

- This is a **reference implementation**.  The exact numbers in the paper were obtained with a large pretrained latent diffusion backbone and full training on ACDC, KiTS23, and TN3K.
- The shipped VAE is a small autoencoder; for research-grade results, pretrain or replace it with a medical-domain VAE/AutoencoderKL.
- The G-SII module implements the routing equations (3)–(7) exactly as described in the paper.
- Inference uses a deterministic forward Euler ODE solver.  You can swap in higher-order solvers (e.g. RK4) without changing the model.

---

## Citation

```bibtex
@article{chen2026lasynet,
  title={LaSyNet: Latent Symbiosis: Adapter-Guided Diffusion Interaction with Gated Cross-Task Routing for Joint Medical Image Enhancement and Segmentation},
  author={Ying Chen and others},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

This code is released for academic/research purposes.  Please refer to the original paper and dataset licenses before using the model clinically.
