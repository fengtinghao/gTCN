# gTCN

Causal dilated TCN (WaveNet-style) renderer for guitar effect modeling, trained
on the paired dry/wet dataset in the sibling `../GTtimbre/data` repo.

## Architecture

```
new dry guitar --> [PerformanceEncoder] --> [renderer, causal dilated TCN] --> predicted wet guitar
                                                      ^ FiLM(z) on every block
                         (dry_ref, wet_ref) --> [EffectEncoder] --> z
```

This is step one of a bigger project: eventually many effects will be trained
this way, the resulting effect-encoder outputs will form a parameter space,
and a hypernetwork will be trained to predict that parameter set directly
from a (dry, wet) pair — at which point `performance_encoder` + `renderer`
are frozen as a shared **backbone**, and only the hypernetwork's predicted
parameters (replacing `effect_encoder`'s forward pass) change per effect.
The module split below exists specifically so that boundary is clean:

- **`PerformanceEncoder`** (`src/model.py`): a couple of causal conv layers
  over the dry input only. Never sees the effect reference — part of the
  frozen backbone in the bigger project.
- **`EffectEncoder`** (`src/model.py`): strided conv stack over a concatenated
  `(dry_ref, wet_ref)` reference crop -> global-average-pooled embedding `z`.
  The reference pair is sampled independently from the training crop, so the
  encoder must learn a general effect embedding rather than copy the target.
  This is the only module a future hypernetwork needs to replace.
- **`TCNRenderer`** (`src/model.py`): WaveNet-style causal residual stack
  (gated activation, dilations `1..2^(layers_per_block-1)` repeated
  `num_blocks` times) over the performance features. Every block is
  FiLM-conditioned on `z`. Feed-forward (non-autoregressive) — a single
  forward pass renders the whole clip, unlike sample-by-sample WaveNet
  generation. Part of the frozen backbone along with `PerformanceEncoder`.
- `GTCNModel.backbone_parameters()` / `.effect_encoder_parameters()` /
  `.freeze_backbone()` expose this split for the future hypernetwork stage.
- One shared model is trained jointly across all 4 timbres (`BritPlexi`,
  `FenderDeluxe`, `FenderDouble`, `minosaur`) instead of one U-Net per timbre —
  better data efficiency, and it's exactly the joint-backbone / per-effect-z
  setup the hypernetwork stage needs. Validation loss (and a best checkpoint)
  is still tracked per effect — see `checkpoints/best_<timbre>.pt` below.
- Loss: multi-resolution STFT loss (spectral convergence + log-magnitude) +
  time-domain L1, the standard combination for neural amp/effect modeling.

### Why this should beat the U-Net baseline

- The U-Net baseline operates on log-mel spectrograms and needs Griffin-Lim
  to get back to a waveform, which discards phase and caps audio quality.
  The TCN renders raw waveform directly end-to-end — no vocoder step.
- Guitar amp/pedal effects (distortion, EQ, compression) are largely
  memoryless-to-short-memory nonlinear waveshaping effects. A causal dilated
  TCN with a ~100-200 ms receptive field is exactly the inductive bias used by
  real-time neural amp modeling work (e.g. WaveNet/TCN amp emulators), and is
  a much better match than 2D image convolutions over a spectrogram.
- FiLM conditioning from a reference-pair encoder (rather than a fixed
  per-timbre embedding) lets the same model be conditioned on a new
  effect/pedal reference at inference time without retraining.

## Data

Expected layout (sibling repo, not copied into this repo):

```
../GTtimbre/data/X/<source>/(...)/<stem>.wav
../GTtimbre/data/y/<timbre>/<source>/(...)/<stem>-<timbre>.wav
```

Dry/wet files are sample-aligned (same length, same sample rate: 44100 Hz
mono). Paths are configurable in `configs/default.yaml`.

## Setup

```bash
conda activate gttcn-py311   # torch 2.6.0+cu126 already installed
pip install -r requirements.txt
```

## Train

Locally (foreground):

```bash
cd src && python train.py --config ../configs/default.yaml
```

On the HPC cluster (`cs-gpu` partition, L40S GPU):

```bash
sbatch scripts/train.slurm
```

Checkpoints go to `checkpoints/{last,best}.pt`, plus `checkpoints/best_<timbre>.pt`
for each effect (`BritPlexi`, `FenderDeluxe`, `FenderDouble`, `minosaur`) — the
model is trained jointly across all effects, but validation loss is tracked
per effect so you can pick the checkpoint that renders a specific effect best.
TensorBoard logs go to `outputs/tensorboard/`.

## Inference

```bash
cd src && python inference.py \
  --checkpoint ../checkpoints/best.pt \
  --input path/to/new_dry.wav \
  --output path/to/predicted_wet.wav \
  --timbre BritPlexi   # or --ref_dry/--ref_wet for a custom reference pair
```

## Project layout

```
configs/default.yaml   # data/model/train hyperparameters
src/dataset.py         # paired dry/wet dataset with random crops
src/model.py           # EffectEncoder + TCNRenderer + GTCNModel
src/losses.py          # multi-resolution STFT + time-domain loss
src/train.py           # training loop, checkpointing, TensorBoard
src/inference.py       # render a new dry clip given a checkpoint + reference
scripts/train.slurm    # SLURM submission script (cs-gpu partition)
scripts/check_gpu.slurm
```