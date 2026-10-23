# Training Details: Real-Time Neural Style Transfer with AdaIN

This document explains the training pipeline used to train the decoder model, based on the notebooks in this directory (`nst-code-train-01.ipynb` and `nst-code-train-epoch50.ipynb`).

---

## 1. Core Idea

Original Neural Style Transfer (Gatys et al.) required running an optimization process for hundreds of iterations for *every single image*, taking minutes to produce one stylized output.

This project implements **Arbitrary Style Transfer using Adaptive Instance Normalization (AdaIN)** ([Huang & Belongie, 2017](https://arxiv.org/abs/1703.06868)):
1. We keep a pre-trained **VGG-19 encoder completely frozen**.
2. Feature representations of both content and style images are extracted.
3. The **AdaIN layer** aligns the channel-wise mean and variance of the content features to match those of the style features.
4. We train **only the decoder network** to invert these stylized feature maps back into a clean RGB image.

Once trained, style transfer takes a single forward pass (< 1 second on GPU, ~2–4 seconds on CPU).

---

## 2. Architecture

```text
Content Image (c) ──> [ Frozen VGG-19 ] ──> f(c) ──┐
                                                    ├──> [ AdaIN ] ──> t ──> [ Trained Decoder ] ──> Stylized Image (g)
Style Image (s)   ──> [ Frozen VGG-19 ] ──> f(s) ──┘
```

### The Encoder (Frozen VGG-19)
- Standard VGG-19 backbone pre-trained on ImageNet (`vgg_normalised.pth`).
- Sliced up to `relu4_1` (first 31 layers) and completely frozen (`requires_grad = False`).
- Split into 4 multi-scale feature stages:
  - `enc_1`: up to `relu1_1` (captures low-level brushstrokes and fine textures)
  - `enc_2`: up to `relu2_1`
  - `enc_3`: up to `relu3_1` (captures medium textures and patterns)
  - `enc_4`: up to `relu4_1` (captures higher-level semantic content layout)

### The AdaIN Layer
Given content features $f(c)$ and style features $f(s)$:

$$\text{AdaIN}(f(c), f(s)) = \sigma(f(s)) \cdot \left( \frac{f(c) - \mu(f(c))}{\sigma(f(c))} \right) + \mu(f(s))$$

- **Centering and dividing by std** removes the content image's original lighting, color casting, and texture contrast.
- **Multiplying by style std** injects the texture sharpness and stroke variance of the artwork.
- **Adding style mean** shifts the color palette and illumination to match the artwork.

### The Decoder (Trained from scratch)
The decoder mirrors the VGG-19 encoder up to `relu4_1`, converting 512-channel feature representations back into a 3-channel RGB image.

Key design choices:
- **Nearest-Neighbor Upsampling + Conv2d** instead of `ConvTranspose2d` to completely eliminate checkerboard artifacts.
- **ReflectionPad2d** instead of zero padding to prevent dark border artifacts around image boundaries.
- No normalization layers (batch norm / instance norm) inside the decoder, as normalization would wipe out the newly transferred style statistics.

---

## 3. Training Setup & Datasets

| Setting | Value |
|---|---|
| **Content Dataset** | MS-COCO 2017 test set (~40,670 images) |
| **Style Dataset** | WikiArt / Painter-by-Numbers train set (~11,025 images) |
| **Image Resolution** | Resized to 256×256, randomly cropped to 224×224 |
| **Batch Size** | 16 (689 batches per epoch) |
| **Total Epochs** | 50 epochs |
| **Optimizer** | Adam ($\beta_1 = 0.9, \beta_2 = 0.999$) |
| **Initial Learning Rate** | `1e-4` |
| **LR Scheduler** | Inverse time decay: $\text{lr} = \frac{\text{lr}_0}{1 + 5\times 10^{-5} \times \text{epoch}}$ |
| **Hardware** | Kaggle GPU (~11.5 minutes per epoch) |

---

## 4. Loss Functions

The training loss is a weighted sum of content loss and style loss:

$$\mathcal{L} = \mathcal{L}_c + \lambda_s \cdot \mathcal{L}_s$$

### Content Loss ($\mathcal{L}_c$)
Ensures the generated image $g$ keeps the structural layout from the AdaIN target feature $t$:

$$\mathcal{L}_c = \| f_{\text{relu4\_1}}(g) - t \|_2^2$$

- **Content Weight**: `1.0`

### Style Loss ($\mathcal{L}_s$)
Ensures the generated image shares the style's statistical signature across multiple scales. We compute the MSE of channel-wise mean and standard deviation across four VGG levels (`relu1_1`, `relu2_1`, `relu3_1`, `relu4_1`):

$$\mathcal{L}_s = \sum_{i=1}^{4} \left( \| \mu(f_i(g)) - \mu(f_i(s)) \|_2^2 + \| \sigma(f_i(g)) - \sigma(f_i(s)) \|_2^2 \right)$$

- **Style Weight ($\lambda_s$)**: `10.0`

---

## 5. Training Progression & Results

The model was trained for 50 full epochs. Here is the convergence progression logged across key milestones:

| Epoch | Total Loss | Content Loss | Style Loss | Notes |
|---|---|---|---|---|
| **Epoch 1** | ~7.85 | ~0.72 | ~7.13 | Initial color blending begins |
| **Epoch 10** | ~2.54 | ~0.51 | ~2.03 | Distinct textures and brushstrokes emerge |
| **Epoch 20** | ~2.12 | ~0.45 | ~1.67 | Consistent style fidelity and edge preservation |
| **Epoch 35** | ~1.85 | ~0.42 | ~1.43 | Sharp contrast and reduced color distortion |
| **Epoch 50** | **1.69** | **0.39** | **1.29** | Final converged model checkpoint (`decoder.pth`) |

Sample visual grids were saved every 10 epochs ([output_10.png](outputs/nst-code-train-epoch50_outputs/output_10.png)) through ([output_50.png](outputs/nst-code-train-epoch50_outputs/output_50.png)) in [outputs/nst-code-train-epoch50_outputs/](outputs/nst-code-train-epoch50_outputs/), showing progressive refinement of texture details while preserving semantic content structure.

---

## 6. How to Re-train or Fine-Tune

### Option A: Run via Jupyter / Kaggle
Open `notebook/nst-code-train-epoch50.ipynb` directly on Kaggle or a local GPU environment and run from top to bottom.

### Option B: Run via Modular Script
You can also run training using `main/src/inference.py`:

```bash
python main/src/inference.py \
  --content_dir ./content_data_demo \
  --style_dir ./style_data_demo \
  --vgg ./main/vgg_normalised.pth \
  --batch_size 4 \
  --epochs 5 \
  --lr 1e-4 \
  --style_weight 10
```