# Neural Style Transfer with AdaIN

A PyTorch-based Neural Style Transfer project that combines the
content structure of one image with the artistic style of another
using Adaptive Instance Normalization (AdaIN).

## First View of the Project

```text
Content Image ──> vgg-19 Encoder ──┐
                                    ├──> AdaIN ──> Decoder ──> Stylized Image 
Style Image ────> vgg-19 Encoder ──┘
```

**VGG extracts features -> AdaIN transfer style statistics -> Decoder reconstructs the image -> Content/style losses train the decoder.**

## About

This project implements Neural Style Transfer using an AdaIN-based
architecture.

The project currently focuses on:
- VGG-19 based feature extraction
- Adaptive Instance Normalization (AdaIN)
- Trainable image decoder
- PyTorch training pipeline
- Content and style loss
- Image-to-image style transfer

## Development Journey

1. Understand Neural Style Transfer
2. Implement feature statistics and AdaIN
3. Build the VGG-19 Encoder
4. Freeze the encoder
5. Build the decoder
6. Build dataset/Dataloaders
7. Connect VGG -> AdaIN -> Decoder
8. Re-encode generated images
9. Calculate content/style losses
10. Train and checkpoint the decoder
11. Generate and inspect outputs
12. Refactor notebook code into resuable modules

## How It Works

1. A content image is provided.
2. A style image is provided.
3. VGG-19 extracts feature representations.
4. AdaIN transfers the style statistics to the content features.
5. The decoder reconstructs the stylized image.
6. The resulting image is returned as the output.

## Current Status

| Component | Status |
|---|---|
| Learning NST research | ✅ |
| AdaIN implementation | ✅ |
| VGG-19 encoder | ✅ |
| Decoder | ✅ |
| Training pipeline | ✅ |
| Initial model training | ✅ |
| Inference pipeline | 🚧 |
| FastAPI integration | 🚧 |
| Deployment | 🚧 |

## Requirements

The notebook uses Python, PyTorch, torchvision, Pillow and tqdm, along with standard Python utilities.

### Installation
```
git clone repo_url
cd neural_style_transfer
pip install -r requirements.txt
```

## Training

Detailed training information is documented separately:
→ [notebook/TRAINING.md](notebook/TRAINING.md)

The original training implementation is available in:
→ [notebook/nst-code-train-01.ipynb](notebook/nst-code-train-01.ipynb)

## Dataset

For Content Images : we use COCODATASET
* [test cocodataset](http://images.cocodataset.org/zips/test2017.zip)

For Style Iamges: kaggle [PAINTER-BY-NUMBERS](https://www.kaggle.com/competitions/painter-by-numbers/data?select=train_1.zip)

## Repository Structure


## Usage

user can provide:
- Content image
- Style image

and receive:
- Stylized output image

## Documentation

- (Training Documentation)[]