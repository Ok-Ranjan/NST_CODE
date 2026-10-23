# Real-Time Neural Style Transfer (AdaIN)

A lightweight PyTorch implementation of arbitrary Neural Style Transfer based on **Adaptive Instance Normalization (AdaIN)**. It blends the structural content of any photo with the colors, strokes, and textures of any artwork in real time.

---

## 🚀 Live Demo

Try the app directly in your browser:

- **Hugging Face Spaces (ZeroGPU):** [Live Demo on Hugging Face](https://ok-ranjan-nst-project.hf.space/) *(Fast GPU inference)*
- **Render (Free Web Service):** [Live Demo on Render](https://nst-project-vruz.onrender.com/) *(CPU deployment with async job queue)*

> *Note on Render free tier: The service spins down when idle and may take 30–50 seconds to wake up on the first request.*

---

## 💡 How It Works

Instead of running slow optimization loops for each image (like traditional style transfer), AdaIN transfers style in a single feed-forward step:

```text
Content Image ──> [ Frozen VGG-19 ] ──> Content Features ──┐
                                                           ├──> [ AdaIN Layer ] ──> [ Trained Decoder ] ──> Stylized Output
Style Image   ──> [ Frozen VGG-19 ] ──> Style Features   ──┘
```

1. **Feature Extraction:** A pre-trained, frozen VGG-19 extracts feature maps up to layer `relu4_1`.
2. **AdaIN Layer:** It strips the content image's original lighting and color, then scales and shifts it using the mean and variance (style signature) of the artwork.
3. **Decoder Reconstruction:** A custom convolutional decoder (trained on ~50k images) maps the stylized features back into an RGB photo.
4. **Style Strength ($\alpha$):** A slider interpolates between original content features and stylized features ($0.0 \le \alpha \le 1.0$), letting you control how strong the painting effect is.

---

## ✨ Features

- **Arbitrary Style Transfer:** Works with any pair of images instantly — no fine-tuning or retraining per style.
- **Adjustable Style Weight ($\alpha$):** Fine-tune the blend from subtle color grading to full artistic transformation.
- **Asynchronous Worker Queue:** Built with background threading and polling on the Flask backend so requests never hit Render's 30-second proxy timeout during CPU inference.
- **Interactive Web UI:** Clean dark-mode interface with drag-and-drop upload, preset example images, and side-by-side comparison.

---

## 🛠️ Tech Stack

- **Model:** PyTorch, torchvision (VGG-19 encoder + custom mirror Decoder)
- **Backend:** Flask, Flask-WTF, Gunicorn, background threading
- **Frontend:** HTML5, CSS3 (glassmorphic dark UI), Vanilla JavaScript
- **Training:** Kaggle GPU environment (50 epochs on MS-COCO + WikiArt)
- **Hosting:** Hugging Face Spaces (ZeroGPU) & Render Free Tier

---

## 📁 Project Structure

```text
NST_CODE/
├── main/
│   ├── app.py                # Flask web server & async inference worker
│   ├── decoder.pth           # Trained decoder weights (epoch 50)
│   ├── vgg_normalised.pth    # Pretrained VGG-19 encoder weights
│   ├── src/
│   │   ├── encoder.py        # Sliced VGG-19 encoder (relu1_1 to relu4_1)
│   │   ├── decoder.py        # Custom decoder network
│   │   ├── adain.py          # Adaptive Instance Normalization logic
│   │   └── utils.py          # Image transformations & helper utilities
│   ├── templates/            # Web interface (index.html)
│   └── static/               # Assets & uploaded image cache
├── notebook/
│   ├── TRAINING.md           # In-depth training documentation & loss analysis
│   ├── nst-code-train-01.ipynb       # Initial training experiment
│   └── nst-code-train-epoch50.ipynb  # 50-epoch Kaggle training run
├── outputs/                  # Sample outputs across training epochs
├── Procfile                  # Gunicorn entrypoint for Render
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 💻 Local Setup & Running

### 1. Clone the repository
```bash
git clone https://github.com/Ok-Ranjan/NST_CODE.git
cd NST_CODE
```

### 2. Install dependencies
It is recommended to use a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the web app
```bash
cd main
python app.py
```
Open your browser and navigate to `http://localhost:5000`.

---

## 🏋️ Training Details

The decoder was trained for **50 epochs** on Kaggle GPU using:
- **Content:** 40,670 images from [MS-COCO 2017](http://images.cocodataset.org/zips/test2017.zip)
- **Style:** 11,025 artwork images from [WikiArt / Painter-by-Numbers](https://www.kaggle.com/competitions/painter-by-numbers)
- **Optimizer:** Adam with inverse decay scheduler (`lr = 1e-4`)
- **Loss:** Content MSE loss at `relu4_1` + multi-layer Style loss at `relu1_1`, `relu2_1`, `relu3_1`, `relu4_1`

For the complete loss formulation, hyperparameter tables, and convergence logs, check [notebook/TRAINING.md](notebook/TRAINING.md).

---

## 📜 References

- Xun Huang & Serge Belongie. *"Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"*, ICCV 2017. [arXiv:1703.06868](https://arxiv.org/abs/1703.06868)
- Leon A. Gatys, Alexander S. Ecker, & Matthias Bethge. *"A Neural Algorithm of Artistic Style"*, CVPR 2016. [arXiv:1508.06576](https://arxiv.org/abs/1508.06576)