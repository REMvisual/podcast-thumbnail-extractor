# Training Custom Categories

Train your own category detectors so the extractor can find exactly the kind of frames you care about — software UIs, finished art, product shots, pet photos, whatever your content shows.

This guide is **UI-first**: the default and recommended flow is the `/#config` web interface launched by `config.bat`. A command-line appendix is included at the end for automation and power users.

---

## Why train custom categories?

Different content types have opposing quality signals. You cannot train one model that scores both well:

- **UI / interface content** (node graphs, code editors, software panels) — the model should reward **clarity and readability**.
- **Art / output content** (renders, fullscreen visuals, motion graphics) — the model should reward **visual impact and aesthetic polish**.
- **Mixed content** (process videos showing interface + output) — the model should reward frames that score well in both.

Training a separate model per category keeps those signals cleanly separated. The built-in `screens-ui` and `screens-art` categories are starting points — the real power is defining your own.

---

## What you need

Before you train, collect images:

- **50+ good examples** (70-100 is ideal) — frames you would happily pick as a thumbnail
- **20-30 bad examples** — frames that superficially match the category but are poor (blurry, cluttered, cropped wrong, off-brand)
- A dedicated folder with this structure:

```
training_data/your-category/
├── good/
│   ├── example_01.jpg
│   ├── example_02.jpg
│   └── ... (50+ images)
└── bad/
    ├── bad_01.jpg
    └── ... (20-30 images)
```

**File formats:** `.jpg`, `.jpeg`, `.png`. No size requirement — the trainer resizes to 224×224 internally.

See [`IMAGE_SOURCES.md`](IMAGE_SOURCES.md) for legal, royalty-free sources of training images.

---

## Training via the UI (recommended)

### 1. Launch the config UI

From the repo root:

```bash
config.bat
```

This starts the Flask server (default port `5000`) and opens your browser to `http://localhost:5000/#config`. The category manager lives there.

On Unix:

```bash
./run.sh  # or: python src/app.py
# then open http://localhost:5000/#config
```

### 2. Click "Add Category"

Fill in the form:

| Field | Example | Notes |
|-------|---------|-------|
| **Category ID** | `product-shots` | lowercase, hyphens only, unique |
| **Label** | `Product Shots` | what users see in the dropdown |
| **Icon** | `📦` | any single emoji |
| **Training data folder** | `training_data/product-shots/` | must contain `good/` and `bad/` subfolders |

The form validates the folder structure before saving. If `good/` has fewer than ~20 images, you will get a warning — the model will train but accuracy will suffer.

### 3. Click "Train"

Training runs as a background job. The UI shows live progress via Server-Sent Events (SSE):

- **Epoch counter** (e.g. `Epoch 5 / 20`)
- **Train loss** and **val loss** per epoch
- **Best model saved** notices when val loss improves

A typical category with 80 good + 25 bad images trains in 2-5 minutes on CPU, under a minute on a modern GPU. When done, the UI shows:

```
Training complete
  Best val loss: 0.094
  Model saved to: models/product-shots.pth
```

The category is now available on the main extraction page.

### 4. Use your category

Go to `http://localhost:5000/` (the main page). Upload a video, pick your new category from the dropdown, and extract. The trained model scores each candidate frame, and the top-N highest-scoring frames are saved.

### 5. Iterate

Training accuracy improves with more and better examples. After extracting a batch, review the results — if the model picked frames you would not, add them to `bad/`; if it missed obvious good frames, add those to `good/`. Then open the category in `/#config` and click **Retrain**.

---

## Understanding the output

The trainer produces a single `.pth` file — a PyTorch state dict for a ResNet50-backed regression head that outputs a 0.0-1.0 quality score per image. At inference time, the extractor:

1. Samples candidate frames from the video (every N seconds).
2. Runs each candidate through every enabled category's model.
3. For categories with multiple enabled models (e.g. "Mixed" screen mode), blends scores 50/50.
4. Keeps the top-N globally by score.

Scores you should expect to see on held-out data:

- **0.8-1.0** — model is confident this is a "good" category frame
- **0.5-0.8** — passable, likely to make the shortlist on easier videos
- **0.0-0.5** — model rejects it

---

## Tips

### Quality over quantity
100 carefully chosen images beat 500 noisy ones. If you are unsure an image belongs in `good/`, leave it out. Borderline examples confuse the model.

### Keep good and bad comparable
Bad examples should look superficially like the category — same subject, same software, same context — but be poor quality. If `good/` is all "TouchDesigner node graphs" and `bad/` is all "cat photos", the model learns "is this a node graph?" not "is this a **good** node graph?"

### Avoid bias
If every "good" image is dark-themed and every "bad" image is light-themed, the model learns to pick dark images regardless of content. Diversify across:
- Color schemes (light and dark)
- Zoom levels (but all readable)
- Subjects within the category (multiple software, multiple artists, multiple scenes)

### Balance the set
Roughly **3:1 good-to-bad** is a healthy ratio. All-good datasets train a model that thinks everything is great. All-bad datasets train a model that hates everything.

### Crop out irrelevant UI
For art categories especially, crop Instagram posts, YouTube UI, and window chrome out before adding to `good/`. The model should learn about the art, not the screenshot border.

### Epochs: 15-25
The default is **20 epochs**. More is rarely better — you start overfitting (val loss climbs while train loss keeps falling). If val loss plateaus early, the model is saturated — add more training data rather than more epochs.

---

## Troubleshooting

### Out of memory during training
Lower the batch size (CLI: `--batch-size 4`; UI: batch size defaults to 8, which is safe for 8 GB GPU / any CPU). If you still OOM on CPU, close other apps.

### Training is very slow
Without a GPU, training on CPU takes 2-5 minutes per category. Install a CUDA-capable PyTorch build if you have an NVIDIA GPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Val loss stays flat or climbs
You are overfitting or under-informed. Options:
- Add more good/bad examples (especially more diverse ones).
- Reduce epochs (try 10 instead of 20).
- Verify your labels — is a "good" image actually mislabeled as "bad"?

### Model scores everything high (or everything low)
Your good/bad distributions are too similar, or you have class imbalance. Re-check:
- Do your `bad/` examples clearly differ from your `good/` ones?
- Is one folder 10x bigger than the other?

### Category shows up but ranks frames poorly
Bad training signal. Retrain with more diverse and more carefully labeled examples. A healthy category has 50+ good and 20+ bad examples drawn from multiple sources.

### "CUDA out of memory" mid-run
Restart Python, lower batch size, close other GPU consumers (browsers, other ML jobs).

### Model file is larger than expected (~100 MB)
Normal. ResNet50 weights dominate the file. If size matters (you are distributing the model), we have planned smaller backbones in a future release.

---

## FAQ

**Q: Can I use the same training data for multiple categories?**
No — each category needs its own folder. Categories learn opposing signals and sharing data defeats the purpose.

**Q: Can I combine two trained categories at inference time?**
Yes — the "Mixed" mode in the UI blends two models 50/50. Pick it when your video contains both categories (e.g. tutorial with interface AND render outputs).

**Q: Do I need a GPU?**
No. CPU training works fine for dataset sizes under ~200 images. GPU is a convenience, not a requirement.

**Q: How do I delete a category?**
In `/#config`, click the trash icon next to the category. The `.pth` file is left on disk — delete it manually if you want to reclaim space.

**Q: Can I retrain without losing my old model?**
Yes — retraining overwrites the `.pth` only if training completes successfully. If training errors out mid-run, the old model stays.

---

## Appendix: Training from the command line

For automation, CI pipelines, or users who prefer the terminal, the trainer is also invocable directly.

### Basic command

```bash
python src/training.py \
  --training-data training_data/your-category/ \
  --output models/your_category.pth \
  --epochs 20
```

### Arguments

| Flag | Default | Purpose |
|------|---------|---------|
| `--training-data` | *(required)* | Folder containing `good/` and `bad/` subfolders |
| `--output` | *(required)* | Where to write the trained `.pth` file |
| `--epochs` | `10` | Number of training epochs (15-25 recommended) |
| `--batch-size` | `8` | Batch size; reduce to `4` or `2` if OOM |
| `--learning-rate` | `0.001` | Adam LR; rarely needs changing |
| `--test` | `None` | Path to a single image — score it after training and print the result |

### Expected output

```
Training Custom Thumbnail Quality Scorer
============================================================
Loaded 85 training samples
  - Good examples: 62
  - Bad examples: 23

Epoch [1/20]
  Train Loss: 0.3421
  Val Loss:   0.3012
  Saved best model

... (training continues)

Epoch [20/20]
  Train Loss: 0.0842
  Val Loss:   0.1123

Training complete
  Model saved to: models/your_category.pth
```

Aim for **val loss between 0.08 and 0.15** at the final epoch. Lower is possible but risks overfitting; higher means the model did not learn the signal well.

### Register the model with the app

The CLI trainer writes the `.pth` file but does not register it in `config/categories.json`. To make the category visible in the UI:

1. Launch `config.bat` and add the category through `/#config` (point it at the existing `.pth` and training folder). The UI auto-fills.
2. Or edit `config/categories.json` directly — see [`CONFIG.md`](CONFIG.md) for the schema.

### Testing a trained model

```bash
python src/training.py \
  --training-data training_data/your-category/ \
  --output models/your_category.pth \
  --test sample_frame.jpg
```

Output:

```
Testing on: sample_frame.jpg
  Quality score: 0.89
  HIGH quality (model likes this)
```

---

## Quick-start checklist

- [ ] Collected 50+ good examples, 20+ bad examples
- [ ] Organized into `training_data/<category>/good/` and `.../bad/`
- [ ] Launched `config.bat` and opened `/#config`
- [ ] Clicked "Add Category", filled in ID, label, icon, folder path
- [ ] Clicked "Train" and watched SSE progress
- [ ] Verified category appears on main page
- [ ] Ran extraction on a test video, reviewed top-N frames
- [ ] Iterated: added misclassified frames to `good/` or `bad/`, retrained

Your trained models live in `models/<category>.pth` and travel with the repo — portable to any machine with Python 3.10+ and the requirements installed.
