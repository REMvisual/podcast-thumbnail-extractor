# Training Image Sources

A curated list of free, legally-safe sources for collecting training images, plus notes on licenses, API setup, and cropping.

Use these alongside [`TRAINING.md`](TRAINING.md), which explains how to turn the images into a trained category.

---

## Legal

Before you collect anything, understand what you are allowed to train on. Model weights trained on copyrighted images retain a derivative signal — ship those weights publicly and you inherit the license problem.

- **Pexels License** — free for commercial use, no attribution required. Modification allowed. See <https://www.pexels.com/license/>.
- **Unsplash License** — free for commercial and non-commercial use. Cannot sell unmodified images. See <https://unsplash.com/license>.
- **Openverse** (<https://openverse.org>) — aggregates Creative Commons and public-domain media from Wikimedia, Flickr, and others. **Check each individual result's license** — some require attribution (CC BY), some disallow commercial use (CC BY-NC), and some are public domain (CC0). Openverse surfaces the license on every result.
- **Creative Commons licenses in general** — CC0 and CC BY are safe; CC BY-NC and CC BY-ND need care. Read <https://creativecommons.org/licenses/> before using.

**Do NOT train on:**
- Stock photos from **iStock, Shutterstock, Getty, Adobe Stock, Alamy** — proprietary licenses, off limits without a paid license.
- Google Image Search results — most are copyrighted by default.
- Instagram / Pinterest posts unless you own them or have explicit permission.
- Screenshots from paid software tutorials where the training materials are behind a license wall.

**When in doubt, use Pexels or Unsplash.** Both have clean commercial licenses and enormous catalogues.

---

## Goal

Collect **50+ good** and **20+ bad** examples per category. Bigger sets train better models, but quality matters more than quantity.

Folder layout (also in [`TRAINING.md`](TRAINING.md)):

```
training_data/your-category/
├── good/   (50+ images)
└── bad/    (20+ images)
```

---

## UI / Interface category examples

Use cases: node graphs, software interfaces, code editors, dashboards.

### TouchDesigner node graphs

**Look for:** clear connections, readable labels, parameter panels visible. **Avoid:** zoomed-out or motion-blurred frames.

- [Official Derivative docs](https://docs.derivative.ca/) — interface screenshots
- [Steve Zafeiriou's TD tutorials](https://stevezafeiriou.com/touchdesigner-tutorial-for-beginners/)
- [The NODE Institute courses](https://thenodeinstitute.org/)
- [AllTouchDesigner community](https://alltd.org/)
- YouTube: search "TouchDesigner tutorial" and pause-screenshot clean node graphs

### Comfy UI workflows

**Look for:** visible nodes, clear connections, good contrast. **Avoid:** overlapping nodes, cluttered layouts.

- [ComfyUI official examples](https://comfyanonymous.github.io/ComfyUI_examples/)
- [ComfyUI Examples GitHub](https://github.com/comfyanonymous/ComfyUI_examples)
- [RunComfy workflows](https://www.runcomfy.com/comfyui-workflows)
- [OpenArt templates](https://openart.ai/workflows/templates)
- Reddit: r/comfyui

### Resolume / VJ software interfaces

**Look for:** layer panels, effect parameters visible, UI in focus. **Avoid:** frames where output preview dominates.

- [Resolume official site](https://www.resolume.com/)
- [DocOptic tutorials](https://docoptic.com/tag/resolume/)
- YouTube: "Resolume tutorial"

### Cinema 4D / Blender viewports

**Look for:** viewport with clear geometry, timeline and tool panels. **Avoid:** full-screen renders with no UI.

- [Maxon C4D docs](https://help.maxon.net/c4d/en-us/)
- [Blender official docs](https://docs.blender.org/)
- Greyscalegorilla tutorials
- YouTube: "Cinema 4D tutorial", "Blender tutorial"

### Unreal Engine blueprints

- [Unreal Engine official docs](https://docs.unrealengine.com/)
- YouTube: "Unreal Engine blueprint tutorial"
- Reddit: r/unrealengine

### Code editors

**Look for:** readable code, good syntax highlighting, clear file structure.

- [VS Code themes gallery](https://vscodethemes.com/)
- GitHub README files (many show editor screenshots)
- Dev.to tutorial articles

### Bad UI examples (20-30 images)

Include on purpose:
- Cluttered or overlapping windows
- Text too small to read
- Poor colour contrast
- Blurry or motion-blurred frames
- Mixed UI + art content (bad signal for a UI-focused model)

Easiest source: take them yourself — pause YouTube videos at the *worst* moments and screenshot.

---

## Art / output category examples

Use cases: renders, motion graphics, generative art, VJ output, fullscreen visuals.

### Generative art

**Look for:** fullscreen visuals, no UI chrome, high visual impact.

- [Genuary Collection](https://genuary.art/) — yearly creative-coding showcase
- [Awesome Creative Coding (GitHub)](https://github.com/terkelg/awesome-creative-coding)
- [fx(hash)](https://www.fxhash.xyz/) — generative-art platform (check each artist's permission before use)
- [Art Blocks](https://www.artblocks.io/) — NFT generative art (permission varies — prefer CC-licensed pieces)

### Motion graphics

- [Behance](https://www.behance.net/) — filter by "Motion Graphics"
- [Dribbble](https://dribbble.com/tags/motion_graphics)
- [Vimeo Staff Picks](https://vimeo.com/channels/staffpicks)

Most individual works are copyrighted. Prefer platforms where the artist explicitly offers images under a permissive license, or contact them for permission.

### 3D renders

- [ArtStation](https://www.artstation.com/) — most work is copyrighted; use only as inspiration unless the artist has published under a permissive license
- [Blender Cloud Gallery](https://cloud.blender.org/) — Blender Foundation demo files (often open content)

### Stock photography (safe commercial use)

- **[Pexels](https://www.pexels.com/)** — huge free library, clean license
- **[Unsplash](https://unsplash.com/)** — curated, high quality, clean license
- **[Openverse](https://openverse.org/)** — CC aggregator (verify each result)
- **[Pixabay](https://pixabay.com/)** — free, clean licenses
- **[Wikimedia Commons](https://commons.wikimedia.org/)** — public domain and CC

### Bad art examples (20-30 images)

Include:
- Frames where UI is still visible (defeats the "fullscreen output" signal)
- Uncropped social-media posts (with username and controls still showing)
- Low-resolution or pixelated renders
- Work-in-progress screenshots
- Badly cropped images leaving window borders

---

## How to collect

### Method 1: Manual download

Right-click → Save As from source pages directly into `training_data/<category>/good/` or `.../bad/`. Slowest but highest quality control — you see every image before you keep it.

### Method 2: Pexels API (bulk download from Pexels)

Pexels has a free API that returns royalty-free images by keyword. The extractor's built-in downloader (`src/downloader.py`) uses it.

**Get an API key:**

1. Sign up at <https://www.pexels.com/api/>
2. Copy your API key (free tier: 200 requests/hour, 20,000/month — plenty for training)

**Set the key as an environment variable** (never hardcode it into a file that might be committed):

Windows (cmd):
```cmd
set PEXELS_API_KEY=your_key_here
```

Windows (PowerShell):
```powershell
$env:PEXELS_API_KEY = "your_key_here"
```

Unix / macOS (bash, zsh):
```bash
export PEXELS_API_KEY="your_key_here"
```

Make it permanent by adding the line to `~/.bashrc`, `~/.zshrc`, or Windows System Environment Variables. The app reads `PEXELS_API_KEY` from the environment at startup.

**Run the downloader:**

```bash
python src/downloader.py --query "abstract art" --count 50 --output training_data/art/good/
```

Tune the query for your category. For "bad" examples, search the same topic but add filters like `low-quality`, `cluttered`, or manually pick the worst-ranked results.

### Method 3: Screenshot from video

For UI screenshots specifically, pause tutorial videos on YouTube / Vimeo at a clear moment and screenshot. Windows: `Win+Shift+S`. macOS: `Cmd+Shift+4`.

Highest quality control, zero licensing risk if you are screenshotting for training only (not redistribution), and you see exactly what the model will learn from.

---

## Cropping social-media posts

If you collect Instagram, Twitter, or Threads posts for your "art" category, crop out the platform chrome before adding to `good/`:

```
Before:                           After:
┌──────────────────────┐          ┌──────────────────────┐
│ @username            │          │                      │
├──────────────────────┤          │    JUST THE ART      │
│                      │          │                      │
│   YOUR ART           │    →     └──────────────────────┘
│                      │
├──────────────────────┤
│ ❤  💬  share        │
└──────────────────────┘
```

Use any image editor: Photoshop, GIMP, Affinity Photo, macOS Preview, Windows Photos. Batch tools like `ImageMagick` or `ffmpeg` can automate fixed-size crops across a folder.

**Bad examples deliberately include the chrome** — leave an Instagram header visible in a `bad/` image to teach the model "this is not the pattern I want."

---

## Quality checklist

Before saving an image to `good/`, ask:

**For UI categories:**
- [ ] Can you read the labels / text clearly?
- [ ] Is the interface the main subject?
- [ ] No art output visible in the frame?
- [ ] Good colour contrast?

**For art categories:**
- [ ] Fullscreen (no UI chrome)?
- [ ] If from social media, cropped to just the art?
- [ ] High visual quality?
- [ ] Matches the aesthetic you want the model to learn?

**For bad examples:**
- [ ] Clearly demonstrates what you do NOT want?
- [ ] Has obvious quality issues the model can learn to reject?
- [ ] 20+ examples per category?

---

## Recommended rhythm

If you split collection over a week:

- **Day 1-2**: UI good examples (~60-80 across multiple software types)
- **Day 3**: UI bad examples (~20-30)
- **Day 4-5**: Art good examples (~60-80 across multiple styles)
- **Day 6**: Art bad examples (~20-30)
- **Day 7**: Review, remove borderline cases, train

At ~2 minutes per image (find + evaluate + save), a category takes 2-3 hours of focused work.

---

## After collection

1. Verify folder counts: `ls training_data/<category>/good/ | wc -l` and same for `bad/`.
2. Sanity-check: open a random sample of 10 images in each folder. Is each one obviously correct for its label?
3. Launch `config.bat`, open `/#config`, add the category, click **Train**. See [`TRAINING.md`](TRAINING.md) for the full UI walkthrough.
4. Test the trained model on a real video. Iterate by moving misclassified frames between `good/` and `bad/` and retraining.
