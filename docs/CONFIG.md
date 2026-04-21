# Configuration Reference

## Category config: `config/categories.json`

Defines detection categories — each becomes a button on the main app.

### Schema

```json
{
  "version": 1,
  "categories": [
    {
      "id": "string",              // unique, lowercase + dashes (e.g. "screens-ui")
      "label": "string",            // human-readable (e.g. "Screens: UI")
      "icon": "string",             // emoji or short text (e.g. "💻")
      "detector": "face|quality|none",
      "model_path": "string | null",  // optional CNN weights path
      "training_data_path": "string | null",  // optional training data folder
      "enabled": true,
      "builtin": false              // true for `faces` and `everything`; user cannot delete these
    }
  ]
}
```

### Detector types

| Detector | Behavior |
|---|---|
| `face` | Runs OpenCV face detection + heuristic quality score. No CNN needed. |
| `quality` | Runs heuristic quality score; if `model_path` points to a `.pth`, also reranks with the CNN. |
| `none` | No filter — returns scene-sampled frames as-is. |

### Managing categories

- **Via UI:** `config.bat` opens the app with the category manager at `/#config`.
- **Via API:** see the `/api/categories` REST routes.
- **Via file:** edit `config/categories.json` directly; the app reads it on every request (no restart needed).

### Builtin categories

- `faces` (icon: 😊) — can be disabled but not deleted.
- `everything` (icon: 🎯) — can be disabled but not deleted.

Any other category (including `screens-ui`, `screens-art`) is user-managed and can be deleted from the UI.

### Adding a new category

1. Collect training examples: `training_data/<your-category>/good/` (50+ images) and `/bad/` (50+ images).
2. Add via UI: `config.bat` → Add Category → fill form.
3. Click "Train" — the CNN fine-tunes on your data.
4. The category appears on the main page; pick it when extracting.

See [`TRAINING.md`](TRAINING.md) for training details.

## Runtime environment variables

All runtime paths are env-var overridable — the defaults assume `cwd = repo root`.

| Variable | Default | Purpose |
|---|---|---|
| `THUMBNAIL_EXTRACTOR_UPLOAD_DIR` | `./uploads` | Video scratch |
| `THUMBNAIL_EXTRACTOR_OUTPUT_DIR` | `./outputs` | Thumbnail output |
| `THUMBNAIL_EXTRACTOR_MODEL_DIR` | `./models` | CNN model storage |
| `THUMBNAIL_EXTRACTOR_CONFIG_PATH` | `./config/categories.json` | Category config |
| `THUMBNAIL_EXTRACTOR_PORT` | `5000` | Flask port |

To run on port 8080 with outputs on an external drive:

**Windows:**
```
set THUMBNAIL_EXTRACTOR_PORT=8080
set THUMBNAIL_EXTRACTOR_OUTPUT_DIR=D:\thumbnails
run.bat
```

**Linux/macOS:**
```
THUMBNAIL_EXTRACTOR_PORT=8080 THUMBNAIL_EXTRACTOR_OUTPUT_DIR=/mnt/thumbnails python src/app.py
```
