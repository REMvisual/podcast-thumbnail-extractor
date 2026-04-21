# Sample Frame Licenses

The 8 frames in `sample_frames/` are used only to build `banner.png`.

All frames sourced from **Lorem Picsum** (<https://picsum.photos/>), which serves curated random photos under the
[Unsplash License](https://unsplash.com/license) — free for commercial and non-commercial use, no attribution required.

| File | Source URL (seed) | License |
|---|---|---|
| `sample_01.jpg` | `https://picsum.photos/seed/pte1/480/360` | Unsplash License |
| `sample_02.jpg` | `https://picsum.photos/seed/pte2/480/360` | Unsplash License |
| `sample_03.jpg` | `https://picsum.photos/seed/pte3/480/360` | Unsplash License |
| `sample_04.jpg` | `https://picsum.photos/seed/pte4/480/360` | Unsplash License |
| `sample_05.jpg` | `https://picsum.photos/seed/pte5/480/360` | Unsplash License |
| `sample_06.jpg` | `https://picsum.photos/seed/pte6/480/360` | Unsplash License |
| `sample_07.jpg` | `https://picsum.photos/seed/pte7/480/360` | Unsplash License |
| `sample_08.jpg` | `https://picsum.photos/seed/pte8/480/360` | Unsplash License |

Regenerate with:
```bash
for i in 1 2 3 4 5 6 7 8; do
  curl -sL -o "tools/sample_frames/sample_0${i}.jpg" "https://picsum.photos/seed/pte${i}/480/360"
done
```
