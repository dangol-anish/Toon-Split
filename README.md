## Webtoon / Manhwa Panel Extractor

This project extracts **clean content panels from long vertical webtoon/manhwa pages** using classic image processing (OpenCV + NumPy), without any AI models.

The script:

- Detects **horizontal whitespace gaps** as panel separators.
- Optionally uses several **fallback heuristics** (text clusters, low-content valleys, content zones) for very tall pages.
- Crops each panel to its **tight content bounding box**, removing left/right/top/bottom margins as much as possible.
- Filters out tiny or empty strips.
- Saves each detected panel as a separate image.
- Produces a **batch summary** and a **“problem page” list** at the end.

---

## 1. Requirements & Installation

### Python & dependencies

- Python 3.8+ recommended.
- Required packages:
  - `opencv-python`
  - `numpy`

Install from the project root:

```bash
cd /Users/anishdangol/Documents/work/Toon-Split
python -m pip install --upgrade pip
python -m pip install opencv-python numpy
```

You can also use a virtual environment if you prefer.

---

## 2. Input & Output Layout

### Input

- You provide a folder containing **long, vertical images** (webtoon/manhwa pages).
- Supported extensions: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, `.tiff`.
- Example structure:

```text
Toon-Split/
  panel_extractor.py
  raw_pages/
    001.png
    002.png
    003.png
    ...
```

### Output

- For each input image, panels are saved to an output folder, e.g.:

```text
panel_output/
  001_panel_001.png
  001_panel_002.png
  002_panel_001.png
  ...
```

- In **debug mode**, a `debug/` subfolder is also created:

```text
panel_output/debug/
  001_debug.png   # red lines showing detected separators
  002_debug.png
```

---

## 3. How the Algorithm Works (High Level)

1. **Load images** from the input folder.
2. For each image:
   - Convert to **grayscale**.
   - For each row, compute the **fraction of white pixels** (gray ≥ `gray-threshold`).
   - Rows with white fraction ≥ `white-ratio` are tagged as **whitespace**.
   - Consecutive whitespace rows of height ≥ `min-whitespace-height` form **separator bands**.
3. **Split into vertical strips** between separators (top → first gap, between gaps, last gap → bottom).
4. For each strip:
   - Compute **non-white mask** (gray < `gray-threshold`).
   - Reject strips with almost no non-white pixels (both absolute and relative thresholds).
   - Compute content densities per row/column and crop to a **tight bounding box**, trimming sparse edge pixels.
   - Run a second pass from top/bottom to remove any **almost-all-white rows** (white fraction ≥ `white-ratio`).
   - Reject cropped panels shorter than `min-panel-height`.
5. For **very tall images with no whitespace separators**, additional fallback strategies may attempt to guess separators based on:
   - **Bright text / speech-bubble clusters**.
   - **Low-content valleys** (low edge density).
   - **High-content zones** (clusters of strong edge activity) and cuts between them.
6. Each valid panel is saved as `baseName_panel_XXX.png`.
7. At the end, a **batch summary** is printed, listing:
   - Total images and panels.
   - Average panels per image.
   - Tall images that produced only 0–1 panels.
   - Images that produced 0 panels.

---

## 4. Command-Line Usage

From the project root:

```bash
python panel_extractor.py --input-folder <INPUT_DIR> --output-folder <OUTPUT_DIR> [options...]
```

Required arguments:

- `--input-folder`  
  Path to the folder with your source images (e.g. `raw_pages`).

- `--output-folder`  
  Path to the folder where extracted panels will be saved (created if missing).

Optional arguments (all have sensible defaults tuned for typical digital webtoon pages):

- `--white-ratio FLOAT`  
  - Default: `0.97`  
  - A row is considered **whitespace** if the fraction of white pixels (gray ≥ `gray-threshold`) is at least this value.

- `--min-whitespace-height INT`  
  - Default: `30`  
  - Minimum number of consecutive whitespace rows to form a separator band.

- `--gray-threshold INT`  
  - Default: `240`  
  - Grayscale values ≥ this are treated as **white**; `<` are treated as content.

- `--min-panel-height INT`  
  - Default: `150`  
  - Panels shorter than this (in pixels) are discarded as too small.

- `--debug`  
  - Enables debug logging and saves **debug images** with red separator lines in `OUTPUT/debug/`.

- `--workers INT`  
  - Default: `1` (no parallelism).  
  - Number of worker threads for **parallel per-image processing**.  
  - Example: `--workers 4` on a 4-core CPU.

If you do **not** pass any of these options, the script runs with all default values.

---

## 5. Batch Summary & Problem Pages

After processing all images, the script prints a summary like:

```text
=== Batch summary ===
Processed images : 120
Total panels     : 430
Avg panels/image : 3.58

Images that are tall but produced only 0–1 panels:
  - 004.png (height=9200, panels=1)

Images that produced 0 panels:
  - 099.png (height=5120)
```

Use this to quickly identify **problem pages** that might need manual review or special treatment.

---

## 6. Recommended Workflow

1. **Prepare input folder**
   - Put your long webtoon pages into a folder, e.g. `raw_pages/`.

2. **Run a small test batch with debug on**

   ```bash
   python panel_extractor.py \
     --input-folder raw_pages \
     --output-folder panel_output_test \
     --debug
   ```

   - Inspect sample panels under `panel_output_test/`.
   - Inspect debug overlays under `panel_output_test/debug/` to see where cuts occur.

3. **Tune thresholds if necessary**
   - If gutters are thinner or pages are noisier, you might try, for example:

   ```bash
   python panel_extractor.py \
     --input-folder raw_pages \
     --output-folder panel_output_tuned \
     --white-ratio 0.95 \
     --min-whitespace-height 20 \
     --gray-threshold 235 \
     --min-panel-height 140 \
     --debug
   ```

4. **Scale up for full chapters with parallel workers**

   Once you’re happy with the quality:

   ```bash
   python panel_extractor.py \
     --input-folder raw_pages \
     --output-folder panel_output \
     --workers 4
   ```

   Adjust `--workers` based on your CPU.

---

## 7. Core Command (Minimal, Recommended Defaults)

For most use cases, once your input folder is ready and dependencies are installed, you can simply run:

```bash
python panel_extractor.py --input-folder raw_pages --output-folder panel_output --workers 4 --debug
```

- `raw_pages` is your folder with source images.
- `panel_output` will contain the extracted panels.
- `--workers 4` parallelizes across 4 threads (tune to your machine).
- `--debug` gives you boundary overlays and extra logs so you can quickly verify segmentation quality.

