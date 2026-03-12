import argparse
import os
from typing import List, Tuple, Iterable

import cv2
import numpy as np

# Default configuration values (can be overridden via CLI).
WHITE_ROW_FRACTION_DEFAULT = 0.97
MIN_WHITESPACE_HEIGHT_DEFAULT = 30
NON_WHITE_GRAY_THRESHOLD_DEFAULT = 240
MIN_PANEL_HEIGHT_DEFAULT = 150
ROW_DENSITY_MIN_DEFAULT = 0.01
COL_DENSITY_MIN_DEFAULT = 0.005
HEIGHT_FALLBACK_THRESHOLD_DEFAULT = 2500

# Runtime configuration (initialized from defaults, updated in main()).
WHITE_ROW_FRACTION = WHITE_ROW_FRACTION_DEFAULT
MIN_WHITESPACE_HEIGHT = MIN_WHITESPACE_HEIGHT_DEFAULT
NON_WHITE_GRAY_THRESHOLD = NON_WHITE_GRAY_THRESHOLD_DEFAULT
MIN_PANEL_HEIGHT = MIN_PANEL_HEIGHT_DEFAULT
ROW_DENSITY_MIN = ROW_DENSITY_MIN_DEFAULT
COL_DENSITY_MIN = COL_DENSITY_MIN_DEFAULT
HEIGHT_FALLBACK_THRESHOLD = HEIGHT_FALLBACK_THRESHOLD_DEFAULT


def _iter_image_paths_under_root(root_folder: str) -> Iterable[tuple[str, str]]:
    """
    Yield (abs_path, rel_path) for supported images under the given root.

    This supports both:
    - Flat input folders (images directly inside root_folder)
    - Chapter folders (subfolders under root_folder containing images)
    """
    supported_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

    if not os.path.isdir(root_folder):
        raise ValueError(
            f"Input folder does not exist or is not a directory: {root_folder}"
        )

    root_folder = os.path.abspath(root_folder)

    for dirpath, _, filenames in os.walk(root_folder):
        for name in filenames:
            _, ext = os.path.splitext(name)
            if ext.lower() not in supported_ext:
                continue
            abs_path = os.path.abspath(os.path.join(dirpath, name))
            rel_path = os.path.relpath(abs_path, root_folder)
            yield abs_path, rel_path

def load_images_from_folder(folder_path: str) -> List[str]:
    """
    Backwards-compatible helper that returns only absolute image paths.

    Prefer using `_iter_image_paths_under_root()` for chapter-aware processing.
    """
    items = list(_iter_image_paths_under_root(folder_path))
    items.sort(key=lambda x: x[1])
    return [abs_path for abs_path, _ in items]


def find_vertical_whitespace(image: np.ndarray) -> List[Tuple[int, int]]:
    """
    Detect continuous horizontal whitespace regions that will be used as
    vertical panel separators.

    A row is considered whitespace if at least WHITE_ROW_FRACTION of its pixels
    are "white" (grayscale value >= NON_WHITE_GRAY_THRESHOLD). Consecutive
    whitespace rows of height >= MIN_WHITESPACE_HEIGHT
    form a separator band.

    Returns a list of (start_row, end_row) tuples indicating whitespace bands,
    where both indices are inclusive and 0-based.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    height, width = gray.shape[:2]
    if height == 0 or width == 0:
        return []

    # True where pixel is considered "white"
    white_mask = gray >= NON_WHITE_GRAY_THRESHOLD

    # Fraction of white pixels per row (vectorized for speed)
    white_fraction_per_row = white_mask.mean(axis=1)

    # Boolean mask of rows that are mostly whitespace
    whitespace_rows = white_fraction_per_row >= float(WHITE_ROW_FRACTION)

    separators: List[Tuple[int, int]] = []

    min_separator_height = int(MIN_WHITESPACE_HEIGHT)

    in_run = False
    run_start = 0

    for row_idx, is_whitespace in enumerate(whitespace_rows):
        if is_whitespace:
            if not in_run:
                in_run = True
                run_start = row_idx
        else:
            if in_run:
                run_end = row_idx - 1
                run_height = run_end - run_start + 1
                if run_height >= min_separator_height:
                    separators.append((run_start, run_end))
                in_run = False

    # Handle run reaching the bottom of the image
    if in_run:
        run_end = height - 1
        run_height = run_end - run_start + 1
        if run_height >= min_separator_height:
            separators.append((run_start, run_end))

    return separators


def find_low_content_separators(image: np.ndarray) -> List[Tuple[int, int]]:
    """
    Fallback separator detection for very tall images that lack clear white gaps.

    Uses per-row edge density (via Canny) to build a vertical content profile,
    then finds low-content bands (valleys) that can serve as approximate panel
    separators.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    height, width = gray.shape[:2]
    if height == 0 or width == 0:
        return []

    # Detect edges; parameters kept relatively modest for speed.
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Edge density per row.
    edge_counts = edges.sum(axis=1) / 255  # edges are 0 or 255
    edge_density = edge_counts.astype(np.float32) / float(width)

    # Smooth the 1D profile to avoid reacting to tiny local variations.
    window = 25  # rows
    if window > 1:
        kernel = np.ones(window, dtype=np.float32) / float(window)
        smoothed = np.convolve(edge_density, kernel, mode="same")
    else:
        smoothed = edge_density

    # Low-content threshold: rows significantly below the typical activity.
    # Use a quantile-based threshold so it adapts to each image.
    low_threshold = float(np.quantile(smoothed, 0.25))
    low_content_rows = smoothed <= low_threshold

    separators: List[Tuple[int, int]] = []
    min_gap_height = 40  # minimum height of a low-content band

    in_run = False
    run_start = 0

    for row_idx, is_low in enumerate(low_content_rows):
        if is_low:
            if not in_run:
                in_run = True
                run_start = row_idx
        else:
            if in_run:
                run_end = row_idx - 1
                run_height = run_end - run_start + 1
                if run_height >= min_gap_height:
                    separators.append((run_start, run_end))
                in_run = False

    if in_run:
        run_end = height - 1
        run_height = run_end - run_start + 1
        if run_height >= min_gap_height:
            separators.append((run_start, run_end))

    return separators


def find_content_zone_separators(image: np.ndarray) -> List[Tuple[int, int]]:
    """
    Fallback splitter for long images with content spread across the height.

    Builds a 1D content profile (edge density per row), finds "high-content"
    zones, and places separators in the quieter regions between zones.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    height, width = gray.shape[:2]
    if height == 0 or width == 0:
        return []

    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    edge_counts = edges.sum(axis=1) / 255
    edge_density = edge_counts.astype(np.float32) / float(width)

    # Smooth the profile.
    window = 25
    if window > 1:
        kernel = np.ones(window, dtype=np.float32) / float(window)
        smoothed = np.convolve(edge_density, kernel, mode="same")
    else:
        smoothed = edge_density

    # High-content threshold using a quantile so it adapts per image.
    high_threshold = float(np.quantile(smoothed, 0.6))
    high_rows = smoothed >= high_threshold

    zones: List[Tuple[int, int]] = []
    min_zone_height = 150

    in_zone = False
    zone_start = 0

    for row_idx, is_high in enumerate(high_rows):
        if is_high:
            if not in_zone:
                in_zone = True
                zone_start = row_idx
        else:
            if in_zone:
                zone_end = row_idx - 1
                zone_height = zone_end - zone_start + 1
                if zone_height >= min_zone_height:
                    zones.append((zone_start, zone_end))
                in_zone = False

    if in_zone:
        zone_end = height - 1
        zone_height = zone_end - zone_start + 1
        if zone_height >= min_zone_height:
            zones.append((zone_start, zone_end))

    if len(zones) < 2:
        return []

    # Derive separators between adjacent zones.
    separators: List[Tuple[int, int]] = []
    min_gap_between_zones = 200
    half_band = 10

    for i in range(len(zones) - 1):
        z1_start, z1_end = zones[i]
        z2_start, z2_end = zones[i + 1]

        if z2_start - z1_end < min_gap_between_zones:
            continue

        mid = int((z1_end + z2_start) / 2)
        start_row = max(0, mid - half_band)
        end_row = min(height - 1, mid + half_band)
        if end_row > start_row:
            separators.append((start_row, end_row))

    return separators


def find_text_cluster_separators(image: np.ndarray) -> List[Tuple[int, int]]:
    """
    Fallback splitter based on bright text / speech-bubble regions.

    Detects bright connected components (likely text or bubbles), groups them
    into vertical clusters, and places separators between those clusters.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    height, width = gray.shape[:2]
    if height == 0 or width == 0:
        return []

    # Threshold to capture bright text / bubbles.
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    # Slight morphological closing to connect characters into larger blobs.
    kernel = np.ones((3, 3), dtype=np.uint8)
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_closed)

    # Collect vertical spans of candidate text/bubble components.
    spans: List[Tuple[int, int]] = []
    min_area = 300
    min_dim = 15

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        if w < min_dim or h < min_dim:
            continue
        top = y
        bottom = y + h - 1
        spans.append((top, bottom))

    if len(spans) < 2:
        return []

    # Merge nearby spans into clusters (content zones driven by text).
    spans.sort(key=lambda s: s[0])
    clusters: List[Tuple[int, int]] = []
    cluster_start, cluster_end = spans[0]
    merge_gap = 250  # max vertical gap within one text cluster

    for top, bottom in spans[1:]:
        if top - cluster_end <= merge_gap:
            cluster_end = max(cluster_end, bottom)
        else:
            clusters.append((cluster_start, cluster_end))
            cluster_start, cluster_end = top, bottom
    clusters.append((cluster_start, cluster_end))

    if len(clusters) < 2:
        return []

    # Place separators between clusters, in the quieter space between them.
    separators: List[Tuple[int, int]] = []
    min_gap_between_clusters = 200
    half_band = 10

    for i in range(len(clusters) - 1):
        c1_start, c1_end = clusters[i]
        c2_start, c2_end = clusters[i + 1]

        if c2_start - c1_end < min_gap_between_clusters:
            continue

        mid = int((c1_end + c2_start) / 2)
        start_row = max(0, mid - half_band)
        end_row = min(height - 1, mid + half_band)
        if end_row > start_row:
            separators.append((start_row, end_row))

    return separators


def split_into_strips(
    image: np.ndarray, separators: List[Tuple[int, int]]
) -> List[np.ndarray]:
    """
    Split the input image into vertical strips using the whitespace separators.

    Each strip corresponds to a candidate panel region before content cropping.
    """
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return []

    if not separators:
        # No separators found; treat the whole image as a single strip.
        return [image.copy()]

    # Ensure separators are sorted by their vertical position.
    separators_sorted = sorted(separators, key=lambda s: s[0])

    strips: List[np.ndarray] = []

    # Top segment: from top of image to just above the first separator.
    prev_bottom = 0
    for start_row, end_row in separators_sorted:
        top = prev_bottom
        bottom = max(start_row - 1, top)
        if bottom >= top:
            strip = image[top : bottom + 1, :, :]
            strips.append(strip)
        prev_bottom = end_row + 1

    # Bottom segment: from after the last separator to bottom of image.
    if prev_bottom < height:
        strip = image[prev_bottom:height, :, :]
        strips.append(strip)

    return strips


def crop_to_content(strip: np.ndarray) -> np.ndarray | None:
    """
    Crop a strip to the minimal bounding box of non-white pixels.

    A pixel is considered content if its grayscale value is < NON_WHITE_GRAY_THRESHOLD.
    Strips that contain almost no non-white pixels, or whose cropped
    height is below a minimum threshold, are rejected.

    Returns the cropped panel as a new image, or None if it should be
    discarded.
    """
    if strip.size == 0:
        return None

    if strip.ndim == 3:
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    else:
        gray = strip

    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return None

    non_white_mask = gray < NON_WHITE_GRAY_THRESHOLD
    non_white_count = int(non_white_mask.sum())
    total_pixels = h * w

    # Reject strips with almost no non-white pixels.
    # This uses both an absolute and relative threshold.
    if non_white_count < 50 or (non_white_count / float(total_pixels)) < 0.001:
        return None

    # Compute non-white density per row and per column to aggressively
    # trim sparse edge pixels (e.g. from slanted lines or speech bubbles).
    row_non_white_counts = non_white_mask.sum(axis=1)
    col_non_white_counts = non_white_mask.sum(axis=0)

    row_density = row_non_white_counts.astype(np.float32) / float(w)
    col_density = col_non_white_counts.astype(np.float32) / float(h)

    # Thresholds tuned to remove most white margins while allowing slight
    # cropping into content, which is acceptable for this use case.
    valid_rows = np.where(row_density >= float(ROW_DENSITY_MIN))[0]
    valid_cols = np.where(col_density >= float(COL_DENSITY_MIN))[0]

    if valid_rows.size == 0 or valid_cols.size == 0:
        return None

    top = int(valid_rows[0])
    bottom = int(valid_rows[-1])
    left = int(valid_cols[0])
    right = int(valid_cols[-1])

    if bottom < top or right < left:
        return None

    cropped = strip[top : bottom + 1, left : right + 1]

    # Second pass: trim any almost-all-white rows at the very top and bottom
    # of the cropped panel to remove residual white bands around bubbles, etc.
    cropped_gray = gray[top : bottom + 1, left : right + 1]
    ch, cw = cropped_gray.shape[:2]

    if ch == 0 or cw == 0:
        return None

    white_mask_cropped = cropped_gray >= NON_WHITE_GRAY_THRESHOLD
    white_fraction_rows = white_mask_cropped.mean(axis=1)

    # Move top_index down while rows are >= WHITE_ROW_FRACTION white.
    top_index = 0
    while top_index < ch and white_fraction_rows[top_index] >= float(WHITE_ROW_FRACTION):
        top_index += 1

    # Move bottom_index up while rows are >= WHITE_ROW_FRACTION white.
    bottom_index = ch - 1
    while bottom_index >= top_index and white_fraction_rows[bottom_index] >= float(WHITE_ROW_FRACTION):
        bottom_index -= 1

    if bottom_index < top_index:
        return None

    final_cropped = cropped[top_index : bottom_index + 1, :, :]

    # Enforce minimum height for a valid panel.
    final_h = final_cropped.shape[0]
    if final_h < int(MIN_PANEL_HEIGHT):
        return None

    return final_cropped


def save_panels(
    panels: List[np.ndarray],
    base_name: str,
    output_folder: str,
) -> None:
    """
    Save each panel image to the output folder using the naming pattern
    {base_name}_panel_{index:03d}.png.
    """
    if not panels:
        return

    os.makedirs(output_folder, exist_ok=True)

    for idx, panel in enumerate(panels, start=1):
        filename = f"{base_name}_panel_{idx:03d}.png"
        out_path = os.path.join(output_folder, filename)

        success = cv2.imwrite(out_path, panel)
        if not success:
            print(f"[WARN] Failed to write panel image: {out_path}")


def save_debug_boundaries(
    image: np.ndarray,
    separators: List[Tuple[int, int]],
    output_folder: str,
    base_name: str,
) -> None:
    """
    Save a debug visualization of detected panel boundaries.

    Draws horizontal lines at the center of each whitespace separator band
    over the original image and writes the result into a 'debug' subfolder
    of the main output folder.
    """
    if not separators:
        return

    debug_folder = os.path.join(output_folder, "debug")
    os.makedirs(debug_folder, exist_ok=True)

    debug_img = image.copy()
    height, width = debug_img.shape[:2]

    # Draw a red horizontal line at the center of each separator band.
    for start_row, end_row in separators:
        center_row = int((start_row + end_row) / 2)
        center_row = max(0, min(center_row, height - 1))
        cv2.line(
            debug_img,
            (0, center_row),
            (width - 1, center_row),
            color=(0, 0, 255),
            thickness=2,
        )

    debug_path = os.path.join(debug_folder, f"{base_name}_debug.png")
    success = cv2.imwrite(debug_path, debug_img)
    if not success:
        print(f"[WARN] Failed to write debug image: {debug_path}")


def process_image(
    image_path: str,
    output_folder: str,
    debug: bool = False,
    rel_path: str | None = None,
) -> dict | None:
    """
    Orchestrate the full pipeline for a single image:
    - load image
    - find whitespace separators
    - split into strips
    - crop to content
    - filter and save panels

    The detailed implementation will be added step by step. Currently this:
    - loads the image
    - finds vertical whitespace separators
    - splits into vertical strips
    - crops each strip to its content region and filters out invalid panels
    Returns a small stats dictionary with keys:
    - 'image_path'
    - 'width'
    - 'height'
    - 'panel_count'
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"[WARN] Failed to read image: {image_path}")
        return None

    height, width = image.shape[:2]
    if debug:
        shown = rel_path if rel_path else os.path.basename(image_path)
        print(f"[INFO] Loaded image '{shown}' with size {width}x{height}")

    separators_whitespace = find_vertical_whitespace(image)

    # Fallbacks for very tall images without clear white gutters.
    separators = list(separators_whitespace)
    height_threshold_for_fallback = 2500
    if not separators and height >= height_threshold_for_fallback:
        # First, try separators based on text / speech-bubble clusters.
        text_cluster_separators = find_text_cluster_separators(image)
        if text_cluster_separators:
            separators = text_cluster_separators

    if not separators and height >= height_threshold_for_fallback:
        # Next, try separators based on low-content valleys.
        low_content_separators = find_low_content_separators(image)
        if low_content_separators:
            separators = low_content_separators

    # If still no separators and the image is tall, try splitting based on
    # high-content zones (clusters of activity) and cut between them.
    if not separators and height >= height_threshold_for_fallback:
        zone_separators = find_content_zone_separators(image)
        if zone_separators:
            separators = zone_separators

    strips = split_into_strips(image, separators)

    panels: list[np.ndarray] = []
    for strip in strips:
        panel = crop_to_content(strip)
        if panel is not None:
            panels.append(panel)

    # Derive a base name for output files from the original image name.
    stem, _ = os.path.splitext(os.path.basename(image_path))
    base_name = stem

    if debug:
        print(
            f"[INFO] Detected {len(separators_whitespace)} whitespace separator bands, "
            f"yielding {len(strips)} vertical strips, "
            f"and {len(panels)} content panels after cropping/filtering."
        )

        # Save a debug visualization of separator positions.
        save_debug_boundaries(
            image=image,
            separators=separators,
            output_folder=output_folder,
            base_name=base_name,
        )

    if panels:
        save_panels(panels, base_name=base_name, output_folder=output_folder)
        if debug:
            print(f"[INFO] Saved {len(panels)} panels for base '{base_name}'.")

    return {
        "image_path": image_path,
        "rel_path": rel_path,
        "width": width,
        "height": height,
        "panel_count": len(panels),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for batch panel extraction."""
    parser = argparse.ArgumentParser(
        description="Extract content panels from vertical webtoon/manhwa images."
    )
    parser.add_argument(
        "--input-folder",
        required=True,
        help="Path to the folder containing input images.",
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        help="Path to the folder where extracted panels will be saved.",
    )
    parser.add_argument(
        "--white-ratio",
        type=float,
        default=WHITE_ROW_FRACTION_DEFAULT,
        help=(
            "Row considered whitespace if fraction of white pixels >= this value "
            f"(default: {WHITE_ROW_FRACTION_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--min-whitespace-height",
        type=int,
        default=MIN_WHITESPACE_HEIGHT_DEFAULT,
        help=(
            "Minimum height (in rows) of a continuous whitespace band to be "
            f"treated as a separator (default: {MIN_WHITESPACE_HEIGHT_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--gray-threshold",
        type=int,
        default=NON_WHITE_GRAY_THRESHOLD_DEFAULT,
        help=(
            "Grayscale threshold for white pixels; values >= this are treated "
            f"as white (default: {NON_WHITE_GRAY_THRESHOLD_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--min-panel-height",
        type=int,
        default=MIN_PANEL_HEIGHT_DEFAULT,
        help=(
            "Minimum height (in rows) for a cropped panel to be kept "
            f"(default: {MIN_PANEL_HEIGHT_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to visualize detected panel boundaries.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of worker threads for parallel processing. "
            "Use 1 to disable parallelism (default: 1)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Update runtime configuration from CLI arguments.
    global WHITE_ROW_FRACTION, MIN_WHITESPACE_HEIGHT, NON_WHITE_GRAY_THRESHOLD
    global MIN_PANEL_HEIGHT

    WHITE_ROW_FRACTION = args.white_ratio
    MIN_WHITESPACE_HEIGHT = args.min_whitespace_height
    NON_WHITE_GRAY_THRESHOLD = args.gray_threshold
    MIN_PANEL_HEIGHT = args.min_panel_height

    input_folder = os.path.abspath(args.input_folder)
    output_folder = os.path.abspath(args.output_folder)

    os.makedirs(output_folder, exist_ok=True)

    image_items = list(_iter_image_paths_under_root(input_folder))
    image_items.sort(key=lambda x: x[1])

    if not image_items:
        print(f"[INFO] No images found in folder: {input_folder}")
        return

    print(
        f"Panel extractor initialized.\n"
        f"  Input folder : {input_folder}\n"
        f"  Output folder: {output_folder}\n"
        f"  Debug mode   : {args.debug}\n"
        f"  Images found : {len(image_items)}"
    )

    stats: list[dict] = []

    if args.workers <= 1:
        # Sequential processing.
        for idx, (image_path, rel_path) in enumerate(image_items, start=1):
            chapter = rel_path.split(os.sep, 1)[0] if os.sep in rel_path else None
            per_image_output = (
                os.path.join(output_folder, chapter) if chapter else output_folder
            )
            if args.debug:
                print(
                    f"[INFO] Processing image {idx}/{len(image_items)}: {rel_path}"
                )
            image_stats = process_image(
                image_path, per_image_output, debug=args.debug, rel_path=rel_path
            )
            if image_stats is not None:
                stats.append(image_stats)
    else:
        # Parallel processing using a thread pool.
        max_workers = max(1, args.workers)
        print(f"[INFO] Running with {max_workers} worker threads.")

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(
                    process_image,
                    abs_path,
                    (os.path.join(output_folder, rel_path.split(os.sep, 1)[0])
                     if os.sep in rel_path
                     else output_folder),
                    args.debug,
                    rel_path,
                ): abs_path
                for abs_path, rel_path in image_items
            }
            for future in as_completed(future_to_path):
                image_stats = future.result()
                if image_stats is not None:
                    stats.append(image_stats)

    if not stats:
        print("[INFO] No images were successfully processed.")
        return

    # Batch statistics.
    total_images = len(stats)
    total_panels = sum(s["panel_count"] for s in stats)
    avg_panels = total_panels / float(total_images)

    tall_single_panel = [
        s
        for s in stats
        if s["panel_count"] <= 1 and s["height"] >= HEIGHT_FALLBACK_THRESHOLD
    ]
    zero_panel = [s for s in stats if s["panel_count"] == 0]

    print("\n=== Batch summary ===")
    print(f"Processed images : {total_images}")
    print(f"Total panels     : {total_panels}")
    print(f"Avg panels/image : {avg_panels:.2f}")

    if tall_single_panel:
        print("\nImages that are tall but produced only 0–1 panels:")
        for s in tall_single_panel:
            rel = s.get("rel_path") or os.path.relpath(s["image_path"], input_folder)
            print(f"  - {rel} (height={s['height']}, panels={s['panel_count']})")

    if zero_panel:
        print("\nImages that produced 0 panels:")
        for s in zero_panel:
            rel = s.get("rel_path") or os.path.relpath(s["image_path"], input_folder)
            print(f"  - {rel} (height={s['height']})")


if __name__ == "__main__":
    main()

