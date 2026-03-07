import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np


def load_images_from_folder(folder_path: str) -> List[str]:
    """
    Return a sorted list of image file paths inside the given folder.

    This supports batch processing of all images in the input directory.
    """
    supported_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

    if not os.path.isdir(folder_path):
        raise ValueError(f"Input folder does not exist or is not a directory: {folder_path}")

    image_paths: List[str] = []
    for name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, name)
        if not os.path.isfile(full_path):
            continue
        _, ext = os.path.splitext(name)
        if ext.lower() in supported_ext:
            image_paths.append(os.path.abspath(full_path))

    image_paths.sort()
    return image_paths


def find_vertical_whitespace(image: np.ndarray) -> List[Tuple[int, int]]:
    """
    Detect continuous horizontal whitespace regions that will be used as
    vertical panel separators.

    A row is considered whitespace if at least 97% of its pixels are "white"
    (grayscale value >= 240). Consecutive whitespace rows of height >= 30
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
    white_mask = gray >= 240

    # Fraction of white pixels per row (vectorized for speed)
    white_fraction_per_row = white_mask.mean(axis=1)

    # Boolean mask of rows that are mostly whitespace
    whitespace_rows = white_fraction_per_row >= 0.97

    separators: List[Tuple[int, int]] = []

    min_separator_height = 30

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

    A pixel is considered content if its grayscale value is < 240.
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

    non_white_mask = gray < 240
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
    min_row_density = 0.01  # at least 1% of pixels in the row are non-white
    min_col_density = 0.005  # at least 0.5% of pixels in the column are non-white

    valid_rows = np.where(row_density >= min_row_density)[0]
    valid_cols = np.where(col_density >= min_col_density)[0]

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

    white_mask_cropped = cropped_gray >= 240
    white_fraction_rows = white_mask_cropped.mean(axis=1)

    # Move top_index down while rows are >= 97% white.
    top_index = 0
    while top_index < ch and white_fraction_rows[top_index] >= 0.97:
        top_index += 1

    # Move bottom_index up while rows are >= 97% white.
    bottom_index = ch - 1
    while bottom_index >= top_index and white_fraction_rows[bottom_index] >= 0.97:
        bottom_index -= 1

    if bottom_index < top_index:
        return None

    final_cropped = cropped[top_index : bottom_index + 1, :, :]

    # Enforce minimum height for a valid panel.
    min_panel_height = 150
    final_h = final_cropped.shape[0]
    if final_h < min_panel_height:
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
) -> None:
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
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"[WARN] Failed to read image: {image_path}")
        return

    height, width = image.shape[:2]
    if debug:
        print(f"[INFO] Loaded image '{os.path.basename(image_path)}' with size {width}x{height}")

    separators = find_vertical_whitespace(image)
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
            f"[INFO] Detected {len(separators)} separator bands, "
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
        "--debug",
        action="store_true",
        help="Enable debug mode to visualize detected panel boundaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_folder = os.path.abspath(args.input_folder)
    output_folder = os.path.abspath(args.output_folder)

    os.makedirs(output_folder, exist_ok=True)

    image_paths = load_images_from_folder(input_folder)

    if not image_paths:
        print(f"[INFO] No images found in folder: {input_folder}")
        return

    print(
        f"Panel extractor initialized.\n"
        f"  Input folder : {input_folder}\n"
        f"  Output folder: {output_folder}\n"
        f"  Debug mode   : {args.debug}\n"
        f"  Images found : {len(image_paths)}"
    )

    for idx, image_path in enumerate(image_paths, start=1):
        if args.debug:
            print(f"[INFO] Processing image {idx}/{len(image_paths)}: {os.path.basename(image_path)}")
        process_image(image_path, output_folder, debug=args.debug)


if __name__ == "__main__":
    main()

