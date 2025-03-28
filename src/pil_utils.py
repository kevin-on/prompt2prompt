import os
import textwrap
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw  # type: ignore


def create_image_grid(
    images: Union[np.ndarray, List[np.ndarray], List[List[np.ndarray]]],
    num_rows: Optional[int] = None,
    num_cols: Optional[int] = None,
    padding: int = 5,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    row_descs: Optional[List[str]] = None,
    col_descs: Optional[List[str]] = None,
    text_color: Tuple[int, int, int] = (0, 0, 0),
    font_size: int = 10,
    row_desc_max_width: int = 200,
    col_desc_max_height: int = 50,
) -> np.ndarray:
    """
    Creates a grid of images with optional row and column descriptions.

    Args:
        images: Input images as numpy arrays. Can be:
            - 1D list/array of images
            - 2D list/array of images
        num_rows: Optional number of rows for 1D input. If None, will be calculated.
        num_cols: Optional number of columns for 1D input. If None, will be calculated.
        padding: Padding between images in pixels
        background_color: Background color for the grid (RGB tuple)
        row_descs: Optional list of row descriptions
        col_descs: Optional list of column descriptions
        text_color: Color for description text (RGB tuple)
        font_size: Size of the text in pixels (default: 10)

    Returns:
        Combined image as numpy array
    """
    images_grid: List[List[np.ndarray]] = []
    if isinstance(images, np.ndarray):
        images_grid = [[images]]
    elif isinstance(images[0], np.ndarray):
        n_images = len(images)
        if num_rows is None and num_cols is None:
            num_cols = n_images
            num_rows = 1
        elif num_rows is None and num_cols is not None:
            num_rows = int(np.ceil(n_images / num_cols))
        elif num_cols is None and num_rows is not None:
            num_cols = int(np.ceil(n_images / num_rows))
        assert isinstance(num_rows, int)
        assert isinstance(num_cols, int)

        if num_rows * num_cols < n_images:
            raise ValueError(
                "num_rows * num_cols must be greater than or equal to n_images"
            )
        padded_images = images + [None] * (num_rows * num_cols - len(images))
        images_grid = [
            padded_images[i : i + num_cols]
            for i in range(0, len(padded_images), num_cols)
        ]  # type: ignore
    else:
        images_grid = images  # type: ignore

    img_height, img_width = images_grid[0][0].shape[:2]

    # Create a temporary image and draw object for text measurements
    temp_img = Image.new("RGB", (1, 1), background_color)
    draw = ImageDraw.Draw(temp_img)

    # Calculate required widths and heights for descriptions
    row_desc_width = padding
    col_desc_height = padding

    if row_descs:
        max_row_width = 0
        for desc in row_descs:
            wrapped_text = textwrap.fill(
                str(desc), width=max(1, row_desc_max_width // (font_size // 2))
            )
            text_bbox = draw.multiline_textbbox((0, 0), wrapped_text)
            text_width = text_bbox[2] - text_bbox[0]
            max_row_width = max(max_row_width, text_width)
        row_desc_width = min(row_desc_max_width, max_row_width + 2 * padding)

    if col_descs:
        max_col_height = 0
        for desc in col_descs:
            img_width = images_grid[0][0].shape[1]
            wrapped_text = textwrap.fill(
                str(desc), width=max(1, img_width // (font_size // 2))
            )
            text_bbox = draw.multiline_textbbox((0, 0), wrapped_text)
            text_height = text_bbox[3] - text_bbox[1]
            max_col_height = max(max_col_height, text_height)
        col_desc_height = min(col_desc_max_height, max_col_height + 2 * padding)

    # Calculate final grid dimensions
    grid_width = (
        img_width * len(images_grid[0])
        + padding * (len(images_grid[0]) - 1)
        + row_desc_width
    )
    grid_height = (
        img_height * len(images_grid)
        + padding * (len(images_grid) - 1)
        + col_desc_height
    )

    # Create the actual output image
    output = np.full(
        (int(grid_height), int(grid_width), 3), background_color, dtype=np.uint8
    )
    output_pil = Image.fromarray(output)
    draw = ImageDraw.Draw(output_pil)

    # Draw column descriptions
    if col_descs:
        for j, desc in enumerate(col_descs[: len(images_grid[0])]):
            x = row_desc_width + j * (img_width + padding)
            img_width_j = images_grid[0][j].shape[1]
            wrapped_text = textwrap.fill(
                str(desc), width=max(1, img_width_j // (font_size // 2))
            )
            text_bbox = draw.multiline_textbbox((0, 0), wrapped_text)
            text_width = text_bbox[2] - text_bbox[0]
            x = x + (img_width - text_width) // 2
            # Center vertically in the available height
            y = (col_desc_height - (text_bbox[3] - text_bbox[1])) // 2
            draw.multiline_text((x, y), wrapped_text, fill=text_color, align="center")

    # Draw row descriptions
    if row_descs:
        for i, desc in enumerate(row_descs[: len(images_grid)]):
            y = col_desc_height + i * (img_height + padding)
            wrapped_text = textwrap.fill(
                str(desc), width=max(1, row_desc_max_width // (font_size // 2))
            )
            text_bbox = draw.multiline_textbbox((0, 0), wrapped_text)
            text_height = text_bbox[3] - text_bbox[1]
            text_width = text_bbox[2] - text_bbox[0]
            y = y + (img_height - text_height) // 2
            x = row_desc_width - text_width - padding
            draw.multiline_text((x, y), wrapped_text, fill=text_color, align="right")

    # Place images in grid
    for i, row in enumerate(images_grid):
        for j, img in enumerate(row):
            if img is not None:
                y = i * (img_height + padding) + col_desc_height
                x = j * (img_width + padding) + row_desc_width
                output_pil_array = np.array(output_pil)
                output_pil_array[y : y + img_height, x : x + img_width] = img
                output_pil = Image.fromarray(output_pil_array)

    return np.array(output_pil)


def save_image(image: np.ndarray, path: str):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(image).save(path)


def batch_save_images(
    images: List[np.ndarray],
    base_dir: str,
    prefix: str = "",
    suffix: str = "",
    format: str = "png",
) -> List[str]:
    """
    Saves a list of images to disk with optional prefix and suffix.

    Args:
        images: Single image as numpy array or list of images
        base_dir: Base directory path where images will be saved
        prefix: Optional prefix for filenames (default: "")
        suffix: Optional suffix for filenames (default: "")
        format: Image format to save as (default: "png")

    Returns:
        List of saved file paths
    """
    os.makedirs(base_dir, exist_ok=True)

    saved_paths = []
    for idx, img in enumerate(images):
        filename = f"{prefix}{idx:03d}{suffix}.{format}"
        filepath = os.path.join(base_dir, filename)
        Image.fromarray(img).save(filepath)
        saved_paths.append(filepath)

    return saved_paths


if __name__ == "__main__":
    print("Running PIL utils tests...")
    images = [
        # red
        np.full((100, 100, 3), (255, 0, 0), dtype=np.uint8),
        # green
        np.full((100, 100, 3), (0, 255, 0), dtype=np.uint8),
        # blue
        np.full((100, 100, 3), (0, 0, 255), dtype=np.uint8),
    ] * 2

    image_grid = create_image_grid(
        images,
        row_descs=["row 1", "row 2 - A very long row description. This is a test."],
        col_descs=[
            "col 1",
            "col 2 - A very long column description. This is a test.",
            "col 3",
        ],
        num_rows=2,
        padding=10,
    )
    Image.fromarray(image_grid).save("image_grid_1.png")

    image_grid = create_image_grid(
        images,
        row_descs=["row 1", "row 2"],
        num_cols=2,
        padding=10,
    )
    Image.fromarray(image_grid).save("image_grid_2.png")
