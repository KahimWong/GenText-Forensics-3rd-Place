"""Mask-to-box and visualization helpers."""

from __future__ import annotations

import shutil
import struct
import zlib
from collections import deque
from pathlib import Path
from typing import Any

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

Mask = Any


def _allow_large_pillow_images() -> None:
    try:
        from PIL import Image
    except ModuleNotFoundError:
        return
    Image.MAX_IMAGE_PIXELS = None


def load_binary_mask(mask_path: str | Path | None, image_size: tuple[int, int] | None = None) -> Mask:
    if mask_path is None:
        if image_size is None:
            raise ValueError("image_size is required when mask_path is None")
        width, height = image_size
        if np is not None:
            return np.zeros((height, width), dtype=bool)
        return [[False for _ in range(width)] for _ in range(height)]
    try:
        from PIL import Image
    except ModuleNotFoundError:
        return _load_png_mask_stdlib(Path(mask_path))
    else:
        _allow_large_pillow_images()
        with Image.open(mask_path) as mask_image:
            gray = mask_image.convert("L")
            if np is not None:
                return np.asarray(gray, dtype=np.uint8) != 0
            width, height = gray.size
            pixels = gray.load()
            return [[bool(pixels[x, y]) for x in range(width)] for y in range(height)]


def _load_png_mask_stdlib(mask_path: Path) -> Mask:
    data = mask_path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("Pillow is required for non-PNG mask files")
    offset = 8
    width = height = bit_depth = color_type = None
    compressed = bytearray()
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk[:10])
        elif chunk_type == b"IDAT":
            compressed.extend(chunk)
        elif chunk_type == b"IEND":
            break
    if width is None or height is None or bit_depth != 8 or color_type not in {0, 2, 6}:
        raise RuntimeError("Pillow is required for this PNG mask encoding")
    channels = {0: 1, 2: 3, 6: 4}[color_type]
    row_bytes = width * channels
    raw = zlib.decompress(bytes(compressed))
    rows: list[bytearray] = []
    pos = 0
    previous = bytearray(row_bytes)
    for _ in range(height):
        filter_type = raw[pos]
        pos += 1
        scanline = bytearray(raw[pos : pos + row_bytes])
        pos += row_bytes
        recon = bytearray(row_bytes)
        for i, value in enumerate(scanline):
            left = recon[i - channels] if i >= channels else 0
            up = previous[i]
            up_left = previous[i - channels] if i >= channels else 0
            if filter_type == 0:
                recon[i] = value
            elif filter_type == 1:
                recon[i] = (value + left) & 0xFF
            elif filter_type == 2:
                recon[i] = (value + up) & 0xFF
            elif filter_type == 3:
                recon[i] = (value + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                predictor = _paeth(left, up, up_left)
                recon[i] = (value + predictor) & 0xFF
            else:
                raise RuntimeError(f"Unsupported PNG filter: {filter_type}")
        rows.append(recon)
        previous = recon
    mask = [[any(row[x * channels : (x + 1) * channels]) for x in range(width)] for row in rows]
    if np is not None:
        return np.asarray(mask, dtype=bool)
    return mask


def _paeth(left: int, up: int, up_left: int) -> int:
    p = left + up - up_left
    pa = abs(p - left)
    pb = abs(p - up)
    pc = abs(p - up_left)
    if pa <= pb and pa <= pc:
        return left
    if pb <= pc:
        return up
    return up_left


def connected_component_boxes(mask: Mask, min_area: int = 256, max_boxes: int = 20) -> list[dict]:
    if np is not None and cv2 is not None:
        mask_array = np.asarray(mask, dtype=np.uint8)
        if mask_array.size == 0:
            return []
        mask_array = (mask_array != 0).astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_array, connectivity=4)
        components = []
        for label_idx in range(1, num_labels):
            x, y, w, h, area = [int(value) for value in stats[label_idx]]
            if area >= min_area:
                components.append({"box": [x, y, x + w, y + h], "area": area})
        components.sort(key=lambda item: item["area"], reverse=True)
        return components[:max_boxes]

    height = len(mask)
    width = len(mask[0]) if height else 0
    visited = [[False for _ in range(width)] for _ in range(height)]
    components: list[dict] = []
    for y in range(height):
        for x0 in range(width):
            if visited[y][x0] or not mask[y][x0]:
                continue
            queue: deque[tuple[int, int]] = deque([(x0, y)])
            visited[y][x0] = True
            min_x = max_x = x0
            min_y = max_y = y
            area = 0
            while queue:
                x, yy = queue.popleft()
                area += 1
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, yy)
                max_y = max(max_y, yy)
                for nx, ny in ((x - 1, yy), (x + 1, yy), (x, yy - 1), (x, yy + 1)):
                    if 0 <= nx < width and 0 <= ny < height and not visited[ny][nx] and mask[ny][nx]:
                        visited[ny][nx] = True
                        queue.append((nx, ny))
            if area >= min_area:
                components.append({"box": [min_x, min_y, max_x + 1, max_y + 1], "area": area})
    components.sort(key=lambda item: item["area"], reverse=True)
    return components[:max_boxes]

from PIL import ImageFont
font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=28)

def render_red_box_overlay(
    image_path: str | Path,
    boxes: list[dict],
    output_path: str | Path,
    line_width: int = 5,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        return output
    try:
        from PIL import Image, ImageDraw
    except ModuleNotFoundError:
        shutil.copyfile(image_path, output)
        return output
    _allow_large_pillow_images()
    with Image.open(image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for idx, component in enumerate(boxes, 1):
        x1, y1, x2, y2 = component["box"]
        area = (x2 - x1) * (y2 - y1)
        label = f"{idx:02d}"
        draw.rectangle((x1, y1, x2, y2), outline="red", width=line_width)
        draw.text((x1, max(0, y1 - 30)), label, fill="red", font=font)
    canvas.save(output)
    return output
