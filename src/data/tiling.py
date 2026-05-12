# src/data/tiling.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.windows import Window
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


@dataclass
class TileInfo:
    tile_path: str
    source_path: str
    tile_size: int
    row: int
    col: int
    x_offset: int
    y_offset: int


def tile_image(
    image_path: str,
    output_dir: str,
    tile_size: int,
    overlap: int = 0,
) -> List[TileInfo]:
    """
    Tile a large image into fixed-size patches.
    Tiles that do not fit exactly at the right/bottom edge are skipped.
    Supports JPEG/PNG (via Pillow) and GeoTIFF (via rasterio).
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if overlap >= tile_size:
        raise ValueError(f"overlap ({overlap}) must be less than tile_size ({tile_size})")

    if image_path.suffix.lower() in (".tif", ".tiff") and HAS_RASTERIO:
        return _tile_geotiff(image_path, output_dir, tile_size, overlap)
    return _tile_pil(image_path, output_dir, tile_size, overlap)


def _tile_pil(
    image_path: Path, output_dir: Path, tile_size: int, overlap: int
) -> List[TileInfo]:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    stride = tile_size - overlap
    tiles: List[TileInfo] = []
    row = 0
    for y in range(0, h - tile_size + 1, stride):
        col = 0
        for x in range(0, w - tile_size + 1, stride):
            crop = img.crop((x, y, x + tile_size, y + tile_size))
            name = f"{image_path.stem}_r{row}_c{col}{image_path.suffix}"
            out = output_dir / name
            crop.save(out)
            tiles.append(TileInfo(str(out), str(image_path), tile_size, row, col, x, y))
            col += 1
        row += 1
    return tiles


def _tile_geotiff(
    image_path: Path, output_dir: Path, tile_size: int, overlap: int
) -> List[TileInfo]:
    stride = tile_size - overlap
    tiles: List[TileInfo] = []
    with rasterio.open(image_path) as src:
        w, h = src.width, src.height
        n_bands = min(src.count, 3)
        row = 0
        for y in range(0, h - tile_size + 1, stride):
            col = 0
            for x in range(0, w - tile_size + 1, stride):
                window = Window(x, y, tile_size, tile_size)
                data = src.read(list(range(1, n_bands + 1)), window=window).astype(np.float32)
                data_min, data_max = data.min(), data.max()
                data = ((data - data_min) / (data_max - data_min + 1e-8) * 255).astype(np.uint8)
                data = data.transpose(1, 2, 0)
                if n_bands == 1:
                    data = np.stack([data[:, :, 0]] * 3, axis=-1)
                # GeoTIFF tiles saved as JPEG for compactness; PIL path preserves source extension
                out = output_dir / f"{image_path.stem}_r{row}_c{col}.jpg"
                Image.fromarray(data).save(out)
                tiles.append(TileInfo(str(out), str(image_path), tile_size, row, col, x, y))
                col += 1
            row += 1
    return tiles
